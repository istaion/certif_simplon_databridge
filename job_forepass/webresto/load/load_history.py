"""
Chargement incrémental de history depuis la nouvelle gateway Webresto (data-lake).

Job forepass indépendant : ne dépend pas du package data_process / de l'API
IAnord. Objectif : pouvoir être déployé en un clic sur la data platform pour
basculer le chargement de history sur la nouvelle gateway, sans attendre la
mise en production de ce repo.

Contrairement à bankdetail, cette table n'a pas de soft delete côté webresto
(pas de champ deletedAt) : tout item renvoyé par l'API est actif.

Logique :
  - SELECT MAX(updated_at) sur la table cible → point de reprise
  - Fetch par tranches de CHUNK_DAYS jours via GET /findAll/history
    (updatedSince / updatedBefore / selects)
  - purge (même id_reg_history) puis bulk_insert, pour couvrir aussi bien
    les créations que les mises à jour
  - Réconciliation hard-delete (filet de sécurité) : la gateway n'expose pas
    les hard deletes — la ligne disparaît simplement. À chaque exécution, on
    refetch systématiquement les RECONCILE_DAYS derniers jours et on supprime
    de Trino les id_reg_history de cette fenêtre absents de la réponse. Si la
    réponse est vide, on ne supprime rien (probable incident API plutôt
    qu'une semaine sans donnée).

PARAMS requis :
    BASE_URL              : host de la gateway (ex: "https://gateway.int.region-centre.ianord.fr"),
                            avec ou sans le suffixe "/data-lake" — les deux formes sont acceptées
    ENVIRONNEMENT_CLIENT  : ex "prodcentre"
    PREFIX_TABLE          : ex "wr_centre_"
    SECRET_KEY_WEBRESTO   : clé x-api-key
"""

import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from forepaas.core.settings import PARAMS
from forepaas.dwh import bulk_insert, connect

logger = logging.getLogger(__name__)

JOB_NAME = "history"

WEBRESTO_BASE_URL    = PARAMS["BASE_URL"]
ENVIRONNEMENT_CLIENT = PARAMS["ENVIRONNEMENT_CLIENT"]
PREFIX_TABLE         = PARAMS["PREFIX_TABLE"]
SECRET_KEY           = PARAMS["SECRET_KEY_WEBRESTO"]

dataset_cible = f"dwh/db_mg6jk45h_{ENVIRONNEMENT_CLIENT}/"
p = PREFIX_TABLE
TABLE = f"{p}history"

# La nouvelle gateway route sous /data-lake — accepte BASE_URL avec ou sans
# ce segment pour ne pas dépendre de la convention retenue au déploiement.
_BASE = WEBRESTO_BASE_URL.rstrip("/")
if _BASE.endswith("/data-lake"):
    _BASE = _BASE[: -len("/data-lake")]
_HISTORY_URL = f"{_BASE}/data-lake/findAll/history"

DEFAULT_START_DATE = "2022-08-01"
CHUNK_DAYS = 120
RECONCILE_DAYS = 7

SELECTS = "id,registrationId,event,createdAt,updatedAt"

_HEADERS = {"x-api-key": SECRET_KEY, "accept": "application/json"}
_RETRY_STATUSES = {502, 503, 504}
_RETRY_DELAYS = [5, 15, 30]


# ── Client gateway ───────────────────────────────────────────────────────────

def _fetch_api(updated_since: str, updated_before: str) -> list:
    params = {"updatedSince": updated_since, "updatedBefore": updated_before, "selects": SELECTS}
    last_error: Exception = RuntimeError("Aucune tentative effectuée")
    total_attempts = len(_RETRY_DELAYS) + 1

    for attempt, delay in enumerate([0] + _RETRY_DELAYS, start=1):
        if delay:
            logger.warning(f"[{JOB_NAME}] retry {attempt}/{total_attempts} dans {delay}s")
            time.sleep(delay)
        try:
            resp = requests.get(_HISTORY_URL, headers=_HEADERS, params=params, timeout=120)
            if resp.status_code in _RETRY_STATUSES:
                last_error = RuntimeError(f"HTTP {resp.status_code}")
                continue
            resp.raise_for_status()
            try:
                return resp.json() or []
            except ValueError as json_err:
                last_error = RuntimeError(
                    f"Réponse non-JSON (HTTP {resp.status_code}, "
                    f"body={resp.text[:200]!r}) : {json_err}"
                )
        except requests.exceptions.RequestException as e:
            last_error = e

    raise last_error


def _get_last_updated(source) -> str:
    try:
        result = source.query(f"SELECT MAX(updated_at) AS last_updated FROM {TABLE}")
        if isinstance(result, pd.DataFrame) and not result.empty:
            val = result.iloc[0]["last_updated"]
            if pd.notna(val):
                return val if isinstance(val, str) else val.strftime("%Y-%m-%d")
    except Exception as e:
        logger.warning(f"[{JOB_NAME}] impossible de lire MAX(updated_at) : {e}")
    return DEFAULT_START_DATE


# ── Transform ──────────────────────────────────────────────────────────────

def _transform(items: list) -> tuple[pd.DataFrame, list]:
    """Retourne (df, ids_to_purge) — pas de soft delete pour cette table,
    tous les items renvoyés par l'API sont actifs."""
    if not items:
        return pd.DataFrame(), []

    df = pd.DataFrame(items)
    ids_to_purge = df["id"].tolist()

    df["createdAt"] = pd.to_datetime(df["createdAt"], format="mixed")
    df["updatedAt"] = pd.to_datetime(df["updatedAt"], format="mixed")
    df["event"] = df["event"].astype(str).str.strip()

    df = df.rename(columns={
        "id": "id_reg_history",
        "registrationId": "registration_id",
        "createdAt": "created_at",
        "updatedAt": "updated_at",
    })[["id_reg_history", "registration_id", "event", "created_at", "updated_at"]]

    return df, ids_to_purge


def _purge(source, ids: list) -> None:
    if not ids:
        return
    ids_sql = ",".join(str(int(i)) for i in ids)
    source.query(f"DELETE FROM {TABLE} WHERE id_reg_history IN ({ids_sql})")


# ── Réconciliation hard-delete ───────────────────────────────────────────────

def _reconcile_hard_deletes(source, days: int = RECONCILE_DAYS) -> None:
    """
    Filet de sécurité : un hard delete côté webresto ne se voit jamais dans la
    réponse, la ligne disparaît simplement. On refetch la fenêtre des `days`
    derniers jours et on supprime de Trino les id_reg_history de cette
    fenêtre que l'API ne renvoie plus.
    """
    reconcile_start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    reconcile_end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(f"[{JOB_NAME}] Réconciliation hard-delete : {reconcile_start} → {reconcile_end}")
    items = _fetch_api(reconcile_start, reconcile_end)
    if not items:
        logger.warning(f"[{JOB_NAME}] Réconciliation : réponse API vide, aucune suppression")
        return

    api_ids = {item["id"] for item in items}

    result = source.query(
        f"SELECT id_reg_history FROM {TABLE} "
        f"WHERE updated_at >= TIMESTAMP '{reconcile_start} 00:00:00'"
    )
    if not isinstance(result, pd.DataFrame) or result.empty:
        logger.info(f"[{JOB_NAME}] Réconciliation : aucune ligne Trino sur la fenêtre")
        return

    trino_ids = set(result["id_reg_history"].dropna().astype(int).tolist())
    stale_ids = trino_ids - api_ids
    if not stale_ids:
        logger.info(f"[{JOB_NAME}] Réconciliation : aucune ligne obsolète")
        return

    logger.info(
        f"[{JOB_NAME}] Réconciliation : {len(stale_ids)} ligne(s) absente(s) de l'API "
        f"(hard delete) — suppression"
    )
    _purge(source, list(stale_ids))


# ── Entry point ────────────────────────────────────────────────────────────

def customfunc(event):
    logger.info(f"Démarrage du job '{JOB_NAME}'")
    t0 = time.time()

    source = connect(dataset_cible)

    last_updated = _get_last_updated(source)
    updated_before_end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    current_start = datetime.strptime(last_updated, "%Y-%m-%d")
    end_date = datetime.strptime(updated_before_end, "%Y-%m-%d")

    total = 0
    try:
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=CHUNK_DAYS), end_date)
            s = current_start.strftime("%Y-%m-%d")
            e = current_end.strftime("%Y-%m-%d")
            logger.info(f"[{JOB_NAME}] Période {s} → {e}")

            items = _fetch_api(s, e)
            if items:
                logger.info(f"[{JOB_NAME}] {len(items)} lignes brutes récupérées")
                df, ids_to_purge = _transform(items)
                _purge(source, ids_to_purge)
                if not df.empty:
                    bulk_insert(source, TABLE, df)
                    total += len(df)
                    logger.info(f"[{JOB_NAME}] {len(df)} lignes chargées ({s} → {e})")
            else:
                logger.info(f"[{JOB_NAME}] Aucune donnée pour cette période")

            current_start = current_end

        duration = round(time.time() - t0, 2)
        logger.info(f"[{JOB_NAME}] OK — {total} lignes chargées — {duration}s")
    except Exception as e:
        logger.error(f"[{JOB_NAME}] Erreur fatale : {type(e).__name__}: {e}")
        raise

    try:
        _reconcile_hard_deletes(source)
    except Exception as e:
        logger.error(f"[{JOB_NAME}] Réconciliation hard-delete échouée : {type(e).__name__}: {e}")

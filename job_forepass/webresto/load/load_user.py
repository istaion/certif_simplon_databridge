"""
Chargement incrémental de user depuis la nouvelle gateway Webresto (data-lake).

Job forepass indépendant : ne dépend pas du package data_process / de l'API
IAnord. Objectif : pouvoir être déployé en un clic sur la data platform pour
basculer le chargement de user sur la nouvelle gateway, sans attendre la
mise en production de ce repo.

Note : l'endpoint change de nom (/findAll/AllUsers → /findAll/user).

first_name/last_name/date_birth se déduisent de uniqueKeyName ("Prenom-Nom")
et uniqueKeyNameAndDateBirth ("Prenom-Nom-YYYY-MM-DD") — vérifié avec
l'utilisateur que le format anonymisé observé sur l'environnement de test
("beltran_6698") n'est pas représentatif de la prod, où le format réel
"Prenom-Nom" est conservé.

Logique :
  - SELECT MAX(updated_at) sur la table cible → point de reprise
  - Fetch par tranches de CHUNK_DAYS jours via GET /findAll/user
    (updatedSince / updatedBefore / selects)
  - Items avec deletedAt renseigné (soft delete côté webresto) :
      1. jamais insérés dans Trino
      2. purgés de Trino s'ils y étaient déjà
  - Items actifs : purge (même id_user) puis bulk_insert
  - Réconciliation hard-delete (filet de sécurité) : refetch systématique des
    RECONCILE_DAYS derniers jours, purge de ce que l'API ne renvoie plus.

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

JOB_NAME = "user"

WEBRESTO_BASE_URL    = PARAMS["BASE_URL"]
ENVIRONNEMENT_CLIENT = PARAMS["ENVIRONNEMENT_CLIENT"]
PREFIX_TABLE         = PARAMS["PREFIX_TABLE"]
SECRET_KEY           = PARAMS["SECRET_KEY_WEBRESTO"]

dataset_cible = f"dwh/db_mg6jk45h_{ENVIRONNEMENT_CLIENT}/"
p = PREFIX_TABLE
TABLE = f"{p}user"

# La nouvelle gateway route sous /data-lake — accepte BASE_URL avec ou sans
# ce segment pour ne pas dépendre de la convention retenue au déploiement.
_BASE = WEBRESTO_BASE_URL.rstrip("/")
if _BASE.endswith("/data-lake"):
    _BASE = _BASE[: -len("/data-lake")]
_USER_URL = f"{_BASE}/data-lake/findAll/user"

DEFAULT_START_DATE = "2022-08-01"
CHUNK_DAYS = 120
RECONCILE_DAYS = 7

SELECTS = "userId,createdAt,updatedAt,subgroupId,uniqueKeyName,uniqueKeyNameAndDateBirth"

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
            resp = requests.get(_USER_URL, headers=_HEADERS, params=params, timeout=120)
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
    """Retourne (df_active, ids_to_purge)."""
    if not items:
        return pd.DataFrame(), []

    df = pd.DataFrame(items)
    ids_to_purge = df["userId"].tolist()

    deleted_mask = df["deletedAt"].notna()
    n_deleted = int(deleted_mask.sum())
    if n_deleted:
        logger.info(f"[{JOB_NAME}] {n_deleted} soft-deleted détecté(s) — purge sans réinsertion")
    df = df[~deleted_mask].copy()

    if df.empty:
        return df, ids_to_purge

    df["subgroupId"] = df["subgroupId"].astype("Int64")
    df["createdAt"] = pd.to_datetime(df["createdAt"], format="mixed")
    df["updatedAt"] = pd.to_datetime(df["updatedAt"], format="mixed")

    # uniqueKeyName = "PRENOM-NOM" ; split at last hyphen to handle compound first names
    df["last_name"] = df["uniqueKeyName"].str.rsplit("-", n=1).str[-1]
    df["first_name"] = df["uniqueKeyName"].str.rsplit("-", n=1).str[0]
    # uniqueKeyNameAndDateBirth = "PRENOM-NOM-YYYY-MM-DD"
    df["date_birth"] = pd.to_datetime(
        df["uniqueKeyNameAndDateBirth"].str[-10:], format="%Y-%m-%d", errors="coerce"
    ).dt.date

    df = df[["userId", "createdAt", "updatedAt", "subgroupId", "first_name", "last_name", "date_birth"]]
    df = df.rename(columns={
        "userId": "id_user",
        "subgroupId": "id_subgroup",
        "updatedAt": "updated_at",
        "createdAt": "created_at",
    })

    return df, ids_to_purge


def _purge(source, ids: list) -> None:
    if not ids:
        return
    ids_sql = ",".join(str(int(i)) for i in ids)
    source.query(f"DELETE FROM {TABLE} WHERE id_user IN ({ids_sql})")


# ── Réconciliation hard-delete ───────────────────────────────────────────────

def _reconcile_hard_deletes(source, days: int = RECONCILE_DAYS) -> None:
    """
    Filet de sécurité : un hard delete côté webresto ne se voit ni via
    deletedAt ni via updatedAt, la ligne disparaît simplement de la réponse.
    On refetch la fenêtre des `days` derniers jours et on supprime de Trino
    les id_user de cette fenêtre que l'API ne renvoie plus.
    """
    reconcile_start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    reconcile_end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(f"[{JOB_NAME}] Réconciliation hard-delete : {reconcile_start} → {reconcile_end}")
    items = _fetch_api(reconcile_start, reconcile_end)
    if not items:
        logger.warning(f"[{JOB_NAME}] Réconciliation : réponse API vide, aucune suppression")
        return

    api_ids = {item["userId"] for item in items}

    result = source.query(
        f"SELECT id_user FROM {TABLE} "
        f"WHERE updated_at >= TIMESTAMP '{reconcile_start} 00:00:00'"
    )
    if not isinstance(result, pd.DataFrame) or result.empty:
        logger.info(f"[{JOB_NAME}] Réconciliation : aucune ligne Trino sur la fenêtre")
        return

    trino_ids = set(result["id_user"].dropna().astype(int).tolist())
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

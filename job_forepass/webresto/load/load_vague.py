"""
Chargement full reload de vague depuis la nouvelle gateway Webresto (data-lake).

Job forepass indépendant : ne dépend pas du package data_process / de l'API
IAnord. Objectif : pouvoir être déployé en un clic sur la data platform pour
basculer le chargement de vague sur la nouvelle gateway, sans attendre la
mise en production de ce repo.

Cette route est full reload (pas d'updatedSince/updatedBefore). Le champ
deletedAt est présent : les vagues soft-deleted côté webresto sont exclues
avant chargement. Si la réponse API est vide, la table n'est pas touchée.

PARAMS requis :
    BASE_URL              : host de la gateway (ex: "https://gateway.int.region-centre.ianord.fr"),
                            avec ou sans le suffixe "/data-lake" — les deux formes sont acceptées
    ENVIRONNEMENT_CLIENT  : ex "prodcentre"
    PREFIX_TABLE          : ex "wr_centre_"
    SECRET_KEY_WEBRESTO   : clé x-api-key
"""

import logging
import time

import pandas as pd
import requests
from forepaas.core.settings import PARAMS
from forepaas.dwh import bulk_insert, connect

logger = logging.getLogger(__name__)

JOB_NAME = "vague"

WEBRESTO_BASE_URL    = PARAMS["BASE_URL"]
ENVIRONNEMENT_CLIENT = PARAMS["ENVIRONNEMENT_CLIENT"]
PREFIX_TABLE         = PARAMS["PREFIX_TABLE"]
SECRET_KEY           = PARAMS["SECRET_KEY_WEBRESTO"]

dataset_cible = f"dwh/db_mg6jk45h_{ENVIRONNEMENT_CLIENT}/"
p = PREFIX_TABLE
TABLE = f"{p}vague"

# La nouvelle gateway route sous /data-lake — accepte BASE_URL avec ou sans
# ce segment pour ne pas dépendre de la convention retenue au déploiement.
_BASE = WEBRESTO_BASE_URL.rstrip("/")
if _BASE.endswith("/data-lake"):
    _BASE = _BASE[: -len("/data-lake")]
_VAGUE_URL = f"{_BASE}/data-lake/findAll/vagues"

SELECTS = "vagueId,name,startDate,endDate,SchoolYearId"

_HEADERS = {"x-api-key": SECRET_KEY, "accept": "application/json"}
_RETRY_STATUSES = {502, 503, 504}
_RETRY_DELAYS = [5, 15, 30]


# ── Client gateway ───────────────────────────────────────────────────────────

def _fetch_api() -> list:
    params = {"selects": SELECTS}
    last_error: Exception = RuntimeError("Aucune tentative effectuée")
    total_attempts = len(_RETRY_DELAYS) + 1

    for attempt, delay in enumerate([0] + _RETRY_DELAYS, start=1):
        if delay:
            logger.warning(f"[{JOB_NAME}] retry {attempt}/{total_attempts} dans {delay}s")
            time.sleep(delay)
        try:
            resp = requests.get(_VAGUE_URL, headers=_HEADERS, params=params, timeout=120)
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


# ── Transform ──────────────────────────────────────────────────────────────

def _transform(items: list) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()

    df = pd.DataFrame(items)

    if "deletedAt" in df.columns:
        deleted_mask = df["deletedAt"].notna()
        n_deleted = int(deleted_mask.sum())
        if n_deleted:
            logger.info(f"[{JOB_NAME}] {n_deleted} soft-deleted ignorés")
        df = df[~deleted_mask].copy()

    if df.empty:
        return df

    df["startDate"] = pd.to_datetime(df["startDate"]).dt.date
    df["endDate"] = pd.to_datetime(df["endDate"]).dt.date

    df = df[["vagueId", "name", "startDate", "endDate", "SchoolYearId"]]
    df = df.rename(columns={
        "vagueId": "id_vague",
        "startDate": "start_date",
        "endDate": "end_date",
        "SchoolYearId": "id_school_year",
    })
    df["name"] = df["name"].astype(str).str.strip()
    return df


# ── Entry point ────────────────────────────────────────────────────────────

def customfunc(event):
    logger.info(f"Démarrage du job '{JOB_NAME}'")
    t0 = time.time()

    source = connect(dataset_cible)

    items = _fetch_api()
    if not items:
        logger.warning(f"[{JOB_NAME}] Réponse API vide — table non modifiée")
        return

    df = _transform(items)
    if df.empty:
        logger.warning(f"[{JOB_NAME}] DataFrame vide après transform — table non modifiée")
        return

    source.query(f"DELETE FROM {TABLE}")
    bulk_insert(source, TABLE, df)

    duration = round(time.time() - t0, 2)
    logger.info(f"[{JOB_NAME}] OK — {len(df)} lignes chargées — {duration}s")

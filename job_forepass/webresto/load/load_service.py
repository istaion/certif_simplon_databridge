"""
Chargement full reload de service depuis la nouvelle gateway Webresto (data-lake).

Job forepass indépendant : ne dépend pas du package data_process / de l'API
IAnord. Remplace l'ancien import CSV (service.csv), qui était un pis-aller
tant que l'ancienne API renvoyait parfois du 400 sur cette route.

Cette route est full reload (pas d'updatedSince/updatedBefore). Le champ
deletedAt est présent : les services soft-deleted côté webresto sont exclus
avant chargement. Si la réponse API est vide (ou HTTP 400, historiquement
observé sur l'ancienne API), la table n'est pas touchée.

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

JOB_NAME = "service"

WEBRESTO_BASE_URL    = PARAMS["BASE_URL"]
ENVIRONNEMENT_CLIENT = PARAMS["ENVIRONNEMENT_CLIENT"]
PREFIX_TABLE         = PARAMS["PREFIX_TABLE"]
SECRET_KEY           = PARAMS["SECRET_KEY_WEBRESTO"]

dataset_cible = f"dwh/db_mg6jk45h_{ENVIRONNEMENT_CLIENT}/"
p = PREFIX_TABLE
TABLE = f"{p}service"

# La nouvelle gateway route sous /data-lake — accepte BASE_URL avec ou sans
# ce segment pour ne pas dépendre de la convention retenue au déploiement.
_BASE = WEBRESTO_BASE_URL.rstrip("/")
if _BASE.endswith("/data-lake"):
    _BASE = _BASE[: -len("/data-lake")]
_SERVICE_URL = f"{_BASE}/data-lake/findAll/services"

SELECTS = "organizationId,serviceId,label"

_HEADERS = {"x-api-key": SECRET_KEY, "accept": "application/json"}
_RETRY_STATUSES = {502, 503, 504}
_RETRY_DELAYS = [5, 15, 30]

# ── Constantes métier ──────────────────────────────────────────────────────────

_ORGA_CENTRE_DEMO  = [2, 3, 4, 5, 6, 7, 8]
_ORGA_93_EXCLUDED  = [127, 160, 161, 162]


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
            resp = requests.get(_SERVICE_URL, headers=_HEADERS, params=params, timeout=120)
            if resp.status_code in _RETRY_STATUSES:
                last_error = RuntimeError(f"HTTP {resp.status_code}")
                continue
            if resp.status_code == 400:
                logger.warning(f"[{JOB_NAME}] HTTP 400 traité comme liste vide")
                return []
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

    missing = [c for c in ["organizationId", "serviceId", "label"] if c not in df.columns]
    if missing:
        logger.error(f"[{JOB_NAME}] Colonnes manquantes dans la réponse : {missing}")
        return pd.DataFrame()

    if "deletedAt" in df.columns:
        deleted_mask = df["deletedAt"].notna()
        n_deleted = int(deleted_mask.sum())
        if n_deleted:
            logger.info(f"[{JOB_NAME}] {n_deleted} soft-deleted ignorés")
        df = df[~deleted_mask].copy()

    if df.empty:
        return df

    df = df[["organizationId", "serviceId", "label"]]

    if "centre" in ENVIRONNEMENT_CLIENT:
        df = df[~df["organizationId"].isin(_ORGA_CENTRE_DEMO)]
    if "93" in ENVIRONNEMENT_CLIENT:
        df = df[~df["organizationId"].isin(_ORGA_93_EXCLUDED)]

    df = df.rename(columns={
        "serviceId": "id_service",
        "organizationId": "id_organization",
    })
    df["label"] = df["label"].astype(str).str.strip()
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

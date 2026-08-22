"""
Chargement full reload de subgroup depuis la nouvelle gateway Webresto (data-lake).

Job forepass indépendant : ne dépend pas du package data_process / de l'API
IAnord. Objectif : pouvoir être déployé en un clic sur la data platform pour
basculer le chargement de subgroup sur la nouvelle gateway, sans attendre la
mise en production de ce repo.

Contrairement à l'ancienne API, la gateway ne renvoie plus d'objet `group`
imbriqué : groupId est un champ plat directement sur subgroup.

Cette route est full reload (pas d'updatedSince/updatedBefore). Le champ
deletedAt est présent : les subgroups soft-deleted côté webresto sont exclus
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

JOB_NAME = "subgroup"

WEBRESTO_BASE_URL    = PARAMS["BASE_URL"]
ENVIRONNEMENT_CLIENT = PARAMS["ENVIRONNEMENT_CLIENT"]
PREFIX_TABLE         = PARAMS["PREFIX_TABLE"]
SECRET_KEY           = PARAMS["SECRET_KEY_WEBRESTO"]

dataset_cible = f"dwh/db_mg6jk45h_{ENVIRONNEMENT_CLIENT}/"
p = PREFIX_TABLE
TABLE = f"{p}subgroup"

# La nouvelle gateway route sous /data-lake — accepte BASE_URL avec ou sans
# ce segment pour ne pas dépendre de la convention retenue au déploiement.
_BASE = WEBRESTO_BASE_URL.rstrip("/")
if _BASE.endswith("/data-lake"):
    _BASE = _BASE[: -len("/data-lake")]
_SUBGROUP_URL = f"{_BASE}/data-lake/findAll/subgroups"

SELECTS = "subgroupId,createdAt,updatedAt,label,groupId,acronym"

_HEADERS = {"x-api-key": SECRET_KEY, "accept": "application/json"}
_RETRY_STATUSES = {502, 503, 504}
_RETRY_DELAYS = [5, 15, 30]

# ── Constantes métier (facturation_type, environnement "centre" uniquement) ───

_SUBGROUP_INTERNES = {
    "interne externé", "interne",
    "internat 2j", "internat 3j", "internat 4j",
    "internat 5j", "interne aidés cfa", "interne cfa",
    "internat 6j", "interne week-end",
}
_SUBGROUP_TICKETS = {
    "ticket", "eleves externes", "apprentis cfa", "eleves exterieurs", "externe cfa",
    "eleves externes verdier", "ticket tarif tranche 1", "ticket tarif tranche 2",
    "ticket tarif tranche 3", "ticket tarif tranche 4",
    "eleves externes, apprentis cfa, eleves exterieurs",
}


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
            resp = requests.get(_SUBGROUP_URL, headers=_HEADERS, params=params, timeout=120)
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

def _facturation_type(label: str) -> str:
    label = (label or "").strip().lower()
    if label in _SUBGROUP_INTERNES:
        return "interne"
    if label in _SUBGROUP_TICKETS:
        return "ticket"
    return "autre"


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

    df["createdAt"] = pd.to_datetime(df["createdAt"], format="mixed")
    df["updatedAt"] = pd.to_datetime(df["updatedAt"], format="mixed")

    cols = ["subgroupId", "createdAt", "updatedAt", "label", "groupId", "acronym"]
    if "centre" in ENVIRONNEMENT_CLIENT:
        df["facturation_type"] = df["label"].apply(_facturation_type)
        cols.append("facturation_type")
    df = df[cols]

    df = df.rename(columns={
        "subgroupId": "id_subgroup",
        "groupId": "id_group",
        "createdAt": "created_at",
        "updatedAt": "updated_at",
    })
    for col in ["label", "acronym"]:
        df[col] = df[col].astype(str).str.strip()
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

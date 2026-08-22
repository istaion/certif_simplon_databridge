"""
Chargement initial complet de la table organization Webresto via appel direct à l'API.
Utilise bulk_insert ForePaaS — ne passe PAS par l'application databridge.

Pré-requis : les tables suivantes doivent être peuplées dans default_dataset avant ce job :
  - etablissement_detail      (POST /etablissement_detail)
  - ref_type_ips_corrections  (POST /ref_type_ips_corrections)
"""

import logging
import time

import pandas as pd
import requests
from forepaas.core.settings import PARAMS
from forepaas.dwh import bulk_insert, connect

logger = logging.getLogger(__name__)

# ── Paramètres ─────────────────────────────────────────────────────────────────

WEBRESTO_BASE_URL = PARAMS["BASE_URL"]
WEBRESTO_DATASET  = PARAMS["ENVIRONNEMENT_CLIENT"]
WEBRESTO_PREFIX   = PARAMS["PREFIX_TABLE"]
SECRET_KEY        = PARAMS["SECRET_KEY_WEBRESTO"]

dataset_cible   = f"dwh/db_mg6jk45h_{WEBRESTO_DATASET}/"
dataset_default = "dwh/default_dataset/"
p = WEBRESTO_PREFIX

_HEADERS = {
    "x-api-key": SECRET_KEY,
    "Content-Type": "application/json",
}

_RETRY_STATUSES = {502, 503, 504}
_RETRY_DELAYS   = [5, 15, 30]

# ── Constantes métier ──────────────────────────────────────────────────────────

_ORGA_CENTRE_DEMO  = [2, 3, 4, 5, 6, 7, 8]
_ORGA_93_EXCLUDED  = [127, 160, 161, 162]


# ── Client API Webresto ────────────────────────────────────────────────────────

def _fetch_api(endpoint: str, method: str = "GET", body: dict = None) -> list:
    url = WEBRESTO_BASE_URL.rstrip("/") + endpoint
    last_error: Exception = RuntimeError("Aucune tentative effectuée")

    for attempt, delay in enumerate([0] + _RETRY_DELAYS, start=1):
        if delay:
            logger.warning(f"[{method}] {endpoint} — retry {attempt} dans {delay}s")
            time.sleep(delay)
        try:
            if method == "GET":
                resp = requests.get(url, headers=_HEADERS, params=body or {}, timeout=120)
            else:
                resp = requests.post(url, headers=_HEADERS, json=body or {}, timeout=120)
            if resp.status_code not in _RETRY_STATUSES:
                resp.raise_for_status()
                return resp.json() or []
            last_error = RuntimeError(f"HTTP {resp.status_code}")
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt > len(_RETRY_DELAYS):
                raise

    raise last_error


# ── Lecture des tables de référence depuis le DWH ─────────────────────────────

def _load_ref_data(source_default) -> tuple:
    """
    Retourne (df_corrections, df_annuaire_dept) depuis default_dataset.
    df_corrections  : colonnes uai, type, ips — filtré pour l'environnement courant.
    df_annuaire_dept: colonnes uai, libelle_departement.
    """
    raw = source_default.query(
        "SELECT uai, type, ips, source FROM ref_type_ips_corrections"
    )
    if isinstance(raw, pd.DataFrame) and not raw.empty:
        df_corrections = raw[
            raw["source"].apply(lambda s: str(s) in WEBRESTO_DATASET)
        ][["uai", "type", "ips"]].copy()
    else:
        df_corrections = pd.DataFrame(columns=["uai", "type", "ips"])
        logger.warning("ref_type_ips_corrections : table vide ou non disponible")

    raw_dept = source_default.query(
        "SELECT DISTINCT uai, libelle_departement FROM etablissement_detail"
        " WHERE libelle_departement IS NOT NULL"
    )
    if isinstance(raw_dept, pd.DataFrame) and not raw_dept.empty:
        df_annuaire_dept = raw_dept
    else:
        df_annuaire_dept = pd.DataFrame(columns=["uai", "libelle_departement"])
        logger.warning("etablissement_detail : table vide ou non disponible")

    logger.info(
        f"Référence — corrections : {len(df_corrections)} lignes, "
        f"annuaire dept : {len(df_annuaire_dept)} établissements"
    )
    return df_corrections, df_annuaire_dept


# ── Transform organization ────────────────────────────────────────────────────

def _build_nom_ville(name: str, city: str) -> str:
    name = name.rstrip("- _")
    city_clean = city.lstrip("0123456789 -")
    if city_clean.lower() not in name.lower():
        name = f"{name} - {city_clean}"
    return name


def _transform_organization(
    df: pd.DataFrame,
    df_corrections: pd.DataFrame,
    df_annuaire_dept: pd.DataFrame,
) -> pd.DataFrame:
    if df.empty:
        return df

    cols = [
        "organizationId", "rne", "name",
        "city", "type", "department", "academy", "accessSoftware", "ips", "vague",
    ]
    df = df[cols]

    if "centre" in WEBRESTO_DATASET:
        df = df[~df["organizationId"].isin(_ORGA_CENTRE_DEMO)]
    if "93" in WEBRESTO_DATASET:
        df = df[~df["organizationId"].isin(_ORGA_93_EXCLUDED)]

    if not df_corrections.empty:
        mapping = df_corrections.set_index("uai")
        df["type"] = df["rne"].map(mapping["type"]).combine_first(df["type"])
        df["ips"]  = df["rne"].map(mapping["ips"]).combine_first(df["ips"])

    if "93" in WEBRESTO_DATASET:
        df["name"] = df.apply(
            lambda row: _build_nom_ville(str(row["name"]), str(row["city"])), axis=1
        )

    df = df.rename(columns={
        "organizationId": "id_organization",
        "accessSoftware":  "access_software",
    })

    if not df_annuaire_dept.empty:
        dept = df_annuaire_dept.rename(columns={"uai": "rne"}).drop_duplicates("rne")
        df = df.merge(dept[["rne", "libelle_departement"]], on="rne", how="left")
        df["department"] = df["libelle_departement"].combine_first(df["department"])
        df = df.drop(columns=["libelle_departement"])

    for col in ["access_software", "name", "city", "rne", "department", "type", "academy"]:
        df[col] = df[col].astype(str).str.strip()

    return df


# ── Sync organization ─────────────────────────────────────────────────────────

def _sync_organization(source, df_corrections: pd.DataFrame, df_annuaire_dept: pd.DataFrame) -> None:
    logger.info("Sync organization...")

    data = _fetch_api(
        "/findAll/organizations",
        method="GET",
        body={
            "isHideDemo": "true",
            "select": "organizationId, name, rne, city, accessSoftware, type, ips, academy, vague, department",
        },
    )

    if not data:
        logger.warning("organization : aucune donnée reçue depuis l'API")
        return

    df = pd.DataFrame(data)
    df = _transform_organization(df, df_corrections, df_annuaire_dept)

    if df.empty:
        logger.warning("organization : DataFrame vide après transform")
        return

    source.query(f"DELETE FROM {p}organization")
    bulk_insert(source, f"{p}organization", df)
    logger.info(f"organization : {len(df)} lignes insérées")


# ── Entry point ────────────────────────────────────────────────────────────────

def customfunc(event):
    source         = connect(dataset_cible)
    source_default = connect(dataset_default)

    df_corrections, df_annuaire_dept = _load_ref_data(source_default)

    _sync_organization(source, df_corrections, df_annuaire_dept)

    logger.info("Synchronisation Webresto full_reload terminée.")

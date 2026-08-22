"""
Chargement full reload de organization depuis la nouvelle gateway Webresto (data-lake).

Job forepass indépendant : ne dépend pas du package data_process / de l'API
IAnord. Objectif : pouvoir être déployé en un clic sur la data platform pour
basculer le chargement de organization sur la nouvelle gateway, sans attendre
la mise en production de ce repo.

Porte la même logique métier que l'ancien job_forepass/webresto/full_reload.py
(filtrage des orgas demo, corrections type/ips depuis les CSV de référence,
department depuis etablissement_detail, construction du nom pour "93") —
adaptée au nouveau contrat de la gateway (selects, deletedAt).

Cette route est full reload (pas d'updatedSince/updatedBefore). Le champ
deletedAt est présent (contrairement à history) : les organisations
soft-deleted côté webresto sont exclues avant chargement.

Pré-requis : les tables suivantes doivent être peuplées dans default_dataset :
  - etablissement_detail      (POST /etablissement_detail)
  - ref_type_ips_corrections  (POST /ref_type_ips_corrections)

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

JOB_NAME = "organization"

WEBRESTO_BASE_URL    = PARAMS["BASE_URL"]
ENVIRONNEMENT_CLIENT = PARAMS["ENVIRONNEMENT_CLIENT"]
PREFIX_TABLE         = PARAMS["PREFIX_TABLE"]
SECRET_KEY           = PARAMS["SECRET_KEY_WEBRESTO"]

dataset_cible   = f"dwh/db_mg6jk45h_{ENVIRONNEMENT_CLIENT}/"
dataset_default = "dwh/default_dataset/"
p = PREFIX_TABLE
TABLE = f"{p}organization"

# La nouvelle gateway route sous /data-lake — accepte BASE_URL avec ou sans
# ce segment pour ne pas dépendre de la convention retenue au déploiement.
_BASE = WEBRESTO_BASE_URL.rstrip("/")
if _BASE.endswith("/data-lake"):
    _BASE = _BASE[: -len("/data-lake")]
_ORGANIZATION_URL = f"{_BASE}/data-lake/findAll/organizations"

SELECTS = "organizationId,rne,name,city,type,department,academy,accessSoftware,ips,vague"

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
            resp = requests.get(_ORGANIZATION_URL, headers=_HEADERS, params=params, timeout=120)
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
            raw["source"].apply(lambda s: str(s) in ENVIRONNEMENT_CLIENT)
        ][["uai", "type", "ips"]].copy()
    else:
        df_corrections = pd.DataFrame(columns=["uai", "type", "ips"])
        logger.warning(f"[{JOB_NAME}] ref_type_ips_corrections : table vide ou non disponible")

    raw_dept = source_default.query(
        "SELECT DISTINCT uai, libelle_departement FROM etablissement_detail"
        " WHERE libelle_departement IS NOT NULL"
    )
    if isinstance(raw_dept, pd.DataFrame) and not raw_dept.empty:
        df_annuaire_dept = raw_dept
    else:
        df_annuaire_dept = pd.DataFrame(columns=["uai", "libelle_departement"])
        logger.warning(f"[{JOB_NAME}] etablissement_detail : table vide ou non disponible")

    logger.info(
        f"[{JOB_NAME}] Référence — corrections : {len(df_corrections)} lignes, "
        f"annuaire dept : {len(df_annuaire_dept)} établissements"
    )
    return df_corrections, df_annuaire_dept


# ── Transform ──────────────────────────────────────────────────────────────

def _build_nom_ville(name: str, city: str) -> str:
    name = name.rstrip("- _")
    city_clean = city.lstrip("0123456789 -")
    if city_clean.lower() not in name.lower():
        name = f"{name} - {city_clean}"
    return name


def _transform(
    items: list,
    df_corrections: pd.DataFrame,
    df_annuaire_dept: pd.DataFrame,
) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()

    df = pd.DataFrame(items)

    # ── Soft deletes : jamais chargés en full reload ──────────────────────────
    if "deletedAt" in df.columns:
        deleted_mask = df["deletedAt"].notna()
        n_deleted = int(deleted_mask.sum())
        if n_deleted:
            logger.info(f"[{JOB_NAME}] {n_deleted} soft-deleted ignorés")
        df = df[~deleted_mask].copy()

    if df.empty:
        return df

    cols = ["organizationId", "rne", "name", "city", "type", "department", "academy",
            "accessSoftware", "ips", "vague"]
    df = df[cols]

    if "centre" in ENVIRONNEMENT_CLIENT:
        df = df[~df["organizationId"].isin(_ORGA_CENTRE_DEMO)]
    if "93" in ENVIRONNEMENT_CLIENT:
        df = df[~df["organizationId"].isin(_ORGA_93_EXCLUDED)]

    if not df_corrections.empty:
        mapping = df_corrections.set_index("uai")
        df["type"] = df["rne"].map(mapping["type"]).combine_first(df["type"])
        df["ips"]  = df["rne"].map(mapping["ips"]).combine_first(df["ips"])

    if "93" in ENVIRONNEMENT_CLIENT:
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


# ── Entry point ────────────────────────────────────────────────────────────

def customfunc(event):
    logger.info(f"Démarrage du job '{JOB_NAME}'")
    t0 = time.time()

    source         = connect(dataset_cible)
    source_default = connect(dataset_default)

    df_corrections, df_annuaire_dept = _load_ref_data(source_default)

    items = _fetch_api()
    if not items:
        logger.warning(f"[{JOB_NAME}] Réponse API vide — table non modifiée")
        return

    df = _transform(items, df_corrections, df_annuaire_dept)
    if df.empty:
        logger.warning(f"[{JOB_NAME}] DataFrame vide après transform — table non modifiée")
        return

    source.query(f"DELETE FROM {TABLE}")
    bulk_insert(source, TABLE, df)

    duration = round(time.time() - t0, 2)
    logger.info(f"[{JOB_NAME}] OK — {len(df)} lignes chargées — {duration}s")

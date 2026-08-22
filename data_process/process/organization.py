"""
Transforms pour les entités liées à l'organisation :
  - organization  (GET /findAll/organizations)
  - subgroup      (POST /findAll/subgroups)
  - service       (GET /findAll/services)
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from data_process.process.constants import (
    ORGA_CENTRE_DEMO,
    ORGA_93_EXCLUDED,
    SUBGROUP_INTERNES,
    SUBGROUP_TICKETS,
)

logger = logging.getLogger(__name__)

_CSV_MAPPING_PATH = (
    Path(__file__).parent.parent / "stats-socle-admin_Type_etablissement_IPS_corrige.csv"
)
_CSV_MAPPING_93_PATH = (
    Path(__file__).parent.parent / "mapping_93_type_ips.csv"
)
_ANNUAIRE_PATH = (
    Path(__file__).parent.parent / "data" / "fr-en-annuaire-education.csv"
)


# ── Preprocess ────────────────────────────────────────────────────────────────

def preprocess_subgroup(items: list, warnings: Optional[list] = None) -> list:
    """Extrait groupId et calcule facturation_type (interne / ticket / autre)."""
    _internes = {s.lower() for s in SUBGROUP_INTERNES}
    _tickets = {s.lower() for s in SUBGROUP_TICKETS}
    for item in items:
        item["groupId"] = item["group"]["groupId"]
        label = item.get("label", "").strip().lower()
        if label in _internes:
            item["facturation_type"] = "interne"
        elif label in _tickets:
            item["facturation_type"] = "ticket"
        else:
            item["facturation_type"] = "autre"
    return items


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_nom_ville(name: str, city: str) -> str:
    name = name.rstrip("- _")
    city_clean = city.lstrip("0123456789 -")
    if city_clean.lower() not in name.lower():
        name = f"{name} - {city_clean}"
    return name


# ── Transforms ────────────────────────────────────────────────────────────────

def transform_organization(
    df: pd.DataFrame,
    environnement_client: str,
    df_corrections: Optional[pd.DataFrame] = None,
    df_annuaire_dept: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Transforme le DataFrame organizations issu de l'API Webresto.

    Paramètres optionnels pour le contexte forepass (sans accès fichier local) :
      df_corrections  : DataFrame (uai, type, ips) pré-filtré pour l'environnement courant.
                        Si None, lecture depuis les CSV locaux.
      df_annuaire_dept: DataFrame (uai, libelle_departement) depuis etablissement_detail DWH.
                        Si None, lecture depuis fr-en-annuaire-education.csv local.
    """
    if df.empty:
        return df

    cols = [
        "organizationId", "rne", "name",
        "city", "type", "department", "academy", "accessSoftware", "ips", "vague",
    ]
    df = df[cols]

    if "centre" in environnement_client:
        df = df[~df["organizationId"].isin(ORGA_CENTRE_DEMO)]
    if "93" in environnement_client:
        df = df[~df["organizationId"].isin(ORGA_93_EXCLUDED)]

    if df_corrections is not None:
        mapping = df_corrections.set_index("uai")
        df["type"] = df["rne"].map(mapping["type"]).combine_first(df["type"])
        df["ips"] = df["rne"].map(mapping["ips"]).combine_first(df["ips"])
    else:
        if "centre" in environnement_client:
            try:
                df_mapping = pd.read_csv(_CSV_MAPPING_PATH, delimiter=";")
                df_mapping = df_mapping.rename(columns={"IPS": "ips"})
                df_mapping.drop(index=[3, 8], inplace=True, errors="ignore")
                df_mapping["ips"] = pd.to_numeric(
                    df_mapping["ips"].astype(str).str.replace(",", ".", regex=False)
                )
                df["type"] = (
                    df["rne"].map(df_mapping.set_index("UAI")["type"]).combine_first(df["type"])
                )
                df["ips"] = (
                    df["rne"].map(df_mapping.set_index("UAI")["ips"]).combine_first(df["ips"])
                )
            except Exception as e:
                logger.warning(f"Impossible de charger le mapping type/ips ({_CSV_MAPPING_PATH}): {e}")

        if "93" in environnement_client:
            try:
                df_mapping_93 = pd.read_csv(_CSV_MAPPING_93_PATH, delimiter=";")
                df_mapping_93 = df_mapping_93.rename(columns={"IPS": "ips"})
                df_mapping_93["ips"] = pd.to_numeric(
                    df_mapping_93["ips"].astype(str).str.replace(",", ".", regex=False)
                )
                df["type"] = (
                    df["rne"].map(df_mapping_93.set_index("UAI")["type"]).combine_first(df["type"])
                )
                df["ips"] = (
                    df["rne"].map(df_mapping_93.set_index("UAI")["ips"]).combine_first(df["ips"])
                )
            except Exception as e:
                logger.warning(f"Impossible de charger le mapping type/ips ({_CSV_MAPPING_93_PATH}): {e}")

    if "93" in environnement_client:
        df["name"] = df.apply(lambda row: _build_nom_ville(str(row["name"]), str(row["city"])), axis=1)

    df = df.rename(columns={
        "organizationId": "id_organization",
        "accessSoftware": "access_software",
    })

    if df_annuaire_dept is not None:
        df_dept = df_annuaire_dept.rename(columns={"uai": "rne"}).drop_duplicates("rne")
        df = df.merge(df_dept[["rne", "libelle_departement"]], on="rne", how="left")
        df["department"] = df["libelle_departement"].combine_first(df["department"])
        df = df.drop(columns=["libelle_departement"])
    else:
        try:
            df_annuaire = pd.read_csv(
                _ANNUAIRE_PATH,
                sep=";",
                dtype=str,
                encoding="utf-8-sig",
                usecols=["Identifiant_de_l_etablissement", "Libelle_departement"],
            )
            df_annuaire = df_annuaire.rename(columns={
                "Identifiant_de_l_etablissement": "rne",
                "Libelle_departement": "libelle_departement",
            }).drop_duplicates("rne")
            df = df.merge(df_annuaire, on="rne", how="left")
            df["department"] = df["libelle_departement"].combine_first(df["department"])
            df = df.drop(columns=["libelle_departement"])
        except Exception as e:
            logger.warning(f"Impossible de charger libelle_departement ({_ANNUAIRE_PATH}): {e}")

    for col in ["access_software", "name", "city", "rne", "department", "type", "academy"]:
        df[col] = df[col].astype(str).str.strip()

    return df


def transform_subgroup(df: pd.DataFrame, environnement_client: str) -> pd.DataFrame:
    if df.empty:
        return df

    cols = ["subgroupId", "createdAt", "updatedAt", "label", "groupId", "acronym"]
    if "centre" in environnement_client:
        cols.append("facturation_type")
    df = df[cols]
    df["createdAt"] = pd.to_datetime(df["createdAt"], format="mixed")
    df["updatedAt"] = pd.to_datetime(df["updatedAt"], format="mixed")
    df = df.rename(columns={
        "subgroupId": "id_subgroup",
        "groupId": "id_group",
        "updatedAt": "updated_at",
        "createdAt": "created_at",
    })
    for col in ["label", "acronym"]:
        df[col] = df[col].astype(str).str.strip()
    return df


def transform_service(df: pd.DataFrame, environnement_client: str) -> pd.DataFrame:
    if df.empty:
        return df

    missing = [c for c in ["organizationId", "serviceId", "label"] if c not in df.columns]
    if missing:
        logger.error(f"Colonnes manquantes dans la réponse service: {missing}")
        return pd.DataFrame()

    df = df[["organizationId", "serviceId", "label"]]

    if "centre" in environnement_client:
        df = df[~df["organizationId"].isin(ORGA_CENTRE_DEMO)]
    if "93" in environnement_client:
        df = df[~df["organizationId"].isin(ORGA_93_EXCLUDED)]

    df = df.rename(columns={
        "serviceId": "id_service",
        "organizationId": "id_organization",
    })
    df["label"] = df["label"].astype(str).str.strip()
    return df

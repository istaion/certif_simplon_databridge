"""
Transforms pour les entités liées au contrôle d'accès :
  - passage          (POST /findAll/passages)
  - passage_partner  (POST /findAll/statistic_passage_partner)
  - subgroup_mapping (GET  /getSubgroupMapping)
"""

import logging
from typing import Optional

import pandas as pd

from data_process.process.constants import ORGA_CENTRE_DEMO, ORGA_93_EXCLUDED

logger = logging.getLogger(__name__)


# ── Preprocess ────────────────────────────────────────────────────────────────

def preprocess_passage(items: list, warnings: Optional[list] = None) -> list:
    """Traduit canceled=True en deletedAt pour déclencher la suppression Trino."""
    for item in items:
        if item.get("canceled") is True:
            item["deletedAt"] = item.get("updatedAt", "2000-01-01T00:00:00.000Z")
    return items


# ── Transforms ────────────────────────────────────────────────────────────────

def transform_passage(df: pd.DataFrame, environnement_client: str) -> pd.DataFrame:
    if df.empty:
        return df

    # TODO: réactiver quand deletedAt sera disponible dans la réponse API
    # df = df[df["deletedAt"].isna()]
    cols = [
        "passageId", "createdAt", "updatedAt", "organizationId",
        "userId", "serviceId", "date", "subgroupId", "trancheId",
    ]
    df = df[cols]

    if "centre" in environnement_client:
        df = df[~df["organizationId"].isin(ORGA_CENTRE_DEMO)]
    if "93" in environnement_client:
        df = df[~df["organizationId"].isin(ORGA_93_EXCLUDED)]

    df = df.rename(columns={
        "passageId": "id_passage",
        "organizationId": "id_organization",
        "userId": "id_user",
        "serviceId": "id_service",
        "subgroupId": "id_subgroup",
        "trancheId": "id_tranche",
        "createdAt": "created_at",
        "updatedAt": "updated_at",
    })
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df["created_at"] = pd.to_datetime(df["created_at"], format="mixed")
    df["updated_at"] = pd.to_datetime(df["updated_at"], format="mixed")
    # id_tranche est INT dans Trino → littéral entier requis.
    # astype("Int64") → _coerce_types → float64 → DOUBLE → TYPE_MISMATCH.
    # Python int dans une colonne object n'est pas quoté par QUOTE_NONNUMERIC → INTEGER.
    df["id_tranche"] = df["id_tranche"].apply(lambda x: int(x) if pd.notna(x) else None)
    return df


def transform_passage_partner(df: pd.DataFrame, environnement_client: str) -> pd.DataFrame:
    if df.empty:
        return df

    cols = ["id", "nbPassages", "tranche", "organizationId", "date", "subgroup", "service"]
    df = df[cols]

    if "centre" in environnement_client:
        df = df[~df["organizationId"].isin(ORGA_CENTRE_DEMO)]
    if "93" in environnement_client:
        df = df[~df["organizationId"].isin(ORGA_93_EXCLUDED)]

    df = df.rename(columns={
        "id": "id_partner",
        "nbPassages": "nb_passages",
        "organizationId": "id_organization",
    })
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    return df


def transform_subgroup_mapping(df: pd.DataFrame, environnement_client: str) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.rename(columns={
        "id": "id_subgroupmaping",
        "subgroupId": "id_subgroup",
    })
    return df

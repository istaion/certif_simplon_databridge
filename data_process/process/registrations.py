"""
Transforms pour les entités liées aux inscriptions et sessions :
  - history      (POST /findAll/history)
  - registration (POST /findAll/registrations)
  - session      (POST /findAll/sessions)
  - vague        (GET  /findAll/vagues)
"""

import logging
from typing import Optional

import pandas as pd

from data_process.process.constants import (
    ORGA_CENTRE_DEMO,
    ORGA_93_EXCLUDED,
    SESSIONS_CENTRE_TO_DEL,
    USERS_CENTRE_DELETED,
)

logger = logging.getLogger(__name__)


# ── Preprocess ────────────────────────────────────────────────────────────────

def preprocess_registration(items: list, warnings: Optional[list] = None) -> list:
    """Extrait sessionId (plat ou imbriqué), normalise registrationId→id, extrait registrationForm."""
    clean = []
    for item in items:
        # sessionId peut être plat (Option C) ou dans un objet session imbriqué (options.select)
        session_obj = item.get("session") or {}
        session_id = item.get("sessionId") or session_obj.get("id")

        if session_id is not None:
            item["sessionId"] = session_id
            item["vagueId"]   = session_obj.get("vagueId")
            # Option C retourne registrationId ; options.select retourne id
            if "id" not in item and "registrationId" in item:
                item["id"] = item["registrationId"]
            rf = item.get("registrationForm") or {}
            item["rfChoiceBankDetail"] = rf.get("choiceBankDetail")
            item["rfTrancheId"]        = rf.get("trancheId")
            item["rfSubgroupId"]       = rf.get("subgroupId")
            clean.append(item)
        else:
            reg_id = item.get("id") or item.get("registrationId")
            msg = f"Registration ignorée (session null) id={reg_id}"
            logger.warning(msg)
            if warnings is not None:
                warnings.append(msg)
    return clean


def preprocess_session(items: list, warnings: Optional[list] = None) -> list:
    """Extrait id_vague depuis l'objet imbriqué vague."""
    for item in items:
        try:
            item["id_vague"] = item["vague"]["vagueId"]
        except (KeyError, TypeError):
            msg = f"Session ignorée (vague null) id={item.get('id')}"
            logger.warning(msg)
            if warnings is not None:
                warnings.append(msg)
    return items


# ── Transforms ────────────────────────────────────────────────────────────────

def transform_history(df: pd.DataFrame, environnement_client: str) -> pd.DataFrame:
    if df.empty:
        return df

    df["createdAt"] = pd.to_datetime(df["createdAt"], format="mixed")
    df["updatedAt"] = pd.to_datetime(df["updatedAt"], format="mixed")
    cols = ["id", "registrationId", "event", "createdAt", "updatedAt"]
    df = df[cols]
    df = df.rename(columns={
        "id": "id_reg_history",
        "registrationId": "registration_id",
        "updatedAt": "updated_at",
        "createdAt": "created_at",
    })
    df["event"] = df["event"].astype(str).str.strip()
    return df


def transform_registration(
    df: pd.DataFrame,
    environnement_client: str,
    vague_map: Optional[dict] = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    if "centre" in environnement_client:
        df = df[~df["sessionId"].isin(SESSIONS_CENTRE_TO_DEL)]
        df = df[~df["userId"].isin(USERS_CENTRE_DELETED)]

    df.dropna(subset=["sessionId"], inplace=True)
    df["sessionId"] = df["sessionId"].astype("int64")
    df["createdAt"] = pd.to_datetime(df["createdAt"], format="mixed")
    df["updatedAt"] = pd.to_datetime(df["updatedAt"], format="mixed")

    cols = ["id", "status", "sessionId", "userId",
            "rfChoiceBankDetail", "rfTrancheId", "rfSubgroupId",
            "vagueId", "createdAt", "updatedAt"]
    df = df[cols]
    df = df.rename(columns={
        "sessionId": "id_session",
        "userId": "id_user",
        "rfChoiceBankDetail": "choice_bank_detail",
        "rfTrancheId": "tranche_id",
        "rfSubgroupId": "subgroup_id",
        "vagueId": "id_vague",
        "createdAt": "created_at",
        "updatedAt": "updated_at",
    })
    df["id_school_year"] = df["id_vague"].map(vague_map) if vague_map else None
    df = df.drop(columns=["id_vague"])
    df["status"] = df["status"].astype(str).str.strip()
    df["choice_bank_detail"] = df["choice_bank_detail"].apply(
        lambda x: str(x).strip() if pd.notna(x) else None
    )
    df["tranche_id"] = df["tranche_id"].apply(lambda x: int(x) if pd.notna(x) else None)
    return df


def transform_session(df: pd.DataFrame, environnement_client: str) -> pd.DataFrame:
    if df.empty:
        return df

    cols = ["id", "organizationId", "id_vague", "name"]
    df = df[cols]

    if "centre" in environnement_client:
        df = df[~df["organizationId"].isin(ORGA_CENTRE_DEMO)]
    if "93" in environnement_client:
        df = df[~df["organizationId"].isin(ORGA_93_EXCLUDED)]

    df = df.rename(columns={"organizationId": "id_organization"})
    df["name"] = df["name"].astype(str).str.strip()
    return df


def transform_vague(df: pd.DataFrame, environnement_client: str) -> pd.DataFrame:
    if df.empty:
        return df

    # Correction de startDate manquante pour les vagues connues (normalement corrigé à la source)
    # df.loc[df["vagueId"] == 1, "startDate"] = "2024-08-31T22:00:00.000Z"
    # df.loc[df["vagueId"] == 2, "startDate"] = "2025-01-01T00:00:00.000Z"
    df["startDate"] = pd.to_datetime(df["startDate"]).dt.date
    df["endDate"] = pd.to_datetime(df["endDate"]).dt.date
    cols = ["vagueId", "name", "startDate", "endDate", "SchoolYearId"]
    df = df[cols]
    df = df.rename(columns={
        "vagueId": "id_vague",
        "startDate": "start_date",
        "endDate": "end_date",
        "SchoolYearId": "id_school_year",
    })
    df["name"] = df["name"].astype(str).str.strip()
    return df

"""
Transforms pour les entités liées aux utilisateurs :
  - user       (POST /findAll/AllUsers)
  - bankdetail (GET /findAll/bankDetails -- migré de POST à GET, cf. incident E5)
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ── Preprocess ────────────────────────────────────────────────────────────────

def preprocess_bankdetail(items: list, warnings: Optional[list] = None) -> list:
    """Reprend userId (champ plat depuis la nouvelle gateway GET). Ignore les items sans userId."""
    clean = []
    for item in items:
        if item.get("userId") is not None:
            item["id_user"] = item["userId"]
            clean.append(item)
        else:
            msg = f"BankDetail ignoré (user null) bankDetailId={item.get('bankDetailId')}"
            logger.debug(msg)
            if warnings is not None:
                warnings.append(msg)
    return clean


# ── Transforms ────────────────────────────────────────────────────────────────

def transform_user(df: pd.DataFrame, environnement_client: str) -> pd.DataFrame:
    if df.empty:
        return df

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

    cols = ["userId", "createdAt", "updatedAt", "subgroupId", "first_name", "last_name", "date_birth"]
    df = df[cols]
    df = df.rename(columns={
        "userId": "id_user",
        "subgroupId": "id_subgroup",
        "updatedAt": "updated_at",
        "createdAt": "created_at",
    })
    return df


def transform_bankdetail(df: pd.DataFrame, environnement_client: str) -> pd.DataFrame:
    if df.empty:
        return df

    df["createdAt"] = pd.to_datetime(df["createdAt"], format="mixed")
    df["updatedAt"] = pd.to_datetime(df["updatedAt"], format="mixed")
    df["choiceBankDetails"] = df["choiceBankDetails"].apply(
        lambda x: x.strip() if isinstance(x, str) else None
    )

    cols = ["bankDetailId", "createdAt", "updatedAt", "id_user", "choiceBankDetails", "trancheId"]
    df = df[cols]
    df = df.rename(columns={
        "bankDetailId": "bank_detail_id",
        "choiceBankDetails": "choice_bank_details",
        "trancheId": "id_tranche",
        "updatedAt": "updated_at",
        "createdAt": "created_at",
    })
    # id_tranche est INT dans Trino → littéral entier requis (même logique que passage)
    df["id_tranche"] = df["id_tranche"].apply(lambda x: int(x) if pd.notna(x) else None)
    return df

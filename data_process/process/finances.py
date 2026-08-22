"""
Transforms pour les entités financières :
  - transaction (POST /findAll/transactions)
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def transform_transaction(df: pd.DataFrame, environnement_client: str) -> pd.DataFrame:
    if df.empty:
        return df

    if "deletedAt" in df.columns:
        df = df[df["deletedAt"].isna()]

    for col in ("createdAt", "updatedAt", "date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    cols = [
        "transactionId", "createdAt", "updatedAt", "date",
        "typeWriting", "label",
        "userId", "organizationId", "serviceId", "reservationId", "passageId",
        "incomingAccount", "outgoingAccount", "amount", "historicalBalance", "canceled",
    ]
    df = df[[c for c in cols if c in df.columns]]
    df = df.rename(columns={
        "transactionId":     "transaction_id",
        "createdAt":         "created_at",
        "updatedAt":         "updated_at",
        "typeWriting":       "type_writing",
        "userId":            "user_id",
        "organizationId":    "organization_id",
        "serviceId":         "service_id",
        "reservationId":     "reservation_id",
        "passageId":         "passage_id",
        "incomingAccount":   "incoming_account",
        "outgoingAccount":   "outgoing_account",
        "historicalBalance": "historical_balance",
    })

    for col in ("transaction_id", "user_id", "organization_id",
                "service_id", "reservation_id", "passage_id",
                "incoming_account", "outgoing_account"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("amount", "historical_balance"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "canceled" in df.columns:
        df["canceled"] = df["canceled"].astype(bool)

    return df

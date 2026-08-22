"""
Chargement des jours fériés depuis jours_feries_metropole.csv.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"


def load_jours_feries() -> pd.DataFrame:
    """
    Charge jours_feries_metropole.csv et retourne un DataFrame avec colonnes :
      date (date Python), annee (int), zone (str), nom_jour_ferie (str)
    """
    path = _DATA_DIR / "jours_feries_metropole.csv"
    df = pd.read_csv(path, dtype={"annee": "Int64", "zone": str, "nom_jour_ferie": str})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["annee"] = df["annee"].astype(int)
    logger.info(f"Jours fériés : {len(df)} entrées chargées ({df['annee'].min()}–{df['annee'].max()})")
    return df

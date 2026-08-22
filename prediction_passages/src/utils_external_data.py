"""
Utilitaires pour récupérer des données externes sur les établissements scolaires
depuis les APIs data.education.gouv.fr.

Deux sources :
- IPS (Indice de Position Sociale) : lycées + collèges, par année scolaire
- Type d'établissement : annuaire national (statique)
"""

import requests
import pandas as pd
from pathlib import Path

_SRC_DIR = Path(__file__).parent
_DATA_DIR = _SRC_DIR.parent / "data"

IPS_FALLBACK_CSV = _DATA_DIR / "ips_educ_gouv_complet.csv"
ETAB_FALLBACK_CSV = _DATA_DIR / "etab_educ_gouv.csv"

IPS_URLS = [
    "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-ips-lycees-ap2023/exports/json",
    "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/fr-en-ips-colleges-ap2023/exports/json",
]

ANNUAIRE_URL = (
    "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets"
    "/fr-en-annuaire-education/exports/json"
)


def fetch_ips() -> pd.DataFrame:
    """
    Récupère les IPS (lycées + collèges) depuis l'API education.gouv.fr.
    Fallback sur le CSV local si l'API est indisponible.

    Returns:
        DataFrame avec colonnes [uai, ips, rentree_scolaire]
    """
    try:
        dfs = []
        for url in IPS_URLS:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            temp_df = pd.DataFrame(r.json())
            if "ips" in temp_df.columns:
                temp_df = temp_df[["uai", "ips", "rentree_scolaire"]]
            else:
                temp_df = temp_df[["uai", "ips_etab", "rentree_scolaire"]]
                temp_df = temp_df.rename(columns={"ips_etab": "ips"})
            dfs.append(temp_df)

        df_ips = pd.concat(dfs, ignore_index=True)
        df_ips["ips"] = pd.to_numeric(df_ips["ips"], errors="coerce")
        return df_ips
    except Exception as e:
        print(f"  API IPS indisponible ({e}), fallback sur CSV local...")
        df_ips = pd.read_csv(IPS_FALLBACK_CSV, usecols=["uai", "ips", "rentree_scolaire"])
        df_ips["ips"] = pd.to_numeric(df_ips["ips"], errors="coerce")
        return df_ips


def fetch_type_etablissement() -> pd.DataFrame:
    """
    Récupère le type d'établissement depuis l'annuaire education.gouv.fr.
    Fallback sur le CSV local si l'API est indisponible.

    Returns:
        DataFrame avec colonnes [login_site, type_etablissement]
    """
    try:
        r = requests.get(ANNUAIRE_URL, timeout=120)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        df = df[["identifiant_de_l_etablissement", "type_etablissement"]].rename(
            columns={"identifiant_de_l_etablissement": "login_site"}
        )
        return df.drop_duplicates("login_site")
    except Exception as e:
        print(f"  API annuaire indisponible ({e}), fallback sur CSV local...")
        df = pd.read_csv(ETAB_FALLBACK_CSV, usecols=["identifiant_de_l_etablissement", "type_etablissement"])
        df = df.rename(columns={"identifiant_de_l_etablissement": "login_site"})
        return df.drop_duplicates("login_site")


def enrich_with_external_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichit le DataFrame principal avec l'IPS et le type d'établissement.

    Stratégie IPS : join sur (login_site, school_year). Si l'année scolaire
    n'est pas disponible dans les données IPS, on utilise l'IPS de l'année
    la plus récente disponible pour ce site.

    Args:
        df: DataFrame principal avec colonnes login_site et school_year

    Returns:
        DataFrame enrichi avec colonnes ips et type_etablissement
    """
    print("  Récupération IPS...")
    try:
        df_ips = fetch_ips()
    except Exception as e:
        print(f"  Erreur IPS (CSV introuvable aussi: {e}), colonne ips sera NaN")
        df["ips"] = float("nan")
        df_ips = None

    print("  Récupération type établissement...")
    try:
        df_type = fetch_type_etablissement()
    except Exception as e:
        print(f"  Erreur type établissement (CSV introuvable aussi: {e}), colonne sera NaN")
        df["type_etablissement"] = float("nan")
        df_type = None

    # --- Join IPS ---
    if df_ips is not None:
        # Join exact sur (login_site, school_year) via rentree_scolaire
        df_ips = df_ips.rename(columns={"uai": "login_site", "rentree_scolaire": "school_year"})

        df = df.merge(df_ips, on=["login_site", "school_year"], how="left")

        # Fallback : pour les lignes sans IPS, prendre la valeur la plus récente disponible
        ips_latest = (
            df_ips.sort_values("school_year")
            .groupby("login_site")["ips"]
            .last()
            .reset_index()
            .rename(columns={"ips": "ips_fallback"})
        )
        df = df.merge(ips_latest, on="login_site", how="left")
        df["ips"] = df["ips"].fillna(df["ips_fallback"])
        df = df.drop(columns=["ips_fallback"])

        n_missing = df["ips"].isna().sum()
        n_total = len(df)
        print(f"  IPS : {n_total - n_missing}/{n_total} lignes renseignées "
              f"({(n_total - n_missing) / n_total * 100:.1f}%)")

    # --- Join type établissement ---
    if df_type is not None:
        df = df.merge(df_type, on="login_site", how="left")
        n_missing = df["type_etablissement"].isna().sum()
        n_total = len(df)
        print(f"  Type établissement : {n_total - n_missing}/{n_total} lignes renseignées "
              f"({(n_total - n_missing) / n_total * 100:.1f}%)")

    return df

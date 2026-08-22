"""
Enrichissement de la table etablissement_detail :
  - Récupère les UAIs depuis les tables Trino source (selon l'environnement)
  - Enrichit depuis fr-en-annuaire-education.csv (données statiques)
  - Joint avec les 8 CSV IPS (un row par UAI × année scolaire trouvée)
  - Ajoute la vacances_zone (A/B/C) depuis le code académie
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from data_process.db.trino_client import TrinoClient

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"
_CSV_CORRECTIONS_CENTRE_PATH = Path(__file__).parent.parent / "stats-socle-admin_Type_etablissement_IPS_corrige.csv"
_CSV_CORRECTIONS_93_PATH = Path(__file__).parent.parent / "mapping_93_type_ips.csv"

# Mapping code académie (2 chiffres, zéro-paddé) → zone de vacances scolaires
ACADEMIE_ZONE: dict[str, str] = {
    "01": "C",  # Paris
    "02": "B",  # Aix-Marseille
    "03": "A",  # Besançon
    "04": "A",  # Bordeaux
    "05": "B",  # Caen
    "06": "A",  # Clermont-Ferrand
    "07": "A",  # Dijon
    "08": "A",  # Grenoble
    "09": "B",  # Lille
    "10": "A",  # Lyon
    "11": "C",  # Montpellier
    "12": "B",  # Nancy-Metz
    "13": "B",  # Nantes
    "14": "B",  # Nice
    "15": "B",  # Orléans-Tours
    "16": "A",  # Poitiers
    "17": "B",  # Reims
    "18": "B",  # Rennes
    "19": "B",  # Rouen
    "20": "B",  # Amiens
    "21": "B",  # Strasbourg
    "22": "A",  # Limoges
    "23": "C",  # Toulouse
    "24": "C",  # Créteil
    "25": "C",  # Versailles
    "70": "B",  # Normandie
}

# (chemin du fichier, colonne UAI, colonne IPS à utiliser)
IPS_FILES: list[tuple[Path, str, str]] = [
    (_DATA_DIR / "fr-en-ips_ecoles_v2.csv",        "UAI", "IPS"),
    (_DATA_DIR / "fr-en-ips-ecoles-ap2022.csv",     "UAI", "IPS"),
    (_DATA_DIR / "fr-en-ips_erea.csv",              "UAI", "IPS"),
    (_DATA_DIR / "fr-en-ips-erea-ap2022.csv",       "UAI", "IPS"),
    (_DATA_DIR / "fr-en-ips-colleges-ap2022.csv",   "UAI", "IPS"),
    (_DATA_DIR / "fr-en-ips-colleges-ap2023.csv",   "UAI", "IPS"),
    (_DATA_DIR / "fr-en-ips-lycees-ap2022.csv",     "UAI", "IPS Ensemble GT-PRO"),
    (_DATA_DIR / "fr-en-ips-lycees-ap2023.csv",     "UAI", "IPS de l'établissement"),
]


def load_ref_type_ips_corrections() -> pd.DataFrame:
    """
    Charge les corrections manuelles type/IPS depuis les 2 CSV sources.
    Colonnes retournées : uai, type, ips, source ("centre" ou "93").
    Les doublons sur uai sont dédupliqués (premier occurrence conservée).
    """
    frames = []
    for path, source_label in [
        (_CSV_CORRECTIONS_CENTRE_PATH, "centre"),
        (_CSV_CORRECTIONS_93_PATH, "93"),
    ]:
        try:
            df = pd.read_csv(path, delimiter=";", dtype=str)
            df = df.rename(columns={"UAI": "uai", "IPS": "ips"})
            df = df.drop_duplicates(subset=["uai"], keep="first")
            df["ips"] = pd.to_numeric(
                df["ips"].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
            df["source"] = source_label
            frames.append(df[["uai", "type", "ips", "source"]].dropna(subset=["uai"]))
            logger.info(f"Corrections {source_label} : {len(frames[-1])} lignes depuis {path.name}")
        except Exception as e:
            logger.warning(f"Impossible de charger {path.name} : {e}")

    if not frames:
        return pd.DataFrame(columns=["uai", "type", "ips", "source"])

    return pd.concat(frames, ignore_index=True)


def get_uais_from_trino(
    db: "TrinoClient",
    prefix_table: str,
    environnement_client: str,
    prefix_webresto: str = None,
) -> set[str]:
    """Retourne l'union des UAIs depuis {prefix}login et, si disponible, {prefix}organization.

    prefix_table    : préfixe Webgerest (ex: "wg_test_") — contient la table login.
    prefix_webresto : préfixe Webresto  (ex: "wr_centre_") — contient la table organization.
                      Si None, utilise prefix_table pour les deux.
    """
    uais: set[str] = set()
    prefix_orga = prefix_webresto if prefix_webresto else prefix_table

    # Table login — Webgerest
    try:
        df_login = db.query_as_dataframe(
            f"SELECT DISTINCT login AS uai FROM {prefix_table}login "
            f"WHERE profil = 2 AND login IS NOT NULL AND login <> ''"
        )
        login_uais = set(df_login["uai"].dropna().tolist())
        uais |= login_uais
        logger.info(f"  {len(login_uais)} UAIs depuis {prefix_table}login")
    except Exception as e:
        logger.warning(f"  Table {prefix_table}login non accessible : {e}")

    # Table organization — Webresto uniquement, absente sur certains environnements
    try:
        df_orga = db.query_as_dataframe(
            f"SELECT DISTINCT rne AS uai FROM {prefix_orga}organization "
            f"WHERE rne IS NOT NULL AND rne <> ''"
        )
        orga_uais = set(df_orga["uai"].dropna().tolist())
        uais |= orga_uais
        logger.info(f"  {len(orga_uais)} UAIs depuis {prefix_orga}organization")
    except Exception as e:
        logger.info(f"  Table {prefix_orga}organization absente ou inaccessible (normal sans Webresto) : {e}")

    logger.info(f"  → {len(uais)} UAIs uniques pour {environnement_client}")
    return uais


def load_annuaire(uais: set[str]) -> pd.DataFrame:
    """
    Charge les colonnes statiques depuis fr-en-annuaire-education.csv.
    Retourne un DataFrame avec les colonnes renommées, codes convertis en numérique.
    """
    path = _DATA_DIR / "fr-en-annuaire-education.csv"
    df = pd.read_csv(
        path,
        sep=";",
        dtype=str,
        encoding="utf-8-sig",
        usecols=[
            "Identifiant_de_l_etablissement",
            "Nom_etablissement",
            "Type_etablissement",
            "Code_departement",
            "Code_academie",
            "Code_region",
            "Libelle_departement",
            "Libelle_academie",
            "Libelle_region",
            "libelle_nature",
        ],
    )
    df = df[df["Identifiant_de_l_etablissement"].isin(uais)].copy()
    df = df.rename(columns={
        "Identifiant_de_l_etablissement": "uai",
        "Nom_etablissement":              "nom_etablissement",
        "Type_etablissement":             "type_etablissement",
        "Code_departement":               "code_departement",
        "Code_academie":                  "code_academie",
        "Code_region":                    "code_region",
        "Libelle_departement":            "libelle_departement",
        "Libelle_academie":               "libelle_academie",
        "Libelle_region":                 "libelle_region",
    })
    for col in ["code_academie", "code_departement", "code_region"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info(f"Annuaire : {len(df)} établissements trouvés sur {len(uais)} UAIs")
    return df.reset_index(drop=True)


def load_ips(uais: set[str]) -> pd.DataFrame:
    """
    Charge (uai, school_year, ips) depuis tous les CSV IPS, filtrés sur les UAIs.
    Retourne un DataFrame dédupliqué sur (uai, school_year), valeur non-nulle prioritaire.
    """
    frames: list[pd.DataFrame] = []
    for path, uai_col, ips_col in IPS_FILES:
        try:
            df = pd.read_csv(
                path,
                sep=";",
                dtype=str,
                encoding="utf-8-sig",
                usecols=["Rentrée scolaire", uai_col, ips_col],
            )
        except FileNotFoundError:
            logger.warning(f"Fichier IPS introuvable, ignoré : {path.name}")
            continue
        except ValueError as e:
            logger.warning(f"Colonne manquante dans {path.name} : {e}")
            continue

        df = df.rename(columns={
            "Rentrée scolaire": "school_year",
            uai_col:            "uai",
            ips_col:            "ips",
        })
        df = df[df["uai"].isin(uais)].copy()
        df["ips"] = pd.to_numeric(
            df["ips"].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        frames.append(df[["uai", "school_year", "ips"]])
        logger.info(f"{path.name} : {len(df)} lignes pour les UAIs cibles")

    if not frames:
        return pd.DataFrame(columns=["uai", "school_year", "ips"])

    combined = pd.concat(frames, ignore_index=True)
    # Déduplication : pour un même (uai, school_year), garder la valeur IPS non-nulle ;
    # si plusieurs valeurs non-nulles, garder la dernière source chargée (ordre IPS_FILES)
    combined = (
        combined
        .sort_values("ips", na_position="first")
        .drop_duplicates(subset=["uai", "school_year"], keep="last")
        .reset_index(drop=True)
    )
    logger.info(
        f"IPS total : {len(combined)} lignes uniques (uai, school_year) "
        f"pour {combined['uai'].nunique()} UAIs"
    )
    return combined


def build_etablissement_df(annuaire: pd.DataFrame, ips: pd.DataFrame) -> pd.DataFrame:
    """
    Merge IPS (uai, school_year, ips) × annuaire (uai, champs statiques).
    Ajoute la colonne vacances_zone depuis ACADEMIE_ZONE.
    Retourne le DataFrame final prêt pour l'upsert.
    """
    df = ips.merge(annuaire, on="uai", how="left")
    df["vacances_zone"] = df["code_academie"].apply(
        lambda c: ACADEMIE_ZONE.get(str(int(c)).zfill(2)) if pd.notna(c) else None
    )

    # UAIs présents dans l'annuaire mais sans aucune donnée IPS
    # → ajouter une ligne par school_year disponible pour qu'ils soient trouvables en fallback
    uais_sans_ips = set(annuaire["uai"]) - set(ips["uai"])
    if uais_sans_ips:
        extra = annuaire[annuaire["uai"].isin(uais_sans_ips)].copy()
        extra["vacances_zone"] = extra["code_academie"].apply(
            lambda c: ACADEMIE_ZONE.get(str(int(c)).zfill(2)) if pd.notna(c) else None
        )
        extra["ips"] = float("nan")
        rows = []
        for sy in sorted(ips["school_year"].unique()):
            r = extra.copy()
            r["school_year"] = sy
            rows.append(r)
        df = pd.concat([df] + rows, ignore_index=True)
        logger.info(f"{len(uais_sans_ips)} UAIs sans IPS ajoutés avec NaN pour {len(ips['school_year'].unique())} school_years")

    cols = [
        "uai",
        "school_year",
        "type_etablissement",
        "nom_etablissement",
        "code_academie",
        "libelle_academie",
        "code_departement",
        "libelle_departement",
        "code_region",
        "libelle_region",
        "libelle_nature",
        "vacances_zone",
        "ips",
    ]
    return df[cols].reset_index(drop=True)

"""
Transformation générique des données Webgerest pilotée par TableSchema.

Logique appliquée dans l'ordre :
  1. snake_case sur les noms de colonnes
  2. column_renames (renommages spécifiques post-snake_case)
  3. .strip() sur toutes les colonnes VARCHAR
  4. Remplacement des chaînes vides / "nan" par NaN
  5. Warning si codss1/codss2 contiennent autre chose que des chiffres
  6. Conversions de types selon trino_type du schéma
  7. Injection de login_site (ou login_group selon site_column)
  8. Construction de la colonne pk = site_column + "_" + pk_source_columns
  9. Injection de descfic_statut si présent dans le schéma
  10. Sélection des seules colonnes du schéma (les colonnes absentes sont ignorées)

Usage :
    from data_process.process.schemas_webgerest import WEBGEREST_SCHEMAS
    from data_process.process.transform_webgerest import transform_generic

    schema = WEBGEREST_SCHEMAS["article"]
    df_clean = transform_generic(df_raw, schema, login_identifier="0410003B", descfic_statut=2)
"""

from __future__ import annotations

import logging
import re
import unicodedata

import numpy as np
import pandas as pd

from data_process.process.ddl import TableSchema

logger = logging.getLogger(__name__)

_CODSS_RE = re.compile(r"^[0-9]*$")


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def to_snake_case(text: str) -> str:
    """Convertit un nom de colonne en snake_case (même logique que les jobs temp)."""
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = text.lower()
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    text = re.sub(r"^id(?=[^_])", "id_", text)
    return text


def _convert_column(series: pd.Series, trino_type: str, date_fmt: str | None) -> pd.Series:
    """Convertit une Series pandas vers le type Trino cible."""
    if trino_type == "DOUBLE":
        return pd.to_numeric(series, errors="coerce").astype("float64")

    if trino_type in ("BIGINT", "INTEGER"):
        return pd.to_numeric(series, errors="coerce").astype("Int64")

    if trino_type == "BOOLEAN":
        def _to_bool(x):
            if pd.isna(x):
                return None
            s = str(x).strip().lower()
            if s in ("true", "1", "yes", "oui"):
                return True
            if s in ("false", "0", "no", "non"):
                return False
            return None
        return series.apply(_to_bool)

    if trino_type == "TIMESTAMP(6)":
        return pd.to_datetime(series, errors="coerce")

    if trino_type == "DATE":
        parsed = pd.to_datetime(series, format=date_fmt, errors="coerce")
        # Convertir en datetime.date pour que TrinoClient génère DATE '...' et non TIMESTAMP
        return parsed.apply(lambda x: x.date() if pd.notna(x) else None)

    # VARCHAR : already handled by strip step, keep as-is
    return series


# ---------------------------------------------------------------------------
# Transform générique
# ---------------------------------------------------------------------------

def transform_generic(
    df: pd.DataFrame,
    schema: TableSchema,
    login_identifier: str,
    code_site=None,
    descfic_statut: int | None = None,
) -> pd.DataFrame:
    """Transforme un DataFrame brut API selon le TableSchema.

    Args:
        df:               DataFrame brut retourné par WebgestFetcher.fetch_table()
        schema:           TableSchema de la table cible
        login_identifier: valeur du login_group ou login_site à injecter
        code_site:        ignoré sauf pour plandis (code_site_from_api=True)
        descfic_statut:   statut DESCFIC du groupe pour cette table (1 ou 2)

    Returns:
        DataFrame transformé, prêt pour bulk_insert ou upsert via TrinoClient.
    """
    df = df.copy()

    # 1. snake_case
    df.rename(columns={col: to_snake_case(col) for col in df.columns}, inplace=True)

    # 2. column_renames spécifiques
    if schema.column_renames:
        df.rename(columns=schema.column_renames, inplace=True)

    # 3. strip + nettoyage des chaînes vides
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
    df.replace("", np.nan, inplace=True)
    df.replace("nan", np.nan, inplace=True)

    # 4. Warning codss1 / codss2
    for codss_col in ("codss1", "codss2"):
        if codss_col in df.columns:
            invalid = df[codss_col].dropna().apply(
                lambda x: not _CODSS_RE.match(str(x))
            )
            if invalid.any():
                bad_values = df.loc[invalid[invalid].index, codss_col].unique().tolist()
                logger.warning(
                    f"[{codss_col}] Valeurs avec caractères non-numériques : {bad_values}"
                )

    # 5. Conversions de types
    schema_col_names = {col.name for col in schema.columns}
    date_formats = schema.date_formats or {}

    for col_def in schema.columns:
        if col_def.name not in df.columns:
            continue
        df[col_def.name] = _convert_column(
            df[col_def.name],
            col_def.trino_type,
            date_formats.get(col_def.name),
        )

    # 6. Injection site_column
    df[schema.site_column] = str(login_identifier)

    schema_col_names_set = {col.name for col in schema.columns}

    # 7. Construction de pk = site_column + "_" + pk_source_columns (élément par élément)
    if schema.pk_source_columns:
        df["pk"] = (
            df[[schema.site_column] + schema.pk_source_columns]
            .astype(str)
            .agg("_".join, axis=1)
        )

    # 8. Injection descfic_statut si présent dans le schéma
    if "descfic_statut" in schema_col_names_set:
        df["descfic_statut"] = descfic_statut

    # 9. Sélection des colonnes du schéma (dans l'ordre du schéma, colonnes manquantes ignorées)
    cols_to_keep = [col.name for col in schema.columns if col.name in df.columns]
    df = df[cols_to_keep]

    return df

"""
Infrastructure de génération DDL pour les tables Trino/Iceberg.

`TableSchema` est la source de vérité unique pour un schéma de table :
- `to_ddl(table_name)`         → SQL CREATE TABLE IF NOT EXISTS avec propriétés Iceberg
- `to_pydantic_model()`        → modèle Pydantic dynamique pour tests / fixtures
- `validate_dataframe(df)`     → liste d'erreurs avant un bulk_insert

Métadonnées de synchronisation (utilisées par les jobs webgerest) :
- `primary_keys`       → clés pour le MERGE (upsert)
- `column_updates`     → colonne de date pour la borne incrémentale (None = full reload)
- `column_renames`     → renommages post-snake_case {nom_api: nom_trino}
- `api_table_name`     → nom de la route API si différent du nom du dict (ex: "detailarticle")
- `code_site_from_api` → True si code_site est dans la réponse API (pas injecté depuis login)
- `site_column`        → colonne identifiant le site/groupe ("login_site" par défaut, "login_group" pour fourn)
- `date_formats`       → format strptime non-standard par colonne (ex: {"dcreart": "%Y%m%d"})

Les schémas par application sont dans des modules séparés :
    data_process.process.schemas_webresto   → WEBRESTO_SCHEMAS
    data_process.process.schemas_webgerest  → WEBGEREST_SCHEMAS
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import pandas as pd
from pydantic import BaseModel, create_model

# ---------------------------------------------------------------------------
# Mapping types Trino → Python
# ---------------------------------------------------------------------------

_TRINO_TO_PYTHON: dict[str, type] = {
    "DOUBLE": float,
    "BIGINT": int,
    "INTEGER": int,
    "VARCHAR": str,
    "TIMESTAMP(6)": datetime,
    "DATE": date,
    "BOOLEAN": bool,
}

_ICEBERG_PROPS = """\
WITH (
    extra_properties = MAP(
        ARRAY[
            'write.target-file-size-bytes',
            'write.metadata.delete-after-commit.enabled',
            'write.metadata.previous-versions-max'
        ],
        ARRAY[
            '268435456',
            'true',
            '50'
        ]
    )
)"""

# ---------------------------------------------------------------------------
# Structures de schéma
# ---------------------------------------------------------------------------


@dataclass
class ColumnDef:
    name: str
    trino_type: str  # ex: "DOUBLE", "BIGINT", "INTEGER", "VARCHAR", "TIMESTAMP(6)", "DATE", "BOOLEAN"


@dataclass
class TableSchema:
    columns: list[ColumnDef]
    partitioning: list[str] | None = None       # ex: ["login_site"]

    # Métadonnées de synchronisation webgerest
    primary_keys: list[str] | None = None
    column_updates: str | None = None           # col pour borne incrémentale (None = full reload)
    column_renames: dict[str, str] | None = None  # {nom_api_post_snake_case: nom_trino}
    api_table_name: str | None = None           # route API si ≠ clé du dict (ex: "detailarticle")
    code_site_from_api: bool = False            # code_site vient de l'API, pas du login row
    site_column: str = "login_site"             # colonne identifiant le site/groupe
    date_formats: dict[str, str] | None = None  # format strptime non-standard {col: fmt}
    pk_source_columns: list[str] = field(default_factory=list)  # colonnes business (hors site_column) pour construire pk

    def to_ddl(self, table_name: str) -> str:
        """Génère le SQL CREATE TABLE IF NOT EXISTS avec propriétés Iceberg.

        Si `partitioning` est défini, ajoute `partitioning = ARRAY[...]` dans le WITH.
        """
        col_defs = ",\n    ".join(
            f"{col.name} {col.trino_type}" for col in self.columns
        )
        if self.partitioning:
            partition_items = ", ".join(f"'{p}'" for p in self.partitioning)
            with_clause = f"""\
WITH (
    partitioning = ARRAY[{partition_items}],
    extra_properties = MAP(
        ARRAY[
            'write.target-file-size-bytes',
            'write.metadata.delete-after-commit.enabled',
            'write.metadata.previous-versions-max'
        ],
        ARRAY[
            '268435456',
            'true',
            '50'
        ]
    )
)"""
        else:
            with_clause = _ICEBERG_PROPS
        return f"CREATE TABLE IF NOT EXISTS {table_name} (\n    {col_defs}\n)\n{with_clause}"

    def to_pydantic_model(self, model_name: str = "TableRow") -> type[BaseModel]:
        """Génère dynamiquement un modèle Pydantic (tous champs Optional).

        Utile pour la validation ou la génération de fixtures de test.
        """
        fields: dict[str, object] = {}
        for col in self.columns:
            py_type = _TRINO_TO_PYTHON.get(col.trino_type, object)
            fields[col.name] = (Optional[py_type], None)
        return create_model(model_name, **fields)

    def validate_dataframe(self, df: pd.DataFrame) -> list[str]:
        """Valide un DataFrame par rapport au schéma.

        Retourne une liste d'erreurs (vide si tout est OK) :
        - colonnes attendues absentes du DataFrame
        - colonnes dont le dtype pandas est incompatible avec le type Trino attendu
        """
        errors: list[str] = []
        expected_cols = {col.name for col in self.columns}
        actual_cols = set(df.columns)

        missing = expected_cols - actual_cols
        if missing:
            errors.append(f"Colonnes manquantes : {sorted(missing)}")

        for col in self.columns:
            if col.name not in actual_cols:
                continue
            py_type = _TRINO_TO_PYTHON.get(col.trino_type)
            if py_type is None:
                continue
            series = df[col.name].dropna()
            if series.empty:
                continue
            if py_type in (float, int) and not pd.api.types.is_numeric_dtype(series):
                errors.append(
                    f"'{col.name}' : type attendu {col.trino_type}, dtype pandas = {series.dtype}"
                )
            elif py_type == bool and not pd.api.types.is_bool_dtype(series):
                errors.append(
                    f"'{col.name}' : type attendu BOOLEAN, dtype pandas = {series.dtype}"
                )
            elif py_type == datetime and not pd.api.types.is_datetime64_any_dtype(series):
                errors.append(
                    f"'{col.name}' : type attendu TIMESTAMP(6), dtype pandas = {series.dtype}"
                )
            elif py_type == date and not (
                pd.api.types.is_datetime64_any_dtype(series)
                or pd.api.types.is_object_dtype(series)
            ):
                errors.append(
                    f"'{col.name}' : type attendu DATE, dtype pandas = {series.dtype}"
                )
        return errors

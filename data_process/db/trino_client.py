import csv
import datetime
import logging
import re
from typing import Optional

# Regex pour ré-enlever les guillemets autour des floats en notation scientifique
# produits par float_format="%.10E" après QUOTE_NONNUMERIC :
# '1.0000000000E+00' → 1.0000000000E+00  (Trino infera DOUBLE)
_QUOTED_FLOAT_RE = re.compile(r"'(-?\d+\.\d+E[+-]\d+)'")

# Limite de longueur de texte SQL imposée par Trino (défaut 1_000_000).
# On se laisse 10 % de marge pour les en-têtes de requête.
_TRINO_MAX_QUERY_LEN = 900_000

import pandas as pd
from trino.auth import BasicAuthentication
from trino.dbapi import connect

logger = logging.getLogger(__name__)

TRINO_HOST = "data-ianord-query.eu.dataplatform.ovh.net"
DEFAULT_START_DATE = "2022-08-01"

# Date seule : YYYY-MM-DD exactement
_ISO_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Datetime : YYYY-MM-DD suivi d'un séparateur T/ et d'une heure
_ISO_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?$"
)


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce les types pandas avant sérialisation CSV pour correspondre aux types Trino.

    - bool        → str ('true'/'false') : Python bool est sous-classe de int,
                    csv.QUOTE_NONNUMERIC ne le quote pas → Trino reçoit un boolean littéral
    - int*        → float64 + notation scientifique (via float_format dans to_csv) :
                    le littéral `1.0` est inféré decimal(2,1) par Trino, `1.0E+00` est DOUBLE
    - date Python → marqueur "DATE YYYY-MM-DD DATE" : les objets date issus de .dt.date
                    ne sont pas des datetime64, il faut les sérialiser en DATE '...'
    - str ISO date seule  → idem marqueur DATE
    - str ISO datetime    → datetime64 → TIMESTAMP '...' via date_format
    """
    import datetime as _dt

    df = df.copy()

    # bool → string ('true' / 'false') pour que Trino insère dans une colonne varchar
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].map({True: "true", False: "false"})

    # int → float64 (sera sérialisé en notation scientifique → DOUBLE Trino)
    for col in df.select_dtypes(include=["integer"]).columns:
        df[col] = df[col].astype("float64")

    # datetime avec timezone (ex: "2024-01-01T00:00:00Z" → Timestamp UTC) →
    # strip tz pour que Trino reçoive TIMESTAMP sans time zone (timestamp(6))
    for col in df.select_dtypes(include=["datetimetz"]).columns:
        df[col] = df[col].dt.tz_convert(None)

    # object : date Python, str ISO date, str ISO datetime
    for col in df.select_dtypes(include=["object"]).columns:
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        sample = non_null.iloc[0]

        # Objets Python datetime.date issus de pd.Series.dt.date
        if isinstance(sample, _dt.date) and not isinstance(sample, _dt.datetime):
            df[col] = df[col].apply(
                lambda x: f"DATE {x} DATE" if pd.notna(x) else None
            )
            continue

        if not isinstance(sample, str):
            continue

        # Strings date seule (YYYY-MM-DD) → marqueur DATE
        if non_null.str.match(_ISO_DATE_ONLY_RE).all():
            df[col] = df[col].apply(
                lambda x: f"DATE {x} DATE" if pd.notna(x) else None
            )
        # Strings datetime → datetime64 → TIMESTAMP via date_format
        elif non_null.str.match(_ISO_DATETIME_RE).all():
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

    return df


def _format_values(df_batch: pd.DataFrame) -> str:
    """Formate un batch DataFrame en bloc VALUES compatible Trino (gestion TIMESTAMP, NULL)."""
    df_batch = _coerce_types(df_batch)
    result = (
        "("
        + "{}".format(
            df_batch.to_csv(
                header=None,
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
                quotechar="'",
                na_rep="NULL",
                date_format="TIMESTAMP %Y-%m-%d %H:%M:%S.%f%z TIMESTAMP",
                float_format="%.10E",   # 1.0 → '1.0000000000E+00' (string quotée)
            )
        )
        .replace("'NULL'", "NULL")
        .replace("'true'", "true")
        .replace("'false'", "false")
        .replace("'TIMESTAMP ", "TIMESTAMP '")
        .replace(" TIMESTAMP'", "'")
        .replace("'DATE ", "DATE '")
        .replace(" DATE'", "'")
        .strip("\n")
        .replace("\n", "),(")
        + ")"
    )
    # Ré-enlever les guillemets autour des floats en notation scientifique :
    # float_format les a convertis en string → QUOTE_NONNUMERIC les a quotés
    # '1.0000000000E+00' → 1.0000000000E+00  (Trino infère DOUBLE)
    return _QUOTED_FLOAT_RE.sub(r"\1", result)


class TrinoClient:
    """
    Client Trino remplaçant forepaas (connect / query / bulk_insert / upsert).

    La connexion est créée à l'initialisation ; les credentials OVH sont passés
    explicitement plutôt que lus depuis os.environ, pour permettre des tests
    et une injection propre depuis l'orchestrateur.
    """

    def __init__(
        self,
        environnement_client: str,
        ovh_api_key: str,
        ovh_secret_key: str,
    ) -> None:
        self._conn = connect(
            host=TRINO_HOST,
            port=443,
            user=ovh_api_key,
            auth=BasicAuthentication(ovh_api_key, ovh_secret_key),
            catalog=f"db_mg6jk45h_{environnement_client}",
            schema=environnement_client,
            http_scheme="https",
        )

    # ── Primitives ────────────────────────────────────────────────────────────

    def _cursor(self):
        return self._conn.cursor()

    def get_last_updated_at(self, table: str, column: str) -> str:
        """Retourne MAX(column) formaté en YYYY-MM-DD, ou DEFAULT_START_DATE si vide."""
        try:
            cursor = self._cursor()
            cursor.execute(f"SELECT MAX({column}) AS last_update FROM {table}")
            row = cursor.fetchone()
            if row and row[0] is not None:
                val = row[0]
                return val if isinstance(val, str) else val.strftime("%Y-%m-%d")
            logger.info(f"Table {table} vide, utilisation de la date par défaut")
            return DEFAULT_START_DATE
        except Exception as e:
            logger.warning(f"Impossible de lire MAX({column}) sur {table}: {e}")
            return DEFAULT_START_DATE

    def run_query(self, sql: str) -> int:
        """Exécute une requête SQL arbitraire (ex: CREATE TABLE AS). Retourne le rowcount."""
        cursor = self._cursor()
        cursor.execute(sql)
        count = cursor.rowcount if cursor.rowcount >= 0 else 0
        logger.info(f"run_query : {count} lignes affectées")
        return count

    def query_as_dataframe(self, sql: str) -> pd.DataFrame:
        """Exécute un SELECT et retourne les résultats en DataFrame."""
        cursor = self._cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(rows, columns=columns)

    def truncate(self, table: str, where: str = None) -> None:
        if where:
            sql = f"DELETE FROM {table} WHERE {where}"
        else:
            sql = f"TRUNCATE TABLE {table}"
        cursor = self._cursor()
        cursor.execute(sql)
        logger.info(sql)

    # ── Écriture ──────────────────────────────────────────────────────────────

    def bulk_insert(self, table: str, df: pd.DataFrame, step: int = 50_000) -> int:
        """INSERT INTO ... VALUES ... par batch. Utilisé après TRUNCATE (full reload).

        Le paramètre `step` est un maximum souhaité ; la taille effective est réduite
        automatiquement pour que chaque requête reste sous _TRINO_MAX_QUERY_LEN.
        """
        df = df.copy()
        df.columns = df.columns.str.replace(".", "_").str.lower()
        total = 0

        # ── Estimation du nombre de chars par ligne (sur un échantillon de 10 lignes) ──
        sample_size = min(10, len(df))
        sample_values = _format_values(df.iloc[:sample_size])
        cols_str = ",".join(df.columns.tolist())
        prefix_len = len(f"INSERT INTO {table} ({cols_str}) VALUES ")
        chars_per_row = max(1, (prefix_len + len(sample_values)) // sample_size)
        safe_step = max(1, int((_TRINO_MAX_QUERY_LEN - prefix_len) // chars_per_row * 0.9))
        effective_step = min(step, safe_step)
        if effective_step < step:
            logger.info(
                f"bulk_insert {table}: step réduit de {step} à {effective_step} "
                f"(~{chars_per_row} chars/ligne, limite Trino)"
            )

        batch_num = 0
        for start in range(0, len(df), effective_step):
            df_batch = df[start : start + effective_step]
            # Exclure les colonnes entièrement nulles : Trino ne peut pas inférer leur type
            # dans la clause VALUES (type "unknown") et lève TYPE_MISMATCH.
            present_cols = [c for c in df_batch.columns if df_batch[c].notna().any()]
            if len(present_cols) < len(df_batch.columns):
                df_batch = df_batch[present_cols]
            values = _format_values(df_batch)
            cols = ",".join(df_batch.columns.tolist())
            sql = f"INSERT INTO {table} ({cols}) VALUES {values}"
            cursor = self._cursor()
            cursor.execute(sql)
            batch_count = cursor.rowcount if cursor.rowcount >= 0 else len(df_batch)
            total += batch_count
            batch_num += 1
            logger.info(f"Batch {batch_num}: {batch_count} lignes insérées")

        logger.info(f"Total bulk_insert {table}: {total} lignes")
        return total

    def delete_rows(self, table: str, primary_keys: list, df: pd.DataFrame, step: int = 500) -> int:
        """
        MERGE DELETE : supprime dans la table les lignes dont les clés primaires
        correspondent à celles du DataFrame.

        Utilise la même logique de typage que upsert (_format_values) pour garantir
        que les littéraux Trino matchent les types des colonnes PK.
        """
        df = df[primary_keys].copy()
        df.columns = df.columns.str.replace(".", "_").str.lower()
        total = 0

        for start in range(0, len(df), step):
            df_batch = df[start : start + step]
            values = _format_values(df_batch)
            cols = df_batch.columns.tolist()

            sql = (
                f"MERGE INTO {table} "
                f"USING (VALUES {values}) AS tmp ({','.join(cols)}) "
                f"ON {' AND '.join(f'{table}.{f} IS NOT DISTINCT FROM tmp.{f}' for f in primary_keys)} "
                f"WHEN MATCHED THEN DELETE"
            )
            cursor = self._cursor()
            cursor.execute(sql)
            batch_count = cursor.rowcount if cursor.rowcount >= 0 else 0
            total += batch_count
            logger.info(f"Delete batch {start // step + 1}: {batch_count} lignes supprimées")

        logger.info(f"Total delete_rows {table}: {total} lignes supprimées")
        return total

    def upsert(self, table: str, primary_keys: list, df: pd.DataFrame, step: int = 500) -> int:
        """MERGE INTO ... (upsert) par batch. Utilisé pour la synchronisation incrémentale."""
        df = df.copy()
        df.columns = df.columns.str.replace(".", "_").str.lower()
        pks_normalized = [pk.replace(".", "_").lower() for pk in primary_keys]
        df = df.drop_duplicates(subset=pks_normalized, keep="last")
        total = 0

        for start in range(0, len(df), step):
            df_batch = df[start : start + step]
            values = _format_values(df_batch)
            cols = df_batch.columns.tolist()

            sql = (
                f"MERGE INTO {table} "
                f"USING (VALUES {values}) AS tmp ({','.join(cols)}) "
                f"ON {' AND '.join([f'{table}.{f} IS NOT DISTINCT FROM tmp.{f}' for f in primary_keys])} "
                f"WHEN MATCHED THEN UPDATE SET {','.join([f'{f}=tmp.{f}' for f in cols if f not in primary_keys])} "
                f"WHEN NOT MATCHED THEN INSERT ({','.join(cols)}) VALUES ({','.join([f'tmp.{f}' for f in cols])})"
            )
            cursor = self._cursor()
            cursor.execute(sql)
            batch_count = cursor.rowcount if cursor.rowcount >= 0 else len(df_batch)
            total += batch_count
            logger.info(f"Batch {start // step + 1}: {batch_count} lignes upsertées")

        logger.info(f"Total upsert {table}: {total} lignes")
        return total

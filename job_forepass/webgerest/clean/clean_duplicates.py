from forepaas.dwh import connect
from forepaas.core.settings import PARAMS
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
prefix_table = PARAMS['PREFIX_TABLE']
environement = PARAMS['ENVIRONNEMENT_CLIENT']
dataset_cible = f"dwh/db_mg6jk45h_{environement}/"

# Toutes les tables et leurs clés primaires
TABLES = [
    (f"{prefix_table}article",          ["pk"]),
    (f"{prefix_table}categ",            ["pk"]),
    (f"{prefix_table}descfic",          ["nomfic", "login_group"]),
    (f"{prefix_table}detail_article",   ["pk"]),
    (f"{prefix_table}detpland",         ["pk"]),
    (f"{prefix_table}effect",           ["pk"]),
    (f"{prefix_table}famart",           ["pk"]),
    (f"{prefix_table}feuille",          ["pk"]),
    (f"{prefix_table}fitech",           ["pk"]),
    (f"{prefix_table}fourn",            ["pk"]),
    (f"{prefix_table}gaspi_saisie_gen", ["pk"]),
    (f"{prefix_table}label",            ["pk"]),
    (f"{prefix_table}mvtart",           ["pk"]),
    (f"{prefix_table}mvtart_det",       ["pk"]),
    (f"{prefix_table}ntarif",           ["pk"]),
    (f"{prefix_table}plandis",          ["pk"]),
    (f"{prefix_table}sfaart",           ["pk"]),
    (f"{prefix_table}trimestre",        ["pk"]),
    (f"{prefix_table}typss1",           ["pk"]),
    (f"{prefix_table}typss2",           ["pk"]),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_duplicate_rows(source, table: str, pk_cols: list) -> int:
    """Retourne le nombre de lignes en trop (sum des copies - 1 par groupe dupliqué)."""
    pk_group = ", ".join(pk_cols)
    result = source.query(f"""
        SELECT COALESCE(SUM(cnt - 1), 0) AS nb_extra
        FROM (
            SELECT COUNT(*) AS cnt
            FROM {table}
            GROUP BY {pk_group}
            HAVING COUNT(*) > 1
        )
    """)
    return int(result.iloc[0, 0]) if not result.empty else 0


def log_duplicate_sample(source, table: str, pk_cols: list):
    """Log un aperçu des PKs dupliquées (max 20)."""
    pk_group = ", ".join(pk_cols)
    try:
        sample = source.query(f"""
            SELECT {pk_group}, COUNT(*) AS nb_copies
            FROM {table}
            GROUP BY {pk_group}
            HAVING COUNT(*) > 1
            ORDER BY nb_copies DESC
            LIMIT 20
        """)
        if not sample.empty:
            logger.warning(
                f"  Aperçu des PKs dupliquées dans {table} :\n"
                f"{sample.to_string(index=False)}"
            )
    except Exception as e:
        logger.warning(f"  Impossible de récupérer l'aperçu pour {table} : {e}")


def deduplicate_table(source, table: str, pk_cols: list):
    """
    Supprime les doublons en ne conservant qu'une ligne par PK.

    Stratégie (compatible Trino/Iceberg) :
      1. Créer une table temporaire contenant UNE seule ligne par PK dupliquée
         (via ROW_NUMBER, en conservant la première occurrence arbitraire).
      2. Supprimer toutes les occurrences de ces PKs dans la table principale.
      3. Réinsérer la ligne conservée depuis la table temporaire.
      4. Supprimer la table temporaire.
    """
    pk_group = ", ".join(pk_cols)
    temp_table = f"{table}_dedup_tmp"

    # Récupérer la liste des colonnes de la table cible
    col_df = source.query(f"SELECT * FROM {table} LIMIT 0")
    col_list = ", ".join(col_df.columns)

    try:
        # 1. Table temporaire : une ligne par PK dupliquée
        source.query(f"DROP TABLE IF EXISTS {temp_table}")
        source.query(f"""
            CREATE TABLE {temp_table} AS
            SELECT {col_list}
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY {pk_group} ORDER BY {pk_group}) AS _rn
                FROM {table}
                WHERE ({pk_group}) IN (
                    SELECT {pk_group}
                    FROM {table}
                    GROUP BY {pk_group}
                    HAVING COUNT(*) > 1
                )
            )
            WHERE _rn = 1
        """)
        logger.info(f"  Table temporaire {temp_table} créée.")

        # 2. Supprimer toutes les occurrences des PKs dupliquées
        source.query(f"""
            DELETE FROM {table}
            WHERE ({pk_group}) IN (
                SELECT {pk_group} FROM {temp_table}
            )
        """)
        logger.info(f"  Toutes les copies des PKs dupliquées supprimées de {table}.")

        # 3. Réinsérer la ligne conservée
        source.query(f"""
            INSERT INTO {table} ({col_list})
            SELECT {col_list} FROM {temp_table}
        """)
        logger.info(f"  Lignes dédoublonnées réinsérées dans {table}.")

    finally:
        try:
            source.query(f"DROP TABLE IF EXISTS {temp_table}")
            logger.info(f"  Table temporaire {temp_table} supprimée.")
        except Exception as e:
            logger.warning(f"  Impossible de supprimer {temp_table} : {e}")


# ---------------------------------------------------------------------------
# Point d'entrée ForePaaS
# ---------------------------------------------------------------------------

def customfunc(event):
    source = connect(dataset_cible)
    total_deleted = 0
    tables_cleaned = 0

    logger.info("=== Job clean_duplicates démarré ===")

    for table, pk_cols in TABLES:
        pk_display = ", ".join(pk_cols)
        logger.info(f"--- {table} (PK : {pk_display}) ---")

        try:
            nb_extra = count_duplicate_rows(source, table, pk_cols)
        except Exception as e:
            logger.warning(f"  Impossible de vérifier {table} (table absente ?) : {e}")
            continue

        if nb_extra == 0:
            logger.info("  Aucun doublon, rien à faire.")
            continue

        tables_cleaned += 1
        total_deleted += nb_extra
        logger.warning(f"  {nb_extra} ligne(s) en doublon détectée(s)")

        log_duplicate_sample(source, table, pk_cols)

        try:
            deduplicate_table(source, table, pk_cols)
            logger.info(f"  Dédoublonnage de {table} terminé.")
        except Exception as e:
            logger.error(f"  Erreur lors du dédoublonnage de {table} : {e}")

    logger.info(
        f"=== Job terminé — {total_deleted} ligne(s) en trop supprimée(s) "
        f"dans {tables_cleaned} table(s) ==="
    )

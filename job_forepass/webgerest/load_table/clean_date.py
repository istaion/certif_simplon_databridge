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

# Tables à nettoyer : (nom_table, colonne_date, clé_primaire_pour_log)
TABLES_TO_CLEAN = [
    (
        f"{prefix_table}feuille",
        "efdate",
        ["fecleunik","code_site","login_site"],       
        "NOW() + INTERVAL '1' YEAR",
    ),
    (
        f"{prefix_table}effect",
        "efdate",
        ["id_effect","code_site","login_site"],        
        "NOW()",
    ),
    (
        f"{prefix_table}mvtart",
        "dtemvt",
        ["mvcleunik","code_site","login_site"],          
        "NOW()",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_future_rows(source, table: str, date_col: str, threshold: str) -> int:
    """Compte les lignes avec date > threshold."""
    result = source.query(f"""
        SELECT COUNT(*) AS nb
        FROM {table}
        WHERE CAST({date_col} AS TIMESTAMP) > {threshold}
    """)
    return int(result.iloc[0, 0]) if not result.empty else 0


def log_future_rows(source, table: str, date_col: str, pk_cols: list, threshold: str):
    """Log un aperçu des lignes concernées (max 50)."""
    pk_select = ", ".join(pk_cols) + ", " if pk_cols else ""
    try:
        sample = source.query(f"""
            SELECT {pk_select}{date_col}
            FROM {table}
            WHERE CAST({date_col} AS TIMESTAMP) > {threshold}
            ORDER BY CAST({date_col} AS TIMESTAMP) DESC
            LIMIT 50
        """)
        if not sample.empty:
            logger.warning(f"  Aperçu des lignes futures dans {table} :\n{sample.to_string(index=False)}")
    except Exception as e:
        logger.warning(f"  Impossible de récupérer l'aperçu pour {table} : {e}")


def delete_future_rows(source, table: str, date_col: str, threshold: str) -> int:
    """Supprime les lignes avec date > threshold et retourne le nombre de lignes supprimées."""
    result = source.query(f"""
        DELETE FROM {table}
        WHERE CAST({date_col} AS TIMESTAMP) > {threshold}
    """)
    try:
        deleted = int(result.iloc[0, 0]) if not result.empty else 0
    except Exception:
        deleted = -1  # inconnu mais DELETE exécuté
    return deleted


# ---------------------------------------------------------------------------
# Point d'entrée ForePaaS
# ---------------------------------------------------------------------------

def customfunc(event):
    source = connect(dataset_cible)
    total_deleted = 0

    logger.info("=== Job clean_future_dates démarré ===")
    logger.info("Seuil : feuille -> NOW() + 1 an | effect / mvtart -> NOW()")

    for table, date_col, pk_cols, threshold in TABLES_TO_CLEAN:
        logger.info(f"--- {table} (colonne : {date_col}, seuil : {threshold}) ---")

        # 1. Compter
        nb = count_future_rows(source, table, date_col, threshold)
        if nb == 0:
            logger.info(f"  Aucune ligne future trouvée, rien à faire.")
            continue

        logger.warning(f"  {nb} ligne(s) avec {date_col} > {threshold} détectée(s)")

        # 2. Logger un aperçu
        log_future_rows(source, table, date_col, pk_cols, threshold)

        # 3. Supprimer
        deleted = delete_future_rows(source, table, date_col, threshold)
        if deleted == -1:
            logger.info(f"  DELETE exécuté (nb de lignes supprimées non rapporté par Trino)")
        else:
            logger.info(f"  {deleted} ligne(s) supprimée(s) dans {table}")
            total_deleted += deleted

    logger.info(f"=== Job terminé — {total_deleted} ligne(s) supprimée(s) au total ===")
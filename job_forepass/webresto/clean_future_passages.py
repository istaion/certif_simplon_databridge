import logging
import time

from forepaas.dwh import connect
from forepaas.core.settings import PARAMS

logger = logging.getLogger(__name__)

JOB_NAME = "clean_future_passages"


def customfunc(event):
    prefix_table = PARAMS["PREFIX_TABLE"]
    environnement_client = PARAMS["ENVIRONNEMENT_CLIENT"]
    dataset_cible = f"dwh/db_mg6jk45h_{environnement_client}/"
    p = prefix_table

    logger.info(f"Démarrage du job '{JOB_NAME}' — suppression des passages avec date > aujourd'hui + 1 mois")
    t0 = time.time()

    source = connect(dataset_cible)

    sql_passage = f"""
DELETE FROM {p}passage
WHERE date > current_date + INTERVAL '1' MONTH
"""

    sql_passage_partner = f"""
DELETE FROM {p}passage_partner
WHERE date > current_date + INTERVAL '1' MONTH
"""

    try:
        source.query(sql_passage)
        logger.info(f"[{JOB_NAME}] passage nettoyée")

        source.query(sql_passage_partner)
        logger.info(f"[{JOB_NAME}] passage_partner nettoyée")

        duration = round(time.time() - t0, 2)
        logger.info(f"[{JOB_NAME}] OK — {duration}s")
    except Exception as e:
        logger.error(f"[{JOB_NAME}] Erreur fatale : {type(e).__name__}: {e}")
        raise

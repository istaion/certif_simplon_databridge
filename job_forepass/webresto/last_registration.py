import logging
import time

from forepaas.dwh import connect
from forepaas.core.settings import PARAMS

logger = logging.getLogger(__name__)

JOB_NAME = "last_registration"

prefix_table = PARAMS["PREFIX_TABLE"]
environnement_client = PARAMS["ENVIRONNEMENT_CLIENT"]
dataset_cible = f"dwh/db_mg6jk45h_{environnement_client}/"

p = prefix_table


def customfunc(event):
    logger.info(f"Démarrage du job '{JOB_NAME}'")
    t0 = time.time()

    source = connect(dataset_cible)

    sql = f"""
        MERGE INTO {p}lastregistration target
        USING (
            SELECT MAX(created_at) AS lastdate, id_school_year
            FROM {p}registration
            GROUP BY id_school_year
        ) src
        ON target.id_school_year = src.id_school_year
        WHEN MATCHED THEN UPDATE SET
            lastdate = src.lastdate
        WHEN NOT MATCHED THEN INSERT (id_school_year, lastdate)
        VALUES (src.id_school_year, src.lastdate)
    """

    try:
        source.query(sql)
        duration = round(time.time() - t0, 2)
        logger.info(f"[{JOB_NAME}] OK — {duration}s")
    except Exception as e:
        logger.error(f"[{JOB_NAME}] Erreur fatale : {type(e).__name__}: {e}")
        raise

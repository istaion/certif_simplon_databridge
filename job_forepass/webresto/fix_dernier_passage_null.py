import logging
import time

from forepaas.dwh import connect
from forepaas.core.settings import PARAMS

logger = logging.getLogger(__name__)

JOB_NAME = "dernier_passage"


def customfunc(event):
    prefix_table = PARAMS["PREFIX_TABLE"]
    environnement_client = PARAMS["ENVIRONNEMENT_CLIENT"]
    dataset_cible = f"dwh/db_mg6jk45h_{environnement_client}/"
    p = prefix_table

    logger.info(f"Démarrage du job '{JOB_NAME}' — inclut les établissements sans passage (NULL)")
    t0 = time.time()

    sql = f"""
CREATE OR REPLACE TABLE {p}dernier_passage AS
WITH combined_passages AS (
    SELECT date, id_organization FROM {p}passage
    UNION ALL
    SELECT date, id_organization FROM {p}passage_partner
),
latest_date_per_org AS (
    SELECT id_organization, MAX(date) AS date_dernier_passage
    FROM combined_passages
    GROUP BY id_organization
)
SELECT
    l.date_dernier_passage  AS last_date,
    o.name                  AS nom_etablissement,
    o.vague                 AS vague_demarrage,
    o.access_software       AS editeur_acces,
    o.ips,
    o.rne                   AS uai
FROM {p}organization o
LEFT JOIN latest_date_per_org l ON l.id_organization = o.id_organization
ORDER BY l.date_dernier_passage DESC NULLS LAST, o.name
"""

    source = connect(dataset_cible)
    try:
        source.query(sql)
        duration = round(time.time() - t0, 2)
        logger.info(f"[{JOB_NAME}] OK — {duration}s")
    except Exception as e:
        logger.error(f"[{JOB_NAME}] Erreur fatale : {type(e).__name__}: {e}")
        raise

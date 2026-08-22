import logging
import time

from forepaas.dwh import connect
from forepaas.core.settings import PARAMS

logger = logging.getLogger(__name__)

JOB_NAME = "tarification_3_by_school_year"

_WITH_PROPS = """
WITH (
    format = 'PARQUET',
    extra_properties = MAP(
        ARRAY[
            'write.target-file-size-bytes',
            'write.metadata.delete-after-commit.enabled',
            'write.metadata.previous-versions-max'
        ],
        ARRAY['268435456', 'true', '50']
    )
)"""


def _sql(p: str, school_year_id: int) -> str:
    return f"""
CREATE OR REPLACE TABLE {p}tarification_3_sy{school_year_id}{_WITH_PROPS}
AS
SELECT *
FROM {p}tarification_3
WHERE id_school_year = {school_year_id}
"""


def customfunc(event):
    prefix_table = PARAMS["PREFIX_TABLE"]
    environnement_client = PARAMS["ENVIRONNEMENT_CLIENT"]
    dataset_cible = f"dwh/db_mg6jk45h_{environnement_client}/"
    p = prefix_table

    if "centre" not in environnement_client:
        raise ValueError(
            f"[{JOB_NAME}] environnement_client={environnement_client!r} non supporté "
            f"(tarification_3 disponible uniquement pour 'centre')"
        )

    logger.info(f"Démarrage du job '{JOB_NAME}' — env={environnement_client}")

    source = connect(dataset_cible)

    df = source.query(
        f"SELECT DISTINCT id_school_year FROM {p}vague"
        f" WHERE id_school_year IS NOT NULL"
        f" ORDER BY id_school_year"
    )

    if df is None or df.empty:
        logger.info(f"[{JOB_NAME}] Aucune année scolaire trouvée — rien à faire.")
        return

    school_year_ids = df["id_school_year"].tolist()
    logger.info(f"[{JOB_NAME}] Années scolaires à traiter : {school_year_ids}")

    for sy_id in school_year_ids:
        sql = _sql(p, sy_id)
        t0 = time.time()
        try:
            source.query(sql)
            duration = round(time.time() - t0, 2)
            logger.info(f"[{JOB_NAME}] tarification_3_sy{sy_id} OK — {duration}s")
        except Exception as e:
            logger.error(f"[{JOB_NAME}] tarification_3_sy{sy_id} Erreur : {type(e).__name__}: {e}")
            raise

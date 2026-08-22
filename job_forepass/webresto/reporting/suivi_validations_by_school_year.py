import logging
import time

from forepaas.dwh import connect
from forepaas.core.settings import PARAMS

logger = logging.getLogger(__name__)

JOB_NAME = "suivi_validations_by_school_year"

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


def _sql_centre(p: str, school_year_id: int) -> str:
    return f"""
CREATE OR REPLACE TABLE {p}suivi_validations_sy{school_year_id}{_WITH_PROPS}
AS
WITH agg_day AS (
    SELECT
        DATE(h.created_at)                                                          AS jour,
        v.id_vague,
        v.name                                                                      AS nom_vague,
        v.id_school_year,
        sy.label                                                                    AS school_year,
        o.id_organization,
        o.access_software,
        o.name                                                                      AS nom_etablissement,
        o.type,
        o.department,
        o.ips,
        COUNT(*)                                                                    AS nb_val_jour,
        COUNT(CASE WHEN r.tranche_id = 1 THEN 1 END)                               AS nb_t1_jour,
        COUNT(CASE WHEN r.tranche_id = 2 THEN 1 END)                               AS nb_t2_jour,
        COUNT(CASE WHEN r.tranche_id = 3 THEN 1 END)                               AS nb_t3_jour,
        COUNT(CASE WHEN r.tranche_id = 4 THEN 1 END)                               AS nb_t4_jour
    FROM {p}history h
    JOIN {p}registration r      ON h.registration_id  = r.id
    JOIN {p}session s           ON r.id_session       = s.id
    JOIN {p}vague v             ON s.id_vague          = v.id_vague
    LEFT JOIN {p}school_year sy ON sy.school_year_id  = v.id_school_year
    JOIN {p}organization o      ON s.id_organization   = o.id_organization
    LEFT JOIN {p}subgroup sg    ON r.subgroup_id       = sg.id_subgroup
    WHERE h.event = 'REGISTRATION_APPROVED'
      AND r.status IN ('merged', 'validated')
      AND DATE(h.created_at) >= v.start_date
      AND (sg.id_group IS NULL OR sg.id_group != 2)
      AND v.id_school_year = {school_year_id}
    GROUP BY
        DATE(h.created_at), v.id_vague, v.name, v.id_school_year, sy.label,
        o.id_organization, o.access_software, o.name, o.type, o.department, o.ips
),
all_days AS (
    SELECT DISTINCT id_vague, jour FROM agg_day
),
all_etablissements AS (
    SELECT DISTINCT
        id_vague, nom_vague, id_school_year, school_year,
        id_organization, access_software, nom_etablissement, type, department, ips
    FROM agg_day
),
all_combinations AS (
    SELECT
        d.jour,
        e.id_vague,
        e.nom_vague,
        e.id_school_year,
        e.school_year,
        e.id_organization,
        e.access_software,
        e.nom_etablissement,
        e.type,
        e.department,
        e.ips
    FROM all_days d
    JOIN all_etablissements e ON d.id_vague = e.id_vague
)
SELECT
    c.jour,
    c.id_vague,
    c.nom_vague,
    c.id_school_year,
    c.school_year,
    c.id_organization,
    c.access_software,
    c.nom_etablissement,
    c.type,
    c.department,
    c.ips,
    SUM(COALESCE(a.nb_val_jour, 0)) OVER w  AS dossiers_valides_cumul,
    SUM(COALESCE(a.nb_t1_jour,  0)) OVER w  AS dossiers_valides_tranche1_cumul,
    SUM(COALESCE(a.nb_t2_jour,  0)) OVER w  AS dossiers_valides_tranche2_cumul,
    SUM(COALESCE(a.nb_t3_jour,  0)) OVER w  AS dossiers_valides_tranche3_cumul,
    SUM(COALESCE(a.nb_t4_jour,  0)) OVER w  AS dossiers_valides_tranche4_cumul
FROM all_combinations c
LEFT JOIN agg_day a
    ON  c.jour            = a.jour
    AND c.id_vague        = a.id_vague
    AND c.id_organization = a.id_organization
WINDOW w AS (
    PARTITION BY c.id_vague, c.id_organization ORDER BY c.jour ROWS UNBOUNDED PRECEDING
)
ORDER BY c.jour, c.nom_etablissement
"""


def customfunc(event):
    prefix_table = PARAMS["PREFIX_TABLE"]
    environnement_client = PARAMS["ENVIRONNEMENT_CLIENT"]
    dataset_cible = f"dwh/db_mg6jk45h_{environnement_client}/"
    p = prefix_table

    if "centre" not in environnement_client:
        raise ValueError(
            f"[{JOB_NAME}] environnement_client={environnement_client!r} non supporté "
            f"(attendu : contient 'centre')"
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
        t0 = time.time()
        try:
            source.query(_sql_centre(p, sy_id))
            duration = round(time.time() - t0, 2)
            logger.info(f"[{JOB_NAME}] suivi_validations_sy{sy_id} OK — {duration}s")
        except Exception as e:
            logger.error(f"[{JOB_NAME}] suivi_validations_sy{sy_id} Erreur : {type(e).__name__}: {e}")
            raise

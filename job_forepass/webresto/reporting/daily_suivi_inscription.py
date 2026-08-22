import logging
import time

from forepaas.dwh import connect
from forepaas.core.settings import PARAMS

logger = logging.getLogger(__name__)

JOB_NAME = "daily_suivi_inscription"

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


def _sql_centre(p: str) -> str:
    return f"""
CREATE OR REPLACE TABLE {p}daily_suivi_inscription{_WITH_PROPS}
AS
SELECT
    event_date                          AS jour,
    id_vague,
    nom_vague,
    id_school_year,
    school_year,
    id_organization,
    access_software,
    nom_etablissement,
    type,
    department,
    ips,
    SUM(is_inscription)                 AS nb_inscriptions_jour,
    SUM(is_depot)                       AS nb_depots_jour,
    SUM(is_validation)                  AS nb_validations_jour,
    SUM(is_validation_tranche1)         AS nb_validations_tranche1_jour,
    SUM(is_validation_tranche2)         AS nb_validations_tranche2_jour,
    SUM(is_validation_tranche3)         AS nb_validations_tranche3_jour,
    SUM(is_validation_tranche4)         AS nb_validations_tranche4_jour,
    SUM(is_validation_hors_tranche)     AS nb_validations_hors_tranche_jour
FROM (
    -- Inscriptions : date de création du dossier
    SELECT
        DATE(r.created_at)              AS event_date,
        v.id_vague,
        v.name                          AS nom_vague,
        v.id_school_year,
        sy.label                        AS school_year,
        o.id_organization,
        o.access_software,
        o.name                          AS nom_etablissement,
        o.type,
        o.department,
        o.ips,
        1 AS is_inscription,
        0 AS is_depot,
        0 AS is_validation,
        0 AS is_validation_tranche1,
        0 AS is_validation_tranche2,
        0 AS is_validation_tranche3,
        0 AS is_validation_tranche4,
        0 AS is_validation_hors_tranche
    FROM {p}registration r
    JOIN {p}session s         ON r.id_session       = s.id
    JOIN {p}vague v           ON s.id_vague          = v.id_vague
    LEFT JOIN {p}school_year sy ON sy.school_year_id = v.id_school_year
    JOIN {p}organization o    ON s.id_organization   = o.id_organization
    LEFT JOIN {p}subgroup sg  ON r.subgroup_id       = sg.id_subgroup
    WHERE r.status != 'canceled'
      AND DATE(r.created_at) >= v.start_date
      AND (sg.id_group IS NULL OR sg.id_group != 2)

    UNION ALL

    -- Dépôts : premier événement history par dossier
    SELECT
        DATE(h.created_at)              AS event_date,
        v.id_vague,
        v.name                          AS nom_vague,
        v.id_school_year,
        sy.label                        AS school_year,
        o.id_organization,
        o.access_software,
        o.name                          AS nom_etablissement,
        o.type,
        o.department,
        o.ips,
        0 AS is_inscription,
        1 AS is_depot,
        0 AS is_validation,
        0 AS is_validation_tranche1,
        0 AS is_validation_tranche2,
        0 AS is_validation_tranche3,
        0 AS is_validation_tranche4,
        0 AS is_validation_hors_tranche
    FROM (
        SELECT registration_id, MIN(created_at) AS created_at
        FROM {p}history
        GROUP BY registration_id
    ) h
    JOIN {p}registration r    ON h.registration_id  = r.id
    JOIN {p}session s         ON r.id_session       = s.id
    JOIN {p}vague v           ON s.id_vague          = v.id_vague
    LEFT JOIN {p}school_year sy ON sy.school_year_id = v.id_school_year
    JOIN {p}organization o    ON s.id_organization   = o.id_organization
    LEFT JOIN {p}subgroup sg  ON r.subgroup_id       = sg.id_subgroup
    WHERE r.status != 'canceled'
      AND DATE(h.created_at) >= v.start_date
      AND (sg.id_group IS NULL OR sg.id_group != 2)

    UNION ALL

    -- Validations : événement REGISTRATION_APPROVED, tranche depuis r.tranche_id
    SELECT
        DATE(h.created_at)              AS event_date,
        v.id_vague,
        v.name                          AS nom_vague,
        v.id_school_year,
        sy.label                        AS school_year,
        o.id_organization,
        o.access_software,
        o.name                          AS nom_etablissement,
        o.type,
        o.department,
        o.ips,
        0 AS is_inscription,
        0 AS is_depot,
        1 AS is_validation,
        CASE WHEN r.tranche_id = 1 THEN 1 ELSE 0 END                                       AS is_validation_tranche1,
        CASE WHEN r.tranche_id = 2 THEN 1 ELSE 0 END                                       AS is_validation_tranche2,
        CASE WHEN r.tranche_id = 3 THEN 1 ELSE 0 END                                       AS is_validation_tranche3,
        CASE WHEN r.tranche_id = 4 THEN 1 ELSE 0 END                                       AS is_validation_tranche4,
        CASE WHEN r.tranche_id IS NULL OR r.tranche_id NOT IN (1,2,3,4) THEN 1 ELSE 0 END AS is_validation_hors_tranche
    FROM {p}history h
    JOIN {p}registration r    ON h.registration_id  = r.id
    JOIN {p}session s         ON r.id_session       = s.id
    JOIN {p}vague v           ON s.id_vague          = v.id_vague
    LEFT JOIN {p}school_year sy ON sy.school_year_id = v.id_school_year
    JOIN {p}organization o    ON s.id_organization   = o.id_organization
    LEFT JOIN {p}subgroup sg  ON r.subgroup_id       = sg.id_subgroup
    WHERE h.event = 'REGISTRATION_APPROVED'
      AND r.status IN ('merged', 'validated')
      AND DATE(h.created_at) >= v.start_date
      AND (sg.id_group IS NULL OR sg.id_group != 2)
) events
GROUP BY
    event_date, id_vague, nom_vague, id_school_year, school_year, id_organization,
    access_software, nom_etablissement, type, department, ips
ORDER BY jour, nom_etablissement
"""


def _sql_93(p: str) -> str:
    return f"""
CREATE OR REPLACE TABLE {p}daily_suivi_inscription{_WITH_PROPS}
AS
SELECT
    event_date                          AS jour,
    id_vague,
    nom_vague,
    id_school_year,
    school_year,
    id_organization,
    nom_etablissement,
    SUM(is_inscription)                 AS nb_inscriptions_jour,
    SUM(is_depot)                       AS nb_depots_jour,
    SUM(is_validation)                  AS nb_validations_jour
FROM (
    -- Inscriptions
    SELECT
        DATE(r.created_at)              AS event_date,
        v.id_vague,
        v.name                          AS nom_vague,
        v.id_school_year,
        sy.label                        AS school_year,
        o.id_organization,
        o.name                          AS nom_etablissement,
        1 AS is_inscription,
        0 AS is_depot,
        0 AS is_validation
    FROM {p}registration r
    JOIN {p}session s         ON r.id_session       = s.id
    JOIN {p}vague v           ON s.id_vague          = v.id_vague
    LEFT JOIN {p}school_year sy ON sy.school_year_id = v.id_school_year
    JOIN {p}organization o    ON s.id_organization   = o.id_organization
    LEFT JOIN {p}user u       ON r.id_user           = u.id_user
    LEFT JOIN {p}subgroup sg  ON u.id_subgroup       = sg.id_subgroup
    WHERE r.status != 'canceled'
      AND DATE(r.created_at) >= v.start_date
      AND (sg.id_group IS NULL OR sg.id_group != 2)

    UNION ALL

    -- Dépôts
    SELECT
        DATE(h.created_at)              AS event_date,
        v.id_vague,
        v.name                          AS nom_vague,
        v.id_school_year,
        sy.label                        AS school_year,
        o.id_organization,
        o.name                          AS nom_etablissement,
        0 AS is_inscription,
        1 AS is_depot,
        0 AS is_validation
    FROM (
        SELECT registration_id, MIN(created_at) AS created_at
        FROM {p}history
        GROUP BY registration_id
    ) h
    JOIN {p}registration r    ON h.registration_id  = r.id
    JOIN {p}session s         ON r.id_session       = s.id
    JOIN {p}vague v           ON s.id_vague          = v.id_vague
    LEFT JOIN {p}school_year sy ON sy.school_year_id = v.id_school_year
    JOIN {p}organization o    ON s.id_organization   = o.id_organization
    LEFT JOIN {p}user u       ON r.id_user           = u.id_user
    LEFT JOIN {p}subgroup sg  ON u.id_subgroup       = sg.id_subgroup
    WHERE r.status != 'canceled'
      AND DATE(h.created_at) >= v.start_date
      AND (sg.id_group IS NULL OR sg.id_group != 2)

    UNION ALL

    -- Validations
    SELECT
        DATE(h.created_at)              AS event_date,
        v.id_vague,
        v.name                          AS nom_vague,
        v.id_school_year,
        sy.label                        AS school_year,
        o.id_organization,
        o.name                          AS nom_etablissement,
        0 AS is_inscription,
        0 AS is_depot,
        1 AS is_validation
    FROM {p}history h
    JOIN {p}registration r    ON h.registration_id  = r.id
    JOIN {p}session s         ON r.id_session       = s.id
    JOIN {p}vague v           ON s.id_vague          = v.id_vague
    LEFT JOIN {p}school_year sy ON sy.school_year_id = v.id_school_year
    JOIN {p}organization o    ON s.id_organization   = o.id_organization
    LEFT JOIN {p}user u       ON r.id_user           = u.id_user
    LEFT JOIN {p}subgroup sg  ON u.id_subgroup       = sg.id_subgroup
    WHERE h.event = 'REGISTRATION_APPROVED'
      AND r.status IN ('merged', 'validated')
      AND DATE(h.created_at) >= v.start_date
      AND (sg.id_group IS NULL OR sg.id_group != 2)
) events
GROUP BY
    event_date, id_vague, nom_vague, id_school_year, school_year,
    id_organization, nom_etablissement
ORDER BY jour, nom_etablissement
"""


def customfunc(event):
    prefix_table = PARAMS["PREFIX_TABLE"]
    environnement_client = PARAMS["ENVIRONNEMENT_CLIENT"]
    dataset_cible = f"dwh/db_mg6jk45h_{environnement_client}/"
    p = prefix_table

    if "centre" not in environnement_client and "93" not in environnement_client:
        raise ValueError(
            f"[{JOB_NAME}] environnement_client={environnement_client!r} non supporté "
            f"(attendu : contient 'centre' ou '93')"
        )

    logger.info(f"Démarrage du job '{JOB_NAME}' — env={environnement_client}")

    source = connect(dataset_cible)

    sql = _sql_centre(p) if "centre" in environnement_client else _sql_93(p)
    t0 = time.time()
    try:
        source.query(sql)
        duration = round(time.time() - t0, 2)
        logger.info(f"[{JOB_NAME}] daily_suivi_inscription OK — {duration}s")
    except Exception as e:
        logger.error(f"[{JOB_NAME}] daily_suivi_inscription Erreur : {type(e).__name__}: {e}")
        raise

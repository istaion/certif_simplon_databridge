import logging
import time

from forepaas.dwh import connect
from forepaas.core.settings import PARAMS

logger = logging.getLogger(__name__)

JOB_NAME = "suivi_inscriptions_by_school_year"

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
CREATE OR REPLACE TABLE {p}suivi_inscriptions_sy{school_year_id}{_WITH_PROPS}
AS
WITH daily_data AS (
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
          AND v.id_school_year = {school_year_id}

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
          AND v.id_school_year = {school_year_id}

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
          AND v.id_school_year = {school_year_id}
    ) events
    GROUP BY
        event_date, id_vague, nom_vague, id_school_year, school_year, id_organization,
        access_software, nom_etablissement, type, department, ips
),
all_days AS (
    SELECT DISTINCT id_vague, jour
    FROM daily_data
),
all_etablissements AS (
    SELECT DISTINCT
        id_vague,
        nom_vague,
        id_school_year,
        school_year,
        id_organization,
        access_software,
        nom_etablissement,
        type,
        department,
        ips
    FROM daily_data
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
),
filled_data AS (
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
        COALESCE(dd.nb_inscriptions_jour, 0)             AS nb_inscriptions_jour,
        COALESCE(dd.nb_depots_jour, 0)                   AS nb_depots_jour,
        COALESCE(dd.nb_validations_jour, 0)              AS nb_validations_jour,
        COALESCE(dd.nb_validations_tranche1_jour, 0)     AS nb_validations_tranche1_jour,
        COALESCE(dd.nb_validations_tranche2_jour, 0)     AS nb_validations_tranche2_jour,
        COALESCE(dd.nb_validations_tranche3_jour, 0)     AS nb_validations_tranche3_jour,
        COALESCE(dd.nb_validations_tranche4_jour, 0)     AS nb_validations_tranche4_jour,
        COALESCE(dd.nb_validations_hors_tranche_jour, 0) AS nb_validations_hors_tranche_jour
    FROM all_combinations c
    LEFT JOIN daily_data dd
        ON  c.jour            = dd.jour
        AND c.id_vague        = dd.id_vague
        AND c.id_organization = dd.id_organization
)
SELECT
    jour,
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
    SUM(nb_inscriptions_jour) OVER w             AS total_connexions_cumul,
    SUM(nb_depots_jour) OVER w                   AS dossiers_deposes_cumul,
    SUM(nb_validations_jour) OVER w              AS dossiers_valides_cumul,
    SUM(nb_validations_tranche1_jour) OVER w     AS dossiers_valides_cumul_tranche1,
    SUM(nb_validations_tranche2_jour) OVER w     AS dossiers_valides_cumul_tranche2,
    SUM(nb_validations_tranche3_jour) OVER w     AS dossiers_valides_cumul_tranche3,
    SUM(nb_validations_tranche4_jour) OVER w     AS dossiers_valides_cumul_tranche4,
    SUM(nb_validations_hors_tranche_jour) OVER w AS dossiers_valides_cumul_hors_tranche
FROM filled_data
WINDOW w AS (
    PARTITION BY id_vague, id_organization ORDER BY jour ROWS UNBOUNDED PRECEDING
)
ORDER BY jour, nom_etablissement
"""


def _sql_93(p: str, school_year_id: int) -> str:
    return f"""
CREATE OR REPLACE TABLE {p}suivi_inscriptions_sy{school_year_id}{_WITH_PROPS}
AS
WITH daily_data AS (
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
          AND v.id_school_year = {school_year_id}

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
          AND v.id_school_year = {school_year_id}

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
          AND v.id_school_year = {school_year_id}
    ) events
    GROUP BY
        event_date, id_vague, nom_vague, id_school_year, school_year,
        id_organization, nom_etablissement
),
all_days AS (
    SELECT DISTINCT id_vague, jour
    FROM daily_data
),
all_etablissements AS (
    SELECT DISTINCT
        id_vague,
        nom_vague,
        id_school_year,
        school_year,
        id_organization,
        nom_etablissement
    FROM daily_data
),
all_combinations AS (
    SELECT
        d.jour,
        e.id_vague,
        e.nom_vague,
        e.id_school_year,
        e.school_year,
        e.id_organization,
        e.nom_etablissement
    FROM all_days d
    JOIN all_etablissements e ON d.id_vague = e.id_vague
),
filled_data AS (
    SELECT
        c.jour,
        c.id_vague,
        c.nom_vague,
        c.id_school_year,
        c.school_year,
        c.id_organization,
        c.nom_etablissement,
        COALESCE(dd.nb_inscriptions_jour, 0) AS nb_inscriptions_jour,
        COALESCE(dd.nb_depots_jour, 0)       AS nb_depots_jour,
        COALESCE(dd.nb_validations_jour, 0)  AS nb_validations_jour
    FROM all_combinations c
    LEFT JOIN daily_data dd
        ON  c.jour            = dd.jour
        AND c.id_vague        = dd.id_vague
        AND c.id_organization = dd.id_organization
)
SELECT
    jour,
    id_vague,
    nom_vague,
    id_school_year,
    school_year,
    id_organization,
    nom_etablissement,
    SUM(nb_inscriptions_jour) OVER w AS total_connexions_cumul,
    SUM(nb_depots_jour)       OVER w AS dossiers_deposes_cumul,
    SUM(nb_validations_jour)  OVER w AS dossiers_valides_cumul
FROM filled_data
WINDOW w AS (
    PARTITION BY id_vague, id_organization ORDER BY jour ROWS UNBOUNDED PRECEDING
)
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
        sql = _sql_centre(p, sy_id) if "centre" in environnement_client else _sql_93(p, sy_id)
        t0 = time.time()
        try:
            source.query(sql)
            duration = round(time.time() - t0, 2)
            logger.info(f"[{JOB_NAME}] suivi_inscriptions_sy{sy_id} OK — {duration}s")
        except Exception as e:
            logger.error(f"[{JOB_NAME}] suivi_inscriptions_sy{sy_id} Erreur : {type(e).__name__}: {e}")
            raise

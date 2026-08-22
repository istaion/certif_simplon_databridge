import logging
import time

from forepaas.dwh import connect
from forepaas.core.settings import PARAMS

logger = logging.getLogger(__name__)

JOB_NAME = "tarification_2_by_school_year"

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
CREATE OR REPLACE TABLE {p}tarification_2_sy{school_year_id}{_WITH_PROPS}
AS
WITH Requetev1 AS (
    SELECT
        o.id_organization,
        o.name                          AS nom_etablissement,
        s.id_vague,
        v.name                          AS nom_vague,
        v.id_school_year,
        sy.label                        AS school_year,
        o.type,
        o.department,
        o.access_software,
        r.subgroup_id                   AS id_subgroup,
        uisy.label_subgroup             AS nom_sous_groupe,
        sg.facturation_type,
        CASE
            WHEN r.status IN ('validated', 'merged')
            THEN COALESCE(
                CONCAT('Tranche ', t.label),
                CONCAT('Tranche ', uisy.label_tranche),
                'Hors tranche'
            )
            ELSE 'Hors tranche'
        END                             AS tranche,
        r.status
    FROM {p}registration r
    INNER JOIN {p}session s      ON r.id_session = s.id
    INNER JOIN {p}organization o ON s.id_organization = o.id_organization
    LEFT JOIN  {p}vague v        ON s.id_vague = v.id_vague
    LEFT JOIN  {p}school_year sy ON sy.school_year_id = v.id_school_year
    LEFT JOIN  {p}user_info_school_year uisy ON uisy.id_user = r.id_user
                                           AND uisy.school_year_id = v.id_school_year
    LEFT JOIN  {p}tranche t      ON t.id_tranche = r.tranche_id
    LEFT JOIN  {p}subgroup sg    ON r.subgroup_id = sg.id_subgroup
    WHERE (sg.id_group IS NULL OR sg.id_group != 2)
      AND v.id_school_year = {school_year_id}
)
SELECT
    id_organization,
    nom_etablissement,
    id_vague,
    nom_vague,
    id_school_year,
    school_year,
    type,
    department,
    access_software,
    id_subgroup,
    nom_sous_groupe,
    tranche,
    COUNT(CASE WHEN status = 'created'   THEN 1 END) AS nb_created,
    COUNT(CASE WHEN status = 'validated' THEN 1 END) AS nb_validated,
    COUNT(CASE WHEN status = 'merged'    THEN 1 END) AS nb_merged,
    COUNT(CASE WHEN status = 'sent'      THEN 1 END) AS nb_sent,
    COUNT(CASE WHEN status = 'rejected'  THEN 1 END) AS nb_rejected,
    COUNT(CASE WHEN status = 'canceled'  THEN 1 END) AS nb_canceled,
    COUNT(CASE WHEN facturation_type = 'interne' THEN 1 END) AS nb_dossiers_interne,
    COUNT(CASE WHEN facturation_type = 'ticket'  THEN 1 END) AS nb_dossier_ticket,
    COUNT(CASE WHEN facturation_type = 'autre' OR facturation_type IS NULL THEN 1 END) AS nb_dossiers_autre
FROM Requetev1
GROUP BY
    id_organization,
    nom_etablissement,
    id_vague,
    nom_vague,
    id_school_year,
    school_year,
    type,
    department,
    access_software,
    id_subgroup,
    nom_sous_groupe,
    tranche
ORDER BY
    nom_vague,
    nom_etablissement,
    nom_sous_groupe
"""


def _sql_93(p: str, school_year_id: int) -> str:
    return f"""
CREATE OR REPLACE TABLE {p}tarification_2_sy{school_year_id}{_WITH_PROPS}
AS
WITH Requetev1 AS (
    SELECT
        o.id_organization,
        o.name                          AS nom_etablissement,
        s.id_vague,
        v.name                          AS nom_vague,
        v.id_school_year,
        sy.label                        AS school_year,
        o.type,
        o.department,
        o.access_software,
        r.subgroup_id                   AS id_subgroup,
        uisy.label_subgroup             AS nom_sous_groupe,
        uisy.label_group,
        COALESCE(t.label, uisy.label_tranche) AS tranche,
        r.status
    FROM {p}registration r
    INNER JOIN {p}session s      ON r.id_session = s.id
    INNER JOIN {p}organization o ON s.id_organization = o.id_organization
    LEFT JOIN  {p}vague v        ON s.id_vague = v.id_vague
    LEFT JOIN  {p}school_year sy ON sy.school_year_id = v.id_school_year
    LEFT JOIN  {p}user_info_school_year uisy ON uisy.id_user = r.id_user
                                           AND uisy.school_year_id = v.id_school_year
    LEFT JOIN  {p}tranche t      ON t.id_tranche = r.tranche_id
    WHERE v.id_school_year = {school_year_id}
)
SELECT
    id_organization,
    nom_etablissement,
    id_vague,
    nom_vague,
    id_school_year,
    school_year,
    type,
    department,
    access_software,
    id_subgroup,
    nom_sous_groupe,
    label_group,
    tranche,
    COUNT(CASE WHEN status = 'created'   THEN 1 END) AS nb_created,
    COUNT(CASE WHEN status = 'validated' THEN 1 END) AS nb_validated,
    COUNT(CASE WHEN status = 'merged'    THEN 1 END) AS nb_merged,
    COUNT(CASE WHEN status = 'sent'      THEN 1 END) AS nb_sent,
    COUNT(CASE WHEN status = 'rejected'  THEN 1 END) AS nb_rejected,
    COUNT(CASE WHEN status = 'canceled'  THEN 1 END) AS nb_canceled
FROM Requetev1
GROUP BY
    id_organization,
    nom_etablissement,
    id_vague,
    nom_vague,
    id_school_year,
    school_year,
    type,
    department,
    access_software,
    id_subgroup,
    nom_sous_groupe,
    label_group,
    tranche
ORDER BY
    nom_vague,
    nom_etablissement,
    nom_sous_groupe
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
            logger.info(f"[{JOB_NAME}] tarification_2_sy{sy_id} OK — {duration}s")
        except Exception as e:
            logger.error(f"[{JOB_NAME}] tarification_2_sy{sy_id} Erreur : {type(e).__name__}: {e}")
            raise

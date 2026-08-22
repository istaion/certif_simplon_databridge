import logging
import time

from forepaas.dwh import connect
from forepaas.core.settings import PARAMS

logger = logging.getLogger(__name__)

JOB_NAME = "tarification_1_by_school_year"

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
CREATE OR REPLACE TABLE {p}tarification_1_sc{school_year_id}{_WITH_PROPS}
AS
WITH Requetev1 AS (
    SELECT
        o.id_organization,
        o.name                          AS nom_etablissement,
        oe.total_enrollment,
        oe.intern_count,
        oe.social_tarif_beneficiaries,
        s.id_vague,
        v.name                          AS nom_vague,
        v.id_school_year,
        sy.label                        AS school_year,
        o.type,
        o.department,
        o.ips,
        o.rne,
        o.access_software,
        CASE
            WHEN r.status IN ('validated', 'merged')
            THEN COALESCE(
                CONCAT('Tranche ', t.label),
                CONCAT('Tranche ', uisy.label_tranche),
                'Hors tranche'
            )
            ELSE 'Hors tranche'
        END                             AS tranche,
        uisy.choice_bank_details,
        sg.facturation_type,
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
    LEFT JOIN  {p}organization_enrollment oe ON oe.organization_id = o.id_organization
                                           AND oe.school_year_id = v.id_school_year
    WHERE (sg.id_group IS NULL OR sg.id_group != 2)
      AND v.id_school_year = {school_year_id}
)
SELECT
    id_organization,
    nom_etablissement,
    total_enrollment                    AS effectif,
    intern_count                        AS effectif_interne,
    social_tarif_beneficiaries          AS effectif_cible,
    id_vague,
    nom_vague,
    id_school_year,
    school_year,
    type,
    department,
    ips,
    rne,
    access_software,
    COALESCE(facturation_type, 'autre') AS facturation_type,
    tranche,
    COUNT(CASE WHEN status IN ('validated', 'merged') THEN 1 END)                                   AS dossiers_valides,
    COUNT(CASE WHEN status NOT IN ('created', 'canceled') THEN 1 END)                               AS dossiers_deposes,
    COUNT(CASE WHEN status = 'rejected' THEN 1 END)                                                 AS dossiers_a_corriger,
    COUNT(CASE WHEN status = 'sent' THEN 1 END)                                                     AS dossiers_en_attente,
    COUNT(CASE WHEN status = 'created' THEN 1 END)                                                  AS connexion_sans_depot,
    COUNT(CASE WHEN status NOT IN ('created', 'canceled') AND choice_bank_details = 'Transmission automatique'    THEN 1 END)   AS transmission_automatique,
    COUNT(CASE WHEN status NOT IN ('created', 'canceled') AND choice_bank_details = 'Transmission manuelle impot' THEN 1 END)   AS transmission_manuelle,
    COUNT(CASE WHEN status NOT IN ('created', 'canceled') AND choice_bank_details = 'Pas de données fournies'     THEN 1 END)   AS pas_donnees_fournies,
    COUNT(CASE WHEN status NOT IN ('created', 'canceled') AND choice_bank_details = 'Refus de transmission'       THEN 1 END)   AS refus_transmission,
    COUNT(CASE WHEN status NOT IN ('created', 'canceled') AND choice_bank_details = 'Identité pivot'              THEN 1 END)   AS identite_pivot,
    COUNT(CASE WHEN status NOT IN ('created', 'canceled') AND choice_bank_details = 'Transmission manuelle CAF'   THEN 1 END)   AS avis_impot,
    COUNT(CASE WHEN status NOT IN ('created', 'canceled') AND choice_bank_details = 'Import départemental'        THEN 1 END)   AS import_departemental,
    COUNT(CASE WHEN status NOT IN ('created', 'canceled') AND choice_bank_details = 'Inscription papier'          THEN 1 END)   AS import_gestionnaire
FROM Requetev1
GROUP BY
    id_organization, nom_etablissement, total_enrollment, intern_count,
    social_tarif_beneficiaries, id_vague, nom_vague, id_school_year, school_year,
    type, department, ips, rne, access_software, facturation_type, tranche
ORDER BY nom_vague, nom_etablissement
"""


def _sql_93(p: str, school_year_id: int) -> str:
    return f"""
CREATE OR REPLACE TABLE {p}tarification_1_sc{school_year_id}{_WITH_PROPS}
AS
WITH Requetev1 AS (
    SELECT
        o.id_organization,
        o.name                          AS nom_etablissement,
        CONCAT(o.name, ' - ', o.city)   AS nom_ville,
        oe.total_enrollment,
        oe.intern_count,
        oe.social_tarif_beneficiaries,
        s.id_vague,
        v.name                          AS nom_vague,
        v.id_school_year,
        sy.label                        AS school_year,
        o.type,
        o.department,
        o.ips,
        o.rne,
        o.access_software,
        COALESCE(t.label, uisy.label_tranche) AS tranche,
        COALESCE(r.choice_bank_detail, uisy.choice_bank_details) AS mode_transmission,
        uisy.label_group,
        r.status
    FROM {p}registration r
    INNER JOIN {p}session s      ON r.id_session = s.id
    INNER JOIN {p}organization o ON s.id_organization = o.id_organization
    LEFT JOIN  {p}vague v        ON s.id_vague = v.id_vague
    LEFT JOIN  {p}school_year sy ON sy.school_year_id = v.id_school_year
    LEFT JOIN  {p}user_info_school_year uisy ON uisy.id_user = r.id_user
                                           AND uisy.school_year_id = v.id_school_year
    LEFT JOIN  {p}tranche t      ON t.id_tranche = r.tranche_id
    LEFT JOIN  {p}organization_enrollment oe ON oe.organization_id = o.id_organization
                                           AND oe.school_year_id = v.id_school_year
    WHERE v.id_school_year = {school_year_id}
)
SELECT
    id_organization,
    nom_etablissement,
    nom_ville,
    total_enrollment                    AS effectif,
    intern_count                        AS effectif_interne,
    social_tarif_beneficiaries          AS effectif_cible,
    id_vague,
    nom_vague,
    id_school_year,
    school_year,
    type,
    department,
    ips,
    rne,
    access_software,
    label_group,
    tranche,
    COUNT(CASE WHEN status = 'validated' THEN 1 END)                                                      AS dossiers_valides,
    COUNT(CASE WHEN status IN ('validated', 'merged') THEN 1 END)                                         AS dossiers_fusionnes,
    COUNT(CASE WHEN status NOT IN ('created', 'canceled') THEN 1 END)                                     AS dossiers_deposes,
    COUNT(CASE WHEN status = 'rejected' THEN 1 END)                                                       AS dossiers_a_corriger,
    COUNT(CASE WHEN status = 'sent' THEN 1 END)                                                           AS dossiers_en_attente,
    COUNT(CASE WHEN status = 'created' THEN 1 END)                                                        AS connexion_sans_depot,
    COUNT(CASE WHEN mode_transmission = 'Transmission automatique'    THEN 1 END)   AS transmission_automatique,
    COUNT(CASE WHEN mode_transmission = 'Transmission manuelle impot' THEN 1 END)   AS transmission_manuelle,
    COUNT(CASE WHEN mode_transmission = 'Pas de données fournies'     THEN 1 END)   AS pas_donnees_fournies,
    COUNT(CASE WHEN mode_transmission = 'Refus de transmission'       THEN 1 END)   AS refus_transmission,
    COUNT(CASE WHEN mode_transmission = 'Identité pivot'              THEN 1 END)   AS identite_pivot,
    COUNT(CASE WHEN mode_transmission = 'Transmission manuelle CAF'   THEN 1 END)   AS avis_impot,
    COUNT(CASE WHEN mode_transmission = 'Import départemental'        THEN 1 END)   AS import_departemental,
    COUNT(CASE WHEN mode_transmission = 'Inscription papier'          THEN 1 END)   AS import_gestionnaire
FROM Requetev1
GROUP BY
    id_organization, nom_etablissement, nom_ville, total_enrollment, intern_count,
    social_tarif_beneficiaries, id_vague, nom_vague, id_school_year, school_year,
    type, department, ips, rne, access_software, label_group, tranche
ORDER BY nom_vague, nom_etablissement
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
            logger.info(f"[{JOB_NAME}] tarification_1_sc{sy_id} OK — {duration}s")
        except Exception as e:
            logger.error(f"[{JOB_NAME}] tarification_1_sc{sy_id} Erreur : {type(e).__name__}: {e}")
            raise

import logging
import time

from forepaas.dwh import connect
from forepaas.core.settings import PARAMS

logger = logging.getLogger(__name__)

JOB_NAME = "tarification_3"


def customfunc(event):
    prefix_table = PARAMS["PREFIX_TABLE"]
    environnement_client = PARAMS["ENVIRONNEMENT_CLIENT"]
    dataset_cible = f"dwh/db_mg6jk45h_{environnement_client}/"
    p = prefix_table

    logger.info(f"Démarrage du job '{JOB_NAME}' — tranche actualisée (uisy) + colonne service + sans filtre déjeuner")
    t0 = time.time()

    sql = f"""
CREATE OR REPLACE TABLE {p}tarification_3 AS
WITH combined_data AS (
    SELECT
        p.date,
        p.id_organization,
        sg.facturation_type,
        CASE
            WHEN s.label = 'Petit déjeuner' THEN 'petit_dejeuner'
            WHEN s.label = 'Déjeuner'       THEN 'dejeuner'
            WHEN s.label = 'Diner'          THEN 'diner'
            ELSE 'autre'
        END AS service,
        CASE COALESCE(uisy.id_tranche, p.id_tranche)
            WHEN 1 THEN 'Tranche 1'
            WHEN 2 THEN 'Tranche 2'
            WHEN 3 THEN 'Tranche 3'
            WHEN 4 THEN 'Tranche 4'
            ELSE 'Hors tranche'
        END AS tranche,
        1 AS nb_passages
    FROM {p}passage p
    INNER JOIN {p}subgroup sg ON p.id_subgroup = sg.id_subgroup
    LEFT JOIN  {p}service s   ON p.id_service  = s.id_service
    LEFT JOIN  {p}school_year sy_match
        ON p.date BETWEEN sy_match.start_date AND sy_match.end_date
    LEFT JOIN  {p}user_info_school_year uisy
        ON uisy.id_user = p.id_user AND uisy.school_year_id = sy_match.school_year_id
    WHERE sg.facturation_type IN ('interne', 'ticket')

    UNION ALL

    SELECT
        p.date,
        p.id_organization,
        sg.facturation_type,
        CASE p.service
            WHEN 1 THEN 'petit_dejeuner'
            WHEN 2 THEN 'dejeuner'
            WHEN 4 THEN 'diner'
            ELSE 'autre'
        END AS service,
        CASE
            WHEN p.tranche = 1 THEN 'Tranche 1'
            WHEN p.tranche = 2 THEN 'Tranche 2'
            WHEN p.tranche = 3 THEN 'Tranche 3'
            WHEN p.tranche = 4 THEN 'Tranche 4'
            ELSE 'Hors tranche'
        END AS tranche,
        p.nb_passages
    FROM {p}passage_partner p
    INNER JOIN {p}subgroup_mapping sm ON p.subgroup      = sm.subgroup
    INNER JOIN {p}subgroup sg          ON sm.id_subgroup = sg.id_subgroup
    WHERE sg.facturation_type IN ('interne', 'ticket')
),
vague_ranked AS (
    SELECT
        cd.date,
        cd.id_organization,
        v.id_vague,
        v.name           AS nom_vague,
        v.id_school_year,
        ROW_NUMBER() OVER (
            PARTITION BY cd.date, cd.id_organization
            ORDER BY v.start_date DESC
        ) AS rn
    FROM (SELECT DISTINCT date, id_organization FROM combined_data) cd
    CROSS JOIN {p}vague v
    WHERE cd.date >= v.start_date
),
vague_data AS (
    SELECT date, id_organization, id_vague, nom_vague, id_school_year
    FROM vague_ranked
    WHERE rn = 1
)
SELECT
    cd.date,
    cd.id_organization,
    o.name                              AS nom_etablissement,
    o.access_software,
    oe.social_tarif_beneficiaries       AS effectif_cible,
    oe.intern_count                     AS effectif_interne,
    oe.total_enrollment                 AS effectif,
    o.rne,
    o.ips,
    o.vague                             AS vague_demarrage,
    vd.id_vague,
    vd.nom_vague,
    vd.id_school_year,
    sy.label                            AS school_year,
    cd.facturation_type,
    cd.tranche,
    cd.service,
    SUM(cd.nb_passages)                 AS nb_passages_total
FROM combined_data cd
LEFT JOIN {p}organization o             ON o.id_organization  = cd.id_organization
LEFT JOIN vague_data vd                 ON vd.date            = cd.date
                                       AND vd.id_organization = cd.id_organization
LEFT JOIN {p}school_year sy             ON sy.school_year_id  = vd.id_school_year
LEFT JOIN {p}organization_enrollment oe ON oe.organization_id = cd.id_organization
                                       AND oe.school_year_id  = vd.id_school_year
GROUP BY
    cd.date, cd.id_organization, o.name, o.access_software,
    oe.social_tarif_beneficiaries, oe.intern_count, oe.total_enrollment,
    o.rne, o.ips, o.vague,
    vd.id_vague, vd.nom_vague, vd.id_school_year, sy.label,
    cd.facturation_type, cd.tranche, cd.service
ORDER BY
    cd.date DESC,
    cd.id_organization,
    cd.facturation_type,
    cd.tranche
"""

    source = connect(dataset_cible)
    try:
        source.query(sql)
        duration = round(time.time() - t0, 2)
        logger.info(f"[{JOB_NAME}] OK — {duration}s")
    except Exception as e:
        logger.error(f"[{JOB_NAME}] Erreur fatale : {type(e).__name__}: {e}")
        raise

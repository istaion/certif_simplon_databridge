import logging
import time

from forepaas.dwh import connect
from forepaas.core.settings import PARAMS

logger = logging.getLogger(__name__)

JOB_NAME = "tarification_passages"


def customfunc(event):
    prefix_table = PARAMS["PREFIX_TABLE"]
    environnement_client = PARAMS["ENVIRONNEMENT_CLIENT"]
    dataset_cible = f"dwh/db_mg6jk45h_{environnement_client}/"
    p = prefix_table

    logger.info(f"Démarrage du job '{JOB_NAME}' — correction mapping service passage_partner (4=diner)")
    t0 = time.time()

    sql = f"""
CREATE OR REPLACE TABLE {p}tarification_passages AS
WITH combined_data AS (
    SELECT
        p.date,
        p.id_organization,
        p.id_user,
        COALESCE(sc.service_category, 'autre')                                               AS service,
        CASE WHEN COALESCE(p.id_tranche, uisy_pre.id_tranche) = 1
                  AND sg.facturation_type IN ('interne', 'ticket') THEN 1 ELSE 0 END         AS is_tranche1,
        CASE WHEN COALESCE(p.id_tranche, uisy_pre.id_tranche) = 2
                  AND sg.facturation_type IN ('interne', 'ticket') THEN 1 ELSE 0 END         AS is_tranche2,
        CASE WHEN COALESCE(p.id_tranche, uisy_pre.id_tranche) = 3
                  AND sg.facturation_type IN ('interne', 'ticket') THEN 1 ELSE 0 END         AS is_tranche3,
        CASE WHEN COALESCE(p.id_tranche, uisy_pre.id_tranche) = 4
                  AND sg.facturation_type IN ('interne', 'ticket') THEN 1 ELSE 0 END         AS is_tranche4,
        CASE WHEN sg.facturation_type NOT IN ('interne', 'ticket') OR sg.facturation_type IS NULL
                  OR COALESCE(p.id_tranche, uisy_pre.id_tranche) IS NULL THEN 1 ELSE 0 END  AS is_hors_tranche,
        CASE WHEN sg.facturation_type = 'interne'                          THEN 1 ELSE 0 END AS is_interne_group,
        CASE WHEN sg.facturation_type = 'ticket'                           THEN 1 ELSE 0 END AS is_ticket_group,
        CASE WHEN sg.facturation_type = 'autre' OR sg.facturation_type IS NULL THEN 1 ELSE 0 END AS is_autres_group,
        1                                                                                     AS nb_passages
    FROM {p}passage p
    LEFT JOIN {p}subgroup sg ON p.id_subgroup = sg.id_subgroup
    LEFT JOIN (
        SELECT id_service,
            CASE label
                WHEN 'Petit déjeuner' THEN 'petit_dejeuner'
                WHEN 'Déjeuner'       THEN 'dejeuner'
                WHEN 'Diner'          THEN 'diner'
                ELSE 'autre'
            END AS service_category
        FROM {p}service
    ) sc ON p.id_service = sc.id_service
    LEFT JOIN (
        SELECT id_user, id_tranche,
               ROW_NUMBER() OVER (PARTITION BY id_user ORDER BY school_year_id DESC) AS rn
        FROM {p}user_info_school_year
    ) uisy_pre ON uisy_pre.id_user = p.id_user AND uisy_pre.rn = 1

    UNION ALL

    SELECT
        p.date,
        p.id_organization,
        NULL                                                                                  AS id_user,
        CASE p.service
            WHEN 1 THEN 'petit_dejeuner'
            WHEN 2 THEN 'dejeuner'
            WHEN 4 THEN 'diner'
            ELSE 'autre'
        END                                                                                   AS service,
        CASE WHEN p.tranche = 1
                  AND sg.facturation_type IN ('interne', 'ticket') THEN p.nb_passages ELSE 0 END AS is_tranche1,
        CASE WHEN p.tranche = 2
                  AND sg.facturation_type IN ('interne', 'ticket') THEN p.nb_passages ELSE 0 END AS is_tranche2,
        CASE WHEN p.tranche = 3
                  AND sg.facturation_type IN ('interne', 'ticket') THEN p.nb_passages ELSE 0 END AS is_tranche3,
        CASE WHEN p.tranche = 4
                  AND sg.facturation_type IN ('interne', 'ticket') THEN p.nb_passages ELSE 0 END AS is_tranche4,
        CASE WHEN sg.facturation_type NOT IN ('interne', 'ticket') OR sg.facturation_type IS NULL
                  OR p.tranche = -1 THEN p.nb_passages ELSE 0 END                            AS is_hors_tranche,
        CASE WHEN sg.facturation_type = 'interne'                          THEN p.nb_passages ELSE 0 END AS is_interne_group,
        CASE WHEN sg.facturation_type = 'ticket'                           THEN p.nb_passages ELSE 0 END AS is_ticket_group,
        CASE WHEN sg.facturation_type = 'autre' OR sg.facturation_type IS NULL THEN p.nb_passages ELSE 0 END AS is_autres_group,
        p.nb_passages
    FROM {p}passage_partner p
    LEFT JOIN {p}subgroup_mapping sm ON p.subgroup      = sm.subgroup
    LEFT JOIN {p}subgroup sg          ON sm.id_subgroup = sg.id_subgroup
),
vague_ranked AS (
    SELECT
        cd.date,
        cd.id_organization,
        v.id_vague,
        v.name          AS nom_vague,
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
    o.access_software                   AS editeur_acces,
    o.rne                               AS uai,
    o.ips,
    o.vague                             AS vague_demarrage,
    vd.id_vague,
    vd.nom_vague,
    vd.id_school_year,
    sy.label                            AS school_year,
    oe.social_tarif_beneficiaries       AS effectif_cible,
    oe.intern_count                     AS effectif_interne,
    oe.total_enrollment                 AS effectif_total,
    cd.service,
    SUM(cd.nb_passages)                 AS nb_passages_service,
    SUM(cd.is_tranche1)                 AS nb_tranche1,
    SUM(cd.is_tranche2)                 AS nb_tranche2,
    SUM(cd.is_tranche3)                 AS nb_tranche3,
    SUM(cd.is_tranche4)                 AS nb_tranche4,
    SUM(cd.is_hors_tranche)             AS nb_hors_tranche,
    SUM(cd.is_interne_group)            AS nb_interne_group,
    SUM(cd.is_ticket_group)             AS nb_ticket_group,
    SUM(cd.is_autres_group)             AS nb_autres_group,
    SUM(cd.nb_passages)                 AS nb_passages_total
FROM combined_data cd
LEFT JOIN {p}organization o             ON o.id_organization  = cd.id_organization
LEFT JOIN vague_data vd                 ON vd.date            = cd.date
                                       AND vd.id_organization = cd.id_organization
LEFT JOIN {p}school_year sy             ON sy.school_year_id  = vd.id_school_year
LEFT JOIN {p}organization_enrollment oe ON oe.organization_id = cd.id_organization
                                       AND oe.school_year_id  = vd.id_school_year
GROUP BY
    cd.date, cd.id_organization,
    o.name, o.access_software, o.rne, o.ips, o.vague,
    vd.id_vague, vd.nom_vague, vd.id_school_year, sy.label,
    oe.social_tarif_beneficiaries, oe.intern_count, oe.total_enrollment,
    cd.service
ORDER BY
    cd.date DESC,
    cd.id_organization,
    cd.service
"""

    source = connect(dataset_cible)
    try:
        source.query(sql)
        duration = round(time.time() - t0, 2)
        logger.info(f"[{JOB_NAME}] OK — {duration}s")
    except Exception as e:
        logger.error(f"[{JOB_NAME}] Erreur fatale : {type(e).__name__}: {e}")
        raise

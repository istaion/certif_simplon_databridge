"""
Requêtes SQL de reporting — tables agrégées calculées depuis les tables de base.

Les fonctions `build_*_sql` retournent une chaîne SQL prête à exécuter via
TrinoClient.run_query(). Elles gèrent les variantes par environnement_client.

Paramètre `recent_only` (bool, défaut False) : si True, les tables dont les
données sont horodatées ne recalculent que la période NOW() - 1 an.
"""


def _passage_date_filter(recent_only: bool) -> str:
    return "\n          AND p.date >= CURRENT_DATE - INTERVAL '1' YEAR" if recent_only else ""


def _vague_date_filter(recent_only: bool) -> str:
    return "\n          AND (v.start_date IS NULL OR v.start_date >= CURRENT_DATE - INTERVAL '1' YEAR)" if recent_only else ""


def build_tarification_2_sql(prefix_table: str, environnement_client: str, recent_only: bool = False) -> str:
    """
    Retourne le SQL CREATE OR REPLACE TABLE pour tarification_2,
    adapté selon l'environnement client.

    - "centre" in environnement_client : facturation_type, tranches 1-4, mode_transmission
    - "93"     in environnement_client : label_group, tranches 1-14, mode_transmission
    """
    if "centre" in environnement_client:
        return _tarification_2_centre(prefix_table, recent_only)
    if "93" in environnement_client:
        return _tarification_2_93(prefix_table, recent_only)
    raise ValueError(
        f"tarification_2 non supportée pour environnement_client={environnement_client!r} "
        f"(attendu : contient 'centre' ou '93')"
    )


# ── tarification_2 variante centre ───────────────────────────────────────────

def _tarification_2_centre(p: str, recent_only: bool = False) -> str:
    extra = _vague_date_filter(recent_only)
    return f"""
CREATE OR REPLACE TABLE {p}tarification_2 AS
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
        END AS tranche,
        CASE uisy.choice_bank_details
            WHEN 'Transmission automatique'    THEN 'transmission auto'
            WHEN 'Transmission manuelle impot' THEN 'transmission manuelle'
            WHEN 'Pas de données fournies'     THEN 'pas de données transmises'
            WHEN 'Refus de transmission'       THEN 'refus de transmission'
            WHEN 'Identité pivot'              THEN 'identité pivot'
            WHEN 'Transmission manuelle CAF'   THEN 'avis impot'
            WHEN 'Import départemental'        THEN 'import departemental'
            WHEN 'Inscription papier'          THEN 'import gestionnaire'
            ELSE 'Donnée absente de nos bases'
        END AS mode_transmission,
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
    WHERE (sg.id_group IS NULL OR sg.id_group != 2){extra}
)
SELECT
    id_organization,
    nom_etablissement,
    total_enrollment,
    intern_count,
    social_tarif_beneficiaries,
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
    mode_transmission,
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
    total_enrollment,
    intern_count,
    social_tarif_beneficiaries,
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
    mode_transmission
ORDER BY
    nom_vague,
    nom_etablissement,
    nom_sous_groupe
"""


# ── tarification_2 variante 93 ────────────────────────────────────────────────

def _tarification_2_93(p: str, recent_only: bool = False) -> str:
    extra = _vague_date_filter(recent_only)
    where = f"    WHERE TRUE{extra}" if recent_only else ""
    return f"""
CREATE OR REPLACE TABLE {p}tarification_2 AS
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
        o.access_software,
        r.subgroup_id                   AS id_subgroup,
        uisy.label_subgroup             AS nom_sous_groupe,
        uisy.label_group,
        COALESCE(t.label, uisy.label_tranche) AS tranche,
        COALESCE(
            r.choice_bank_detail,
            CASE uisy.choice_bank_details
                WHEN 'Transmission automatique'    THEN 'transmission auto'
                WHEN 'Transmission manuelle impot' THEN 'transmission manuelle'
                WHEN 'Pas de données fournies'     THEN 'pas de données transmises'
                WHEN 'Refus de transmission'       THEN 'refus de transmission'
                WHEN 'Identité pivot'              THEN 'identité pivot'
                WHEN 'Transmission manuelle CAF'   THEN 'avis impot'
                WHEN 'Import départemental'        THEN 'import departemental'
                WHEN 'Inscription papier'          THEN 'import gestionnaire'
                ELSE 'Donnée absente de nos bases'
            END
        ) AS mode_transmission,
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
{where})
SELECT
    id_organization,
    nom_etablissement,
    total_enrollment,
    intern_count,
    social_tarif_beneficiaries,
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
    mode_transmission,
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
    total_enrollment,
    intern_count,
    social_tarif_beneficiaries,
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
    mode_transmission
ORDER BY
    nom_vague,
    nom_etablissement,
    nom_sous_groupe
"""


# ── tarification_1 ────────────────────────────────────────────────────────────

def build_tarification_1_sql(prefix_table: str, environnement_client: str, recent_only: bool = False) -> str:
    """
    Retourne le SQL CREATE OR REPLACE TABLE pour tarification_1,
    adapté selon l'environnement client.

    - "centre" in environnement_client : facturation_type (groupes 1,2,3,...), tranches 1-4
    - "93"     in environnement_client : label_group (Eleves/Commensaux),     tranches 1-14
    """
    if "centre" in environnement_client:
        return _tarification_1_centre(prefix_table, recent_only)
    if "93" in environnement_client:
        return _tarification_1_93(prefix_table, recent_only)
    raise ValueError(
        f"tarification_1 non supportée pour environnement_client={environnement_client!r} "
        f"(attendu : contient 'centre' ou '93')"
    )


# ── Variante centre ───────────────────────────────────────────────────────────

def _tarification_1_centre(p: str, recent_only: bool = False) -> str:
    extra = _vague_date_filter(recent_only)
    return f"""
CREATE OR REPLACE TABLE {p}tarification_1 AS
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
    WHERE (sg.id_group IS NULL OR sg.id_group != 2){extra}
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


# ── Variante 93 ───────────────────────────────────────────────────────────────

def _tarification_1_93(p: str, recent_only: bool = False) -> str:
    extra = _vague_date_filter(recent_only)
    where = f"    WHERE TRUE{extra}" if recent_only else ""
    return f"""
CREATE OR REPLACE TABLE {p}tarification_1 AS
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
{where})
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


# ── suivi_validations (cumul journalier des validations par vague) ────────────

def build_suivi_validations_sql(prefix_table: str, environnement_client: str, recent_only: bool = False) -> str:
    """
    Retourne le SQL CREATE OR REPLACE TABLE pour suivi_validations.
    Matérialise les cumuls journaliers de validations par (jour, vague, établissement),
    fenêtre cumulée au niveau vague (PARTITION BY id_vague).

    Uniquement disponible pour l'environnement "centre".
    """
    if "centre" in environnement_client:
        return _suivi_validations_centre(prefix_table, recent_only)
    raise ValueError(
        f"suivi_validations non supportée pour environnement_client={environnement_client!r} "
        f"(seul 'centre' est implémenté)"
    )


def _suivi_validations_centre(p: str, recent_only: bool = False) -> str:
    extra = (
        "\n          AND DATE(r.updated_at) >= CURRENT_DATE - INTERVAL '1' YEAR"
        if recent_only else ""
    )
    return f"""
CREATE OR REPLACE TABLE {p}suivi_validations AS
WITH daily_data AS (
    SELECT
        event_date                          AS jour,
        id_vague,
        nom_vague,
        id_school_year,
        school_year,
        access_software,
        nom_etablissement,
        type,
        department,
        ips,
        SUM(is_validation)                  AS nb_validations_jour,
        SUM(is_validation_tranche1)         AS nb_validations_tranche1_jour,
        SUM(is_validation_tranche2)         AS nb_validations_tranche2_jour,
        SUM(is_validation_tranche3)         AS nb_validations_tranche3_jour,
        SUM(is_validation_tranche4)         AS nb_validations_tranche4_jour
    FROM (
        SELECT
            DATE(r.updated_at)              AS event_date,
            v.id_vague,
            v.name                          AS nom_vague,
            v.id_school_year,
            sy.label                        AS school_year,
            o.access_software,
            o.name                          AS nom_etablissement,
            o.type,
            o.department,
            o.ips,
            1                               AS is_validation,
            CASE WHEN r.tranche_id = 1 THEN 1 ELSE 0 END AS is_validation_tranche1,
            CASE WHEN r.tranche_id = 2 THEN 1 ELSE 0 END AS is_validation_tranche2,
            CASE WHEN r.tranche_id = 3 THEN 1 ELSE 0 END AS is_validation_tranche3,
            CASE WHEN r.tranche_id = 4 THEN 1 ELSE 0 END AS is_validation_tranche4
        FROM {p}registration r
        JOIN {p}session s         ON r.id_session       = s.id
        JOIN {p}vague v           ON s.id_vague          = v.id_vague
        LEFT JOIN {p}school_year sy ON sy.school_year_id = v.id_school_year
        JOIN {p}organization o    ON s.id_organization   = o.id_organization
        LEFT JOIN {p}subgroup sg  ON r.subgroup_id       = sg.id_subgroup
        WHERE r.status IN ('validated', 'merged')
          AND DATE(r.updated_at) >= v.start_date
          AND (sg.id_group IS NULL OR sg.id_group != 2){extra}
    ) events
    GROUP BY
        event_date, id_vague, nom_vague, id_school_year, school_year,
        access_software, nom_etablissement, type, department, ips
)
SELECT
    jour,
    id_vague,
    nom_vague,
    id_school_year,
    school_year,
    access_software,
    nom_etablissement,
    type,
    department,
    ips,
    SUM(nb_validations_jour) OVER w          AS dossiers_valides_cumul,
    SUM(nb_validations_tranche1_jour) OVER w AS dossiers_valides_tranche1_cumul,
    SUM(nb_validations_tranche2_jour) OVER w AS dossiers_valides_tranche2_cumul,
    SUM(nb_validations_tranche3_jour) OVER w AS dossiers_valides_tranche3_cumul,
    SUM(nb_validations_tranche4_jour) OVER w AS dossiers_valides_tranche4_cumul
FROM daily_data
WINDOW w AS (
    PARTITION BY id_vague ORDER BY jour ROWS UNBOUNDED PRECEDING
)
ORDER BY jour
"""


# ── tarification_3 (passages déjeuner) ───────────────────────────────────────

def build_tarification_3_sql(prefix_table: str, environnement_client: str, recent_only: bool = False) -> str:
    """
    Retourne le SQL CREATE OR REPLACE TABLE pour tarification_3.
    Agrège les passages déjeuner (directs + partenaires) par (date, organisation,
    facturation_type, tranche), adapté aux schémas post-refonte.

    Uniquement disponible pour l'environnement "centre".
    """
    if "centre" in environnement_client:
        return _tarification_3_centre(prefix_table, recent_only)
    raise ValueError(
        f"tarification_3 non supportée pour environnement_client={environnement_client!r} "
        f"(seul 'centre' est implémenté)"
    )


def build_suivi_inscriptions_sql(prefix_table: str, environnement_client: str, recent_only: bool = False) -> str:
    """
    Retourne le SQL CREATE OR REPLACE TABLE pour suivi_inscriptions.
    Matérialise les cumuls journaliers d'inscriptions, dépôts et validations
    par (jour, vague, établissement), avec remplissage des jours sans événement.

    - "centre" : version complète avec tranches, effectifs, id_organization
    - "93"     : version simplifiée avec school_year et id_school_year
    """
    if "centre" in environnement_client:
        return _suivi_inscriptions_centre(prefix_table, recent_only)
    if "93" in environnement_client:
        return _suivi_inscriptions_93(prefix_table, recent_only)
    raise ValueError(
        f"suivi_inscriptions non supportée pour environnement_client={environnement_client!r} "
        f"(seul 'centre' est implémenté)"
    )


def _suivi_inscriptions_centre(p: str, recent_only: bool = False) -> str:
    extra_created = (
        "\n          AND DATE(r.created_at) >= CURRENT_DATE - INTERVAL '1' YEAR"
        if recent_only else ""
    )
    extra_h = (
        "\n          AND DATE(h.created_at) >= CURRENT_DATE - INTERVAL '1' YEAR"
        if recent_only else ""
    )
    return f"""
CREATE OR REPLACE TABLE {p}suivi_inscriptions AS
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
        SUM(is_validation_hors_tranche)     AS nb_validations_hors_tranche_jour,
        total_enrollment,
        social_tarif_beneficiaries
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
            0 AS is_validation_hors_tranche,
            oe.total_enrollment,
            oe.social_tarif_beneficiaries
        FROM {p}registration r
        JOIN {p}session s         ON r.id_session       = s.id
        JOIN {p}vague v           ON s.id_vague          = v.id_vague
        LEFT JOIN {p}school_year sy ON sy.school_year_id = v.id_school_year
        JOIN {p}organization o    ON s.id_organization   = o.id_organization
        LEFT JOIN {p}subgroup sg  ON r.subgroup_id       = sg.id_subgroup
        LEFT JOIN {p}organization_enrollment oe
                                  ON oe.organization_id  = o.id_organization
                                 AND oe.school_year_id   = v.id_school_year
        WHERE r.status != 'canceled'
          AND DATE(r.created_at) >= v.start_date
          AND (sg.id_group IS NULL OR sg.id_group != 2){extra_created}

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
            0 AS is_validation_hors_tranche,
            oe.total_enrollment,
            oe.social_tarif_beneficiaries
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
        LEFT JOIN {p}organization_enrollment oe
                                  ON oe.organization_id  = o.id_organization
                                 AND oe.school_year_id   = v.id_school_year
        WHERE r.status != 'canceled'
          AND DATE(h.created_at) >= v.start_date
          AND (sg.id_group IS NULL OR sg.id_group != 2){extra_h}

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
            CASE WHEN r.tranche_id IS NULL OR r.tranche_id NOT IN (1,2,3,4) THEN 1 ELSE 0 END AS is_validation_hors_tranche,
            oe.total_enrollment,
            oe.social_tarif_beneficiaries
        FROM {p}history h
        JOIN {p}registration r    ON h.registration_id  = r.id
        JOIN {p}session s         ON r.id_session       = s.id
        JOIN {p}vague v           ON s.id_vague          = v.id_vague
        LEFT JOIN {p}school_year sy ON sy.school_year_id = v.id_school_year
        JOIN {p}organization o    ON s.id_organization   = o.id_organization
        LEFT JOIN {p}subgroup sg  ON r.subgroup_id       = sg.id_subgroup
        LEFT JOIN {p}organization_enrollment oe
                                  ON oe.organization_id  = o.id_organization
                                 AND oe.school_year_id   = v.id_school_year
        WHERE h.event = 'REGISTRATION_APPROVED'
          AND r.status IN ('merged', 'validated')
          AND DATE(h.created_at) >= v.start_date
          AND (sg.id_group IS NULL OR sg.id_group != 2){extra_h}
    ) events
    GROUP BY
        event_date, id_vague, nom_vague, id_school_year, school_year, id_organization,
        access_software, nom_etablissement, type, department, ips,
        total_enrollment, social_tarif_beneficiaries
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
        ips,
        total_enrollment,
        social_tarif_beneficiaries
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
        e.ips,
        e.total_enrollment,
        e.social_tarif_beneficiaries
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
        COALESCE(dd.nb_validations_hors_tranche_jour, 0) AS nb_validations_hors_tranche_jour,
        c.total_enrollment,
        c.social_tarif_beneficiaries
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
    SUM(nb_validations_hors_tranche_jour) OVER w AS dossiers_valides_cumul_hors_tranche,
    total_enrollment,
    social_tarif_beneficiaries                   AS social_tariff_beneficiaries
FROM filled_data
WINDOW w AS (
    PARTITION BY id_vague, id_organization ORDER BY jour ROWS UNBOUNDED PRECEDING
)
ORDER BY jour, nom_etablissement
"""


def _suivi_inscriptions_93(p: str, recent_only: bool = False) -> str:
    extra_created = (
        "\n          AND DATE(r.created_at) >= CURRENT_DATE - INTERVAL '1' YEAR"
        if recent_only else ""
    )
    extra_h = (
        "\n          AND DATE(h.created_at) >= CURRENT_DATE - INTERVAL '1' YEAR"
        if recent_only else ""
    )
    return f"""
CREATE OR REPLACE TABLE {p}suivi_inscriptions AS
WITH daily_data AS (
    SELECT
        event_date                          AS jour,
        id_vague,
        nom_vague,
        id_school_year,
        school_year,
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
          AND (sg.id_group IS NULL OR sg.id_group != 2){extra_created}

        UNION ALL

        -- Dépôts
        SELECT
            DATE(h.created_at)              AS event_date,
            v.id_vague,
            v.name                          AS nom_vague,
            v.id_school_year,
            sy.label                        AS school_year,
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
          AND (sg.id_group IS NULL OR sg.id_group != 2){extra_h}

        UNION ALL

        -- Validations
        SELECT
            DATE(h.created_at)              AS event_date,
            v.id_vague,
            v.name                          AS nom_vague,
            v.id_school_year,
            sy.label                        AS school_year,
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
          AND (sg.id_group IS NULL OR sg.id_group != 2){extra_h}
    ) events
    GROUP BY
        event_date, id_vague, nom_vague, id_school_year, school_year, nom_etablissement
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
        c.nom_etablissement,
        COALESCE(dd.nb_inscriptions_jour, 0) AS nb_inscriptions_jour,
        COALESCE(dd.nb_depots_jour, 0)       AS nb_depots_jour,
        COALESCE(dd.nb_validations_jour, 0)  AS nb_validations_jour
    FROM all_combinations c
    LEFT JOIN daily_data dd
        ON  c.jour              = dd.jour
        AND c.id_vague          = dd.id_vague
        AND c.nom_etablissement = dd.nom_etablissement
)
SELECT
    jour,
    id_vague,
    nom_vague,
    id_school_year,
    school_year,
    nom_etablissement,
    SUM(nb_inscriptions_jour) OVER w AS total_connexions_cumul,
    SUM(nb_depots_jour)       OVER w AS dossiers_deposes_cumul,
    SUM(nb_validations_jour)  OVER w AS dossiers_valides_cumul
FROM filled_data
WINDOW w AS (
    PARTITION BY id_vague, nom_etablissement ORDER BY jour ROWS UNBOUNDED PRECEDING
)
ORDER BY jour, nom_etablissement
"""


def _tarification_3_centre(p: str, recent_only: bool = False) -> str:
    extra = _passage_date_filter(recent_only)
    return f"""
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
    WHERE sg.facturation_type IN ('interne', 'ticket'){extra}

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
    WHERE sg.facturation_type IN ('interne', 'ticket'){extra}
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


# ── tarification_passages (93) ───────────────────────────────────────────────

def build_tarification_passages_sql(prefix_table: str, environnement_client: str, recent_only: bool = False) -> str:
    """
    Retourne le SQL CREATE OR REPLACE TABLE pour tarification_passages.
    Agrège les passages (directs + partenaires) par (date, organisation, groupe, service, tranche).
    L'année scolaire et les effectifs sont déterminés dynamiquement via la vague courante.
    """
    if "centre" in environnement_client:
        return _tarification_passages_centre(prefix_table, recent_only)
    if "93" in environnement_client:
        return _tarification_passages_93(prefix_table, recent_only)
    raise ValueError(
        f"tarification_passages non supportée pour environnement_client={environnement_client!r} "
        f"(attendu : contient 'centre' ou '93')"
    )


def _passages_detail_centre(p: str, recent_only: bool = False) -> str:
    where = f"\n    WHERE p.date >= CURRENT_DATE - INTERVAL '1' YEAR" if recent_only else ""
    return f"""
CREATE OR REPLACE TABLE {p}passages_detail AS
WITH combined_data AS (
    SELECT
        p.date,
        p.id_organization,
        p.id_user,
        COALESCE(sg.facturation_type, 'autre')                                              AS groupe,
        COALESCE(sc.service_category, 'autre')                                              AS service,
        CASE WHEN COALESCE(p.id_tranche, uisy_pre.id_tranche) = 1
                  AND sg.facturation_type IN ('interne', 'ticket') THEN 1 ELSE 0 END        AS is_tranche1,
        CASE WHEN COALESCE(p.id_tranche, uisy_pre.id_tranche) = 2
                  AND sg.facturation_type IN ('interne', 'ticket') THEN 1 ELSE 0 END        AS is_tranche2,
        CASE WHEN COALESCE(p.id_tranche, uisy_pre.id_tranche) = 3
                  AND sg.facturation_type IN ('interne', 'ticket') THEN 1 ELSE 0 END        AS is_tranche3,
        CASE WHEN COALESCE(p.id_tranche, uisy_pre.id_tranche) = 4
                  AND sg.facturation_type IN ('interne', 'ticket') THEN 1 ELSE 0 END        AS is_tranche4,
        CASE WHEN sg.facturation_type NOT IN ('interne', 'ticket') OR sg.facturation_type IS NULL
                  OR COALESCE(p.id_tranche, uisy_pre.id_tranche) IS NULL THEN 1 ELSE 0 END AS is_hors_tranche,
        1                                                                                    AS nb_passages
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
    ) uisy_pre ON uisy_pre.id_user = p.id_user AND uisy_pre.rn = 1{where}

    UNION ALL

    SELECT
        p.date,
        p.id_organization,
        NULL                                                                                 AS id_user,
        COALESCE(sg.facturation_type, 'autre')                                              AS groupe,
        CASE p.service
            WHEN 1 THEN 'petit_dejeuner'
            WHEN 2 THEN 'dejeuner'
            WHEN 4 THEN 'diner'
            ELSE 'autre'
        END                                                                                  AS service,
        CASE WHEN p.tranche = 1
                  AND sg.facturation_type IN ('interne', 'ticket') THEN p.nb_passages ELSE 0 END AS is_tranche1,
        CASE WHEN p.tranche = 2
                  AND sg.facturation_type IN ('interne', 'ticket') THEN p.nb_passages ELSE 0 END AS is_tranche2,
        CASE WHEN p.tranche = 3
                  AND sg.facturation_type IN ('interne', 'ticket') THEN p.nb_passages ELSE 0 END AS is_tranche3,
        CASE WHEN p.tranche = 4
                  AND sg.facturation_type IN ('interne', 'ticket') THEN p.nb_passages ELSE 0 END AS is_tranche4,
        CASE WHEN sg.facturation_type NOT IN ('interne', 'ticket') OR sg.facturation_type IS NULL
                  OR p.tranche = -1 THEN p.nb_passages ELSE 0 END                           AS is_hors_tranche,
        p.nb_passages
    FROM {p}passage_partner p
    LEFT JOIN {p}subgroup_mapping sm ON p.subgroup      = sm.subgroup
    LEFT JOIN {p}subgroup sg          ON sm.id_subgroup = sg.id_subgroup{where}
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
    cd.groupe,
    cd.service,
    SUM(cd.is_tranche1)                 AS nb_tranche1,
    SUM(cd.is_tranche2)                 AS nb_tranche2,
    SUM(cd.is_tranche3)                 AS nb_tranche3,
    SUM(cd.is_tranche4)                 AS nb_tranche4,
    SUM(cd.is_hors_tranche)             AS nb_hors_tranche,
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
    cd.groupe, cd.service
ORDER BY
    cd.date DESC,
    cd.id_organization,
    cd.groupe,
    cd.service
"""


def _tarification_passages_93(p: str, recent_only: bool = False) -> str:
    where = f"\n    WHERE p.date >= CURRENT_DATE - INTERVAL '1' YEAR" if recent_only else ""
    return f"""
CREATE OR REPLACE TABLE {p}tarification_passages AS
WITH combined_data AS (
    SELECT
        p.date,
        p.id_organization,
        COALESCE(sc.service_category, 'autre')                                       AS service,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 1  AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_tranche1,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 2  AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_tranche2,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 3  AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_tranche3,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 4  AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_tranche4,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 5  AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_tranche5,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 6  AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_tranche6,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 7  AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_tranche7,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 8  AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_tranche8,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 9  AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_tranche9,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 10 AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_tranche10,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 11 AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_tranche11,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 12 AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_tranche12,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 13 AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_tranche13,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 14 AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_tranche14,
        CASE WHEN sg.id_group != 1 OR sg.id_group IS NULL
                  OR p.id_tranche IS NULL
                  OR TRY_CAST(p.id_tranche AS INTEGER) IS NULL
                  OR TRY_CAST(p.id_tranche AS INTEGER) NOT BETWEEN 1 AND 14
             THEN 1 ELSE 0 END                                                       AS is_hors_tranche,
        CASE WHEN sg.id_group = 1    THEN 1 ELSE 0 END                              AS is_eleve_group,
        CASE WHEN sg.id_group = 2    THEN 1 ELSE 0 END                              AS is_commensaux_group,
        CASE WHEN sg.id_group IS NULL THEN 1 ELSE 0 END                             AS is_autres_group,
        1                                                                            AS nb_passages
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
    ) sc ON p.id_service = sc.id_service{where}

    UNION ALL

    SELECT
        p.date,
        p.id_organization,
        CASE p.service
            WHEN 1 THEN 'petit_dejeuner'
            WHEN 2 THEN 'dejeuner'
            WHEN 4 THEN 'diner'
            ELSE 'autre'
        END                                                                          AS service,
        CASE WHEN p.tranche = 1  AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END AS is_tranche1,
        CASE WHEN p.tranche = 2  AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END AS is_tranche2,
        CASE WHEN p.tranche = 3  AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END AS is_tranche3,
        CASE WHEN p.tranche = 4  AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END AS is_tranche4,
        CASE WHEN p.tranche = 5  AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END AS is_tranche5,
        CASE WHEN p.tranche = 6  AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END AS is_tranche6,
        CASE WHEN p.tranche = 7  AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END AS is_tranche7,
        CASE WHEN p.tranche = 8  AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END AS is_tranche8,
        CASE WHEN p.tranche = 9  AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END AS is_tranche9,
        CASE WHEN p.tranche = 10 AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END AS is_tranche10,
        CASE WHEN p.tranche = 11 AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END AS is_tranche11,
        CASE WHEN p.tranche = 12 AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END AS is_tranche12,
        CASE WHEN p.tranche = 13 AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END AS is_tranche13,
        CASE WHEN p.tranche = 14 AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END AS is_tranche14,
        CASE WHEN sg.id_group != 1 OR sg.id_group IS NULL
                  OR p.tranche = -1 OR p.tranche NOT BETWEEN 1 AND 14
             THEN p.nb_passages ELSE 0 END                                           AS is_hors_tranche,
        CASE WHEN sg.id_group = 1    THEN p.nb_passages ELSE 0 END                 AS is_eleve_group,
        CASE WHEN sg.id_group = 2    THEN p.nb_passages ELSE 0 END                 AS is_commensaux_group,
        CASE WHEN sg.id_group IS NULL THEN p.nb_passages ELSE 0 END                AS is_autres_group,
        p.nb_passages
    FROM {p}passage_partner p
    LEFT JOIN {p}subgroup_mapping sm ON p.subgroup      = sm.subgroup
    LEFT JOIN {p}subgroup sg          ON sm.id_subgroup = sg.id_subgroup{where}
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
    SUM(cd.is_tranche5)                 AS nb_tranche5,
    SUM(cd.is_tranche6)                 AS nb_tranche6,
    SUM(cd.is_tranche7)                 AS nb_tranche7,
    SUM(cd.is_tranche8)                 AS nb_tranche8,
    SUM(cd.is_tranche9)                 AS nb_tranche9,
    SUM(cd.is_tranche10)                AS nb_tranche10,
    SUM(cd.is_tranche11)                AS nb_tranche11,
    SUM(cd.is_tranche12)                AS nb_tranche12,
    SUM(cd.is_tranche13)                AS nb_tranche13,
    SUM(cd.is_tranche14)                AS nb_tranche14,
    SUM(cd.is_hors_tranche)             AS nb_hors_tranche,
    SUM(cd.is_eleve_group)              AS nb_eleve_group,
    SUM(cd.is_commensaux_group)         AS nb_commensaux_group,
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


# ── passages_detail (93) ─────────────────────────────────────────────────────

def build_passages_detail_sql(prefix_table: str, environnement_client: str, recent_only: bool = False) -> str:
    """
    Retourne le SQL CREATE OR REPLACE TABLE pour passages_detail.
    L'année scolaire et les effectifs sont déterminés dynamiquement via la vague courante.

    - "93"     : agrège par (date, organisation, groupe, label subgroup, service, tranche A-Mbis)
    - "centre" : agrège par (date, organisation, service) avec tranches 1-4 et groupes interne/ticket/autre
    """
    if "centre" in environnement_client:
        return _passages_detail_centre(prefix_table, recent_only)
    if "93" in environnement_client:
        return _passages_detail_93(prefix_table, recent_only)
    raise ValueError(
        f"passages_detail non supportée pour environnement_client={environnement_client!r} "
        f"(attendu : contient 'centre' ou '93')"
    )


def _tarification_passages_centre(p: str, recent_only: bool = False) -> str:
    where = f"\n    WHERE p.date >= CURRENT_DATE - INTERVAL '1' YEAR" if recent_only else ""
    return f"""
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
    ) uisy_pre ON uisy_pre.id_user = p.id_user AND uisy_pre.rn = 1{where}

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
    LEFT JOIN {p}subgroup sg          ON sm.id_subgroup = sg.id_subgroup{where}
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


def _passages_detail_93(p: str, recent_only: bool = False) -> str:
    where = f"\n    WHERE p.date >= CURRENT_DATE - INTERVAL '1' YEAR" if recent_only else ""
    return f"""
CREATE OR REPLACE TABLE {p}passages_detail AS
WITH combined_data AS (
    SELECT
        p.date,
        p.id_organization,
        sg.label,
        CASE
            WHEN sg.id_group = 1 THEN 'Elève'
            WHEN sg.id_group = 2 THEN 'Commensaux'
            ELSE 'autre'
        END                                                                              AS groupe,
        COALESCE(sc.service_category, 'autre')                                           AS service,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 1  AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_trancheA,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 2  AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_trancheB,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 3  AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_trancheC,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 4  AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_trancheD,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 5  AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_trancheE,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 6  AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_trancheF,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 7  AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_trancheG,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 8  AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_trancheH,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 9  AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_trancheI,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 10 AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_trancheJ,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 11 AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_trancheK,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 12 AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_trancheL,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 13 AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_trancheM,
        CASE WHEN TRY_CAST(p.id_tranche AS INTEGER) = 14 AND sg.id_group = 1 THEN 1 ELSE 0 END AS is_trancheMbis,
        CASE WHEN sg.id_group != 1 OR sg.id_group IS NULL
                  OR p.id_tranche IS NULL
                  OR TRY_CAST(p.id_tranche AS INTEGER) IS NULL
                  OR TRY_CAST(p.id_tranche AS INTEGER) NOT BETWEEN 1 AND 14
             THEN 1 ELSE 0 END                                                           AS is_hors_tranche,
        1                                                                                AS nb_passages
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
    ) sc ON p.id_service = sc.id_service{where}

    UNION ALL

    SELECT
        p.date,
        p.id_organization,
        sg.label,
        CASE
            WHEN sg.id_group = 1 THEN 'Elève'
            WHEN sg.id_group = 2 THEN 'Commensaux'
            ELSE 'autre'
        END                                                                              AS groupe,
        CASE p.service
            WHEN 1 THEN 'petit_dejeuner'
            WHEN 2 THEN 'dejeuner'
            WHEN 4 THEN 'diner'
            ELSE 'autre'
        END                                                                              AS service,
        CASE WHEN p.tranche = 1  AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END     AS is_trancheA,
        CASE WHEN p.tranche = 2  AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END     AS is_trancheB,
        CASE WHEN p.tranche = 3  AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END     AS is_trancheC,
        CASE WHEN p.tranche = 4  AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END     AS is_trancheD,
        CASE WHEN p.tranche = 5  AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END     AS is_trancheE,
        CASE WHEN p.tranche = 6  AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END     AS is_trancheF,
        CASE WHEN p.tranche = 7  AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END     AS is_trancheG,
        CASE WHEN p.tranche = 8  AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END     AS is_trancheH,
        CASE WHEN p.tranche = 9  AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END     AS is_trancheI,
        CASE WHEN p.tranche = 10 AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END     AS is_trancheJ,
        CASE WHEN p.tranche = 11 AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END     AS is_trancheK,
        CASE WHEN p.tranche = 12 AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END     AS is_trancheL,
        CASE WHEN p.tranche = 13 AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END     AS is_trancheM,
        CASE WHEN p.tranche = 14 AND sg.id_group = 1 THEN p.nb_passages ELSE 0 END     AS is_trancheMbis,
        CASE WHEN sg.id_group != 1 OR sg.id_group IS NULL
                  OR p.tranche = -1 OR p.tranche NOT BETWEEN 1 AND 14
             THEN p.nb_passages ELSE 0 END                                               AS is_hors_tranche,
        p.nb_passages
    FROM {p}passage_partner p
    LEFT JOIN {p}subgroup_mapping sm ON p.subgroup      = sm.subgroup
    LEFT JOIN {p}subgroup sg          ON sm.id_subgroup = sg.id_subgroup{where}
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
    cd.groupe,
    cd.label,
    cd.service,
    SUM(cd.is_trancheA)                 AS nb_trancheA,
    SUM(cd.is_trancheB)                 AS nb_trancheB,
    SUM(cd.is_trancheC)                 AS nb_trancheC,
    SUM(cd.is_trancheD)                 AS nb_trancheD,
    SUM(cd.is_trancheE)                 AS nb_trancheE,
    SUM(cd.is_trancheF)                 AS nb_trancheF,
    SUM(cd.is_trancheG)                 AS nb_trancheG,
    SUM(cd.is_trancheH)                 AS nb_trancheH,
    SUM(cd.is_trancheI)                 AS nb_trancheI,
    SUM(cd.is_trancheJ)                 AS nb_trancheJ,
    SUM(cd.is_trancheK)                 AS nb_trancheK,
    SUM(cd.is_trancheL)                 AS nb_trancheL,
    SUM(cd.is_trancheM)                 AS nb_trancheM,
    SUM(cd.is_trancheMbis)              AS nb_trancheMbis,
    SUM(cd.is_hors_tranche)             AS nb_hors_tranche,
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
    cd.groupe, cd.label, cd.service
ORDER BY
    cd.date DESC,
    cd.id_organization,
    cd.groupe,
    cd.label,
    cd.service
"""


# ── constatation_reporting (93) ──────────────────────────────────────────────

def build_constatation_sql(prefix_table: str, environnement_client: str) -> str:
    """
    Retourne le SQL CREATE OR REPLACE TABLE pour constatation_reporting.
    Agrège les montants (fee_adjustment, aid, bill) par (trimestre, organisation,
    label_group, subgroup, tranche, user).

    Uniquement disponible pour l'environnement "93".
    """
    if "93" in environnement_client:
        return _constatation_93(prefix_table)
    raise ValueError(
        f"constatation_reporting non supportée pour environnement_client={environnement_client!r} "
        f"(seul '93' est implémenté)"
    )


def _constatation_93(p: str) -> str:
    return f"""
CREATE OR REPLACE TABLE {p}constatation_reporting AS
WITH latest_constatation AS (
    SELECT
        c.*
    FROM {p}constatation c
    WHERE c.billing_type = 'Forfait'
      AND c.subgroup != 'Ticket'
)
SELECT
    c.trimestre_id,
    t.organization_id,
    t.index                             AS trimester_index,
    t.school_year,
    c.label_group,
    c.subgroup,
    c.tranche_label,
    o.rne,
    o.name                              AS nom_etablissement,
    c.user_id,
    SUM(c.fee_adjustment_amount)        AS total_fee_adjustment_amount,
    SUM(c.aid_amount)                   AS total_aid_amount,
    SUM(c.bill_amount)                  AS total_bill_amount
FROM latest_constatation c
INNER JOIN {p}trimester t    ON c.trimestre_id    = t.trimester_id
INNER JOIN {p}organization o ON t.organization_id = o.id_organization
GROUP BY
    c.trimestre_id,
    t.organization_id,
    t.index,
    t.school_year,
    c.label_group,
    c.subgroup,
    c.tranche_label,
    o.rne,
    c.user_id,
    o.name
ORDER BY
    t.school_year,
    t.index,
    o.name
"""


# ── tarification_filter ───────────────────────────────────────────────────────

_TARIFICATION_FILTER_COLUMNS_93 = [
    "school_year",
    "nom_etablissement",
    "department",
    "label_group",
    "type",
    "ips",
]

_TARIFICATION_FILTER_COLUMNS_CENTRE = [
    "school_year",
    "nom_etablissement",
    "facturation_type",
    "department",
    "type",
    "access_software",
    "ips",
]


def build_dernier_passage_sql(prefix_table: str, recent_only: bool = False) -> str:
    """
    Retourne le SQL CREATE OR REPLACE TABLE pour dernier_passage.
    Calcule la date du dernier passage (passage + passage_partner) par organisation.
    """
    p = prefix_table
    date_filter = " WHERE date >= CURRENT_DATE - INTERVAL '1' YEAR" if recent_only else ""
    return f"""
CREATE OR REPLACE TABLE {p}dernier_passage AS
WITH combined_passages AS (
    SELECT date, id_organization FROM {p}passage{date_filter}
    UNION ALL
    SELECT date, id_organization FROM {p}passage_partner{date_filter}
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


def build_tarification_filter_sql(prefix_table: str, environnement_client: str) -> list[str]:
    """
    Retourne une liste de SQL CREATE OR REPLACE TABLE, une par colonne de filtre,
    alimentée par SELECT DISTINCT depuis {p}tarification_1.
    """
    if "centre" in environnement_client:
        cols = _TARIFICATION_FILTER_COLUMNS_CENTRE
    elif "93" in environnement_client:
        cols = _TARIFICATION_FILTER_COLUMNS_93
    else:
        raise ValueError(
            f"tarification_filter non supportée pour environnement_client={environnement_client!r} "
            f"(attendu : contient 'centre' ou '93')"
        )
    p = prefix_table
    return [
        f"CREATE OR REPLACE TABLE {p}filter_tarification_{col} AS "
        f"SELECT DISTINCT {col} FROM {p}tarification_1 WHERE {col} IS NOT NULL ORDER BY {col}"
        for col in cols
    ]

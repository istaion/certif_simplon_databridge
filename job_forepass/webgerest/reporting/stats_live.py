from forepaas.dwh import connect, bulk_insert
from forepaas.core.settings import PARAMS
import ctypes
import gc
import logging
import time
from datetime import datetime


def _release_memory():
    """gc.collect() libère les objets Python mais glibc ne rend pas forcément la
    mémoire libérée à l'OS (fragmentation du tas sur des cycles alloc/free répétés
    de gros DataFrames) — la RSS du process peut grimper au fil des itérations
    même si la mémoire "vivante" ne grossit pas. malloc_trim(0) force ce rendu."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def parse_annee(annee):
    if not annee:
        return None, None
    annee = str(annee).strip()
    if not annee.isdigit() or len(annee) != 4:
        raise ValueError(f"ANNEE invalide : '{annee}'. Format attendu : '2024'")
    return f"{annee}-01-01", f"{int(annee) + 1}-01-01"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
prefix_table = PARAMS['PREFIX_TABLE']
environement = PARAMS['ENVIRONNEMENT_CLIENT']
dataset_cible = f"dwh/db_mg6jk45h_{environement}/"

table_liv59      = f"{prefix_table}stats_liv59"
table_liv        = f"{prefix_table}stats_liv"
table_liv_mois   = f"{prefix_table}stats_liv_mois"
table_liv_annee  = f"{prefix_table}stats_liv_annee"
table_liv_egalim      = f"{prefix_table}stats_liv_egalim"
table_liv_egalim_jour = f"{prefix_table}stats_liv_egalim_jour"
table_stats_dashboard = f"{prefix_table}stats_dashboard"


logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# CTE enriched — logique commune aux 4 tables
# ---------------------------------------------------------------------------
# Sources :
#   - mvtart (statut 2) : mouvements de stock, filtre typmvt=1 (entrées/livraisons)
#   - article (statut 2) : famille/sous-famille, unité, conversion
#   - mvtart_det (statut 2) : marqueurs de modification Egalim (BIO, LABEL, ORIGINE)
#                             Si présent, on utilise les valeurs de mvtart directement
#                             (données vérifiées par l'utilisateur).
#   - detail_article (statut 2) : fallback circuit_court/id_label/id_origine
#   - fourn (statut 1) : nom fournisseur, BIO, circuit_court, id_label
#                        join via fourn.login_site avec routing descfic statut 1/2
#
# Clés de période (colonnes séparées) :
#   ANNEE    = année du jeudi de la semaine ISO (VARCHAR, ex: "2022")
#   SEMAINE  = numéro de semaine ISO (VARCHAR 2 chiffres, ex: "03")
#   MOIS     = mois de la date réelle (VARCHAR 2 chiffres, ex: "01")

def _enriched_cte(p, date_debut=None, date_fin=None):
    """
    Retourne le fragment SQL WITH enriched AS (...).
    p = prefix_table (ex: "wg_")
    """
    date_filter = ""
    if date_debut:
        date_filter += f"\n          AND DATE(CAST(m.dtemvt AS TIMESTAMP)) >= DATE '{date_debut}'"
    if date_fin:
        date_filter += f"\n          AND DATE(CAST(m.dtemvt AS TIMESTAMP)) <  DATE '{date_fin}'"
    return f"""
    WITH enriched AS (
        SELECT
            m.login_site,
            lpc.logingroupe AS logingroup,
            lpc.code_site,

            -- Année du jeudi de la semaine ISO (format YYYY)
            LPAD(CAST(year(date_trunc('week', DATE(CAST(m.dtemvt AS TIMESTAMP)))
                          + interval '3' day) AS VARCHAR), 4, '0')
                AS annee,

            -- Numéro de semaine ISO (format WW, 2 chiffres)
            LPAD(CAST(week(DATE(CAST(m.dtemvt AS TIMESTAMP))) AS VARCHAR), 2, '0')
                AS semaine,

            -- Mois de la date réelle (format MM, 2 chiffres)
            LPAD(CAST(month(DATE(CAST(m.dtemvt AS TIMESTAMP))) AS VARCHAR), 2, '0')
                AS mois,

            -- Date du jour (yyyy-mm-dd)
            DATE(CAST(m.dtemvt AS TIMESTAMP)) AS jour,

            -- Famille et sous-famille article (codes + libellés)
            TRIM(a.codfamart) AS famille_article,
            TRIM(a.sfaart)    AS sous_famille_article,
            TRIM(COALESCE(fam.libfamart, '')) AS lib_famille_article,
            TRIM(COALESCE(sfa.libsfaart, '')) AS lib_sous_famille_article,

            -- Fournisseur (libellé sans espaces de début/fin, vide si absent)
            TRIM(COALESCE(f.libfou, ''))           AS libfou,
            COALESCE(CAST(m.f_ocleunik AS INTEGER), 0) AS idfou,

            -- BIO
            -- Si MVTART_DET existe  → valeur directe de MVTART (vérifiée utilisateur)
            -- Sinon                 → OR entre article/fourn/mvtart
            CASE
                WHEN md.mvcleunik IS NOT NULL
                THEN CASE WHEN CAST(m.bio AS BOOLEAN) = TRUE THEN 1 ELSE 0 END
                ELSE CASE
                    WHEN CAST(a.bio AS BOOLEAN) = TRUE
                      OR COALESCE(CAST(f.bio AS BOOLEAN), FALSE) = TRUE
                      OR CAST(m.bio AS BOOLEAN) = TRUE THEN 1
                    ELSE 0
                END
            END AS bio,

            -- CLOCAL (circuit court)
            CASE
                WHEN md.mvcleunik IS NOT NULL
                THEN CASE WHEN CAST(m.circuit_court AS BOOLEAN) = TRUE THEN 1 ELSE 0 END
                ELSE CASE
                    WHEN COALESCE(CAST(da.circuit_court AS BOOLEAN), FALSE) = TRUE
                      OR COALESCE(CAST(f.circuit_court  AS BOOLEAN), FALSE) = TRUE
                      OR CAST(m.circuit_court AS BOOLEAN) = TRUE THEN 1
                    ELSE 0
                END
            END AS clocal,

            -- LABEL1 (priorité : mvtart > detail_article > fourn)
            CASE
                WHEN md.mvcleunik IS NOT NULL
                THEN COALESCE(CAST(m.id_label AS INTEGER), 0)
                ELSE CASE
                    WHEN COALESCE(CAST(m.id_label  AS INTEGER), 0) != 0
                        THEN CAST(m.id_label AS INTEGER)
                    WHEN da.codart IS NOT NULL
                     AND COALESCE(CAST(da.id_label AS INTEGER), 0) != 0
                        THEN CAST(da.id_label AS INTEGER)
                    ELSE COALESCE(CAST(f.id_label AS INTEGER), 0)
                END
            END AS label1,

            -- id_origine (priorité : mvtart > detail_article)
            CASE
                WHEN COALESCE(CAST(m.id_origine  AS INTEGER), 0) != 0
                    THEN CAST(m.id_origine AS INTEGER)
                ELSE COALESCE(CAST(da.id_origine AS INTEGER), 0)
            END AS id_origine,

            -- Montant TTC = qtef * puf * (1 + TauxTVA/100)
            CAST(m.qtef      AS DOUBLE)
                * CAST(m.puf    AS DOUBLE)
                * (1.0 + CAST(m.taux_tva AS DOUBLE) / 100.0) AS montant,

            -- Montant HT = qtef * puf
            CAST(m.qtef AS DOUBLE) * CAST(m.puf AS DOUBLE) AS montantht,

            -- QTE en unité de la famille (ufam)
            -- usartversufam = 0 → pas de conversion définie :
            --   si usart = KG, on prend qteusart ; sinon QTE = 0
            -- usartversufam ≠ 0 → QTE = qteusart * usartversufam
            CASE
                WHEN CAST(a.usart_vers_ufam AS DOUBLE) = 0.0
                THEN CASE
                    WHEN TRIM(a.usart) = 'KG' THEN CAST(m.qteusart AS DOUBLE)
                    ELSE 0.0
                END
                ELSE CAST(m.qteusart AS DOUBLE) * CAST(a.usart_vers_ufam AS DOUBLE)
            END AS qte

        FROM {p}mvtart m

        -- Mapping login_site → logingroupe (nécessaire pour tous les routings descfic)
        LEFT JOIN {p}login lpc
            ON  lpc.login = m.login_site

        -- Descfic ARTICLE (routing statut 1/2)
        LEFT JOIN {p}descfic dart
            ON  dart.nomfic     = 'ARTICLE'
            AND dart.login_group = lpc.logingroupe

        -- Article obligatoire (sans article, le mouvement est ignoré)
        JOIN {p}article a
            ON  a.arcleunik  = m.arcleunik
            AND a.login_site = CASE WHEN dart.statut = 2 THEN m.login_site ELSE lpc.logingroupe END

        -- Descfic MVTART_DET (routing statut 1/2)
        LEFT JOIN {p}descfic dmd
            ON  dmd.nomfic     = 'MVTART_DET'
            AND dmd.login_group = lpc.logingroupe

        -- MVTART_DET : dédupliqué (relation 0..n), seule la présence compte
        LEFT JOIN (
            SELECT DISTINCT mvcleunik, login_site
            FROM {p}mvtart_det
        ) md
            ON  md.mvcleunik  = m.mvcleunik
            AND md.login_site = CASE WHEN dmd.statut = 2 THEN m.login_site ELSE lpc.logingroupe END

        -- Descfic DETAILARTICLE (routing statut 1/2)
        LEFT JOIN {p}descfic dda
            ON  dda.nomfic     = 'DETAILARTICLE'
            AND dda.login_group = lpc.logingroupe

        -- Détail article : fallback Egalim
        LEFT JOIN {p}detail_article da
            ON  da.codart    = a.codart
            AND da.login_site = CASE WHEN dda.statut = 2 THEN m.login_site ELSE lpc.logingroupe END

        -- Descfic FOURN (routing statut 1/2)
        LEFT JOIN {p}descfic dfourn
            ON  dfourn.nomfic      = 'FOURN'
            AND dfourn.login_group = lpc.logingroupe

        -- Fournisseur : login_site = logingroupe (statut 1) ou login_site réel (statut 2)
        LEFT JOIN {p}fourn f
            ON  f.f_ocleunik = m.f_ocleunik
            AND f.login_site = CASE WHEN dfourn.statut = 2 THEN m.login_site ELSE lpc.logingroupe END

        -- Libellé famille article (descfic FAMART)
        LEFT JOIN {p}descfic dfam
            ON  dfam.nomfic     = 'FAMART'
            AND dfam.login_group = lpc.logingroupe
        LEFT JOIN {p}famart fam
            ON  TRIM(fam.codfamart) = TRIM(a.codfamart)
            AND fam.login_site      = CASE WHEN dfam.statut = 2 THEN m.login_site ELSE lpc.logingroupe END

        -- Libellé sous-famille article (descfic SFAART)
        LEFT JOIN {p}descfic dsfa
            ON  dsfa.nomfic     = 'SFAART'
            AND dsfa.login_group = lpc.logingroupe
        LEFT JOIN {p}sfaart sfa
            ON  TRIM(sfa.codfamart) = TRIM(a.codfamart)
            AND TRIM(sfa.sfaart)    = TRIM(a.sfaart)
            AND sfa.login_site      = CASE WHEN dsfa.statut = 2 THEN m.login_site ELSE lpc.logingroupe END

        WHERE CAST(m.typmvt AS INTEGER) = 1
          AND COALESCE(CAST(lpc.fictif AS BOOLEAN), FALSE) = FALSE{date_filter}
    )
    """


# ---------------------------------------------------------------------------
# Calculs des 4 tables
# ---------------------------------------------------------------------------

def compute_stats_liv59(source, date_debut=None, date_fin=None):
    """
    STATS_LIV59 : agrégation hebdomadaire sans dimension fournisseur.
    Reproduit recupLIV.txt (recupLIV_59).
    Clé : (annee, semaine, code_site, login_site, famille_article, sous_famille_article, bio, clocal, label1)
    """
    logger.info("Calcul STATS_LIV59...")
    query = _enriched_cte(prefix_table, date_debut, date_fin) + """
        SELECT
            annee,
            semaine,
            code_site,
            login_site,
            logingroup,
            famille_article,
            sous_famille_article,
            lib_famille_article,
            lib_sous_famille_article,
            bio,
            clocal,
            label1,
            0                AS label2,
            SUM(qte)         AS qte,
            SUM(montant)     AS montant,
            SUM(montantht)   AS montantht
        FROM enriched
        GROUP BY
            annee, semaine, code_site, login_site, logingroup,
            famille_article, sous_famille_article,
            lib_famille_article, lib_sous_famille_article,
            bio, clocal, label1
    """
    df = source.query(query)
    logger.info(f"  {len(df)} lignes STATS_LIV59")
    return df


def compute_stats_liv(source, date_debut=None, date_fin=None):
    """
    STATS_LIV : agrégation hebdomadaire avec dimension fournisseur (libfou/idfou).
    Reproduit recupLIV_2.txt.
    Clé : (annee, semaine, code_site, login_site, famille_article, sous_famille_article, bio, clocal, label1, libfou)
    """
    logger.info("Calcul STATS_LIV...")
    query = _enriched_cte(prefix_table, date_debut, date_fin) + """
        SELECT
            annee,
            semaine,
            code_site,
            login_site,
            logingroup,
            famille_article,
            sous_famille_article,
            lib_famille_article,
            lib_sous_famille_article,
            bio,
            clocal,
            label1,
            0                        AS label2,
            SUM(qte)                 AS qte,
            SUM(montant)             AS montant,
            libfou,
            idfou,
            SUM(montantht)           AS montantht,
            CAST(NULL AS VARCHAR)    AS codfamfou
        FROM enriched
        GROUP BY
            annee, semaine, code_site, login_site, logingroup,
            famille_article, sous_famille_article,
            lib_famille_article, lib_sous_famille_article,
            bio, clocal, label1,
            libfou, idfou
    """
    df = source.query(query)
    logger.info(f"  {len(df)} lignes STATS_LIV")
    return df


def compute_stats_liv_mois(source, date_debut=None, date_fin=None):
    """
    STATS_LIV_MOIS : agrégation mensuelle avec fournisseur et origine.
    Reproduit recupLIV_2_MOIS.txt.
    Clé : (annee, mois, code_site, login_site, famille_article, sous_famille_article, bio, clocal, label1, id_origine, libfou)
    """
    logger.info("Calcul STATS_LIV_MOIS...")
    query = _enriched_cte(prefix_table, date_debut, date_fin) + """
        SELECT
            annee,
            mois,
            code_site,
            login_site,
            logingroup,
            famille_article,
            sous_famille_article,
            lib_famille_article,
            lib_sous_famille_article,
            bio,
            clocal,
            label1,
            id_origine                   AS idorigine,
            0                            AS label2,
            SUM(qte)                     AS qte,
            SUM(montant)                 AS montant,
            libfou,
            idfou,
            SUM(montantht)               AS montantht,
            CAST(NULL AS VARCHAR)        AS codfamfou
        FROM enriched
        GROUP BY
            annee, mois, code_site, login_site, logingroup,
            famille_article, sous_famille_article,
            lib_famille_article, lib_sous_famille_article,
            bio, clocal, label1, id_origine,
            libfou, idfou
    """
    df = source.query(query)
    logger.info(f"  {len(df)} lignes STATS_LIV_MOIS")
    return df


def compute_stats_liv_annee(source, date_debut=None, date_fin=None):
    """
    STATS_LIV_ANNEE : agrégation annuelle avec fournisseur et origine.
    Reproduit recupLIV_2_ANNEE.txt.
    Clé : (annee, code_site, login_site, famille_article, sous_famille_article, bio, clocal, label1, id_origine, libfou)
    """
    logger.info("Calcul STATS_LIV_ANNEE...")
    query = _enriched_cte(prefix_table, date_debut, date_fin) + """
        SELECT
            annee,
            code_site,
            login_site,
            logingroup,
            famille_article,
            sous_famille_article,
            lib_famille_article,
            lib_sous_famille_article,
            bio,
            clocal,
            label1,
            id_origine                   AS idorigine,
            0                            AS label2,
            SUM(qte)                     AS qte,
            SUM(montant)                 AS montant,
            libfou,
            idfou,
            SUM(montantht)               AS montantht,
            CAST(NULL AS VARCHAR)        AS codfamfou
        FROM enriched
        GROUP BY
            annee, code_site, login_site, logingroup,
            famille_article, sous_famille_article,
            lib_famille_article, lib_sous_famille_article,
            bio, clocal, label1, id_origine,
            libfou, idfou
    """
    df = source.query(query)
    logger.info(f"  {len(df)} lignes STATS_LIV_ANNEE")
    return df


def compute_stats_liv_egalim(source):
    """
    STATS_LIV_EGALIM : table pour Superset, filtre (annee, semaine, code_site).
    Agrège stats_liv59 par (famille_article, sous_famille_article) avec ventilation
    conditionnelle local / bio / egalim / bio+local.
    """
    logger.info("Calcul STATS_LIV_EGALIM depuis stats_liv59...")
    query = f"""
        SELECT
            annee,
            semaine,
            code_site,
            login_site,
            famille_article,
            sous_famille_article,
            lib_famille_article,
            lib_sous_famille_article,
            SUM(montant)                                                          AS montant,
            SUM(montantht)                                                        AS montantht,
            SUM(qte)                                                              AS qte,
            SUM(CASE WHEN clocal = 1 THEN montant ELSE 0 END)                    AS local_valeur,
            SUM(CASE WHEN bio = 1 THEN montant ELSE 0 END)                       AS bio_valeur,
            SUM(CASE WHEN bio = 1 OR clocal = 1 OR label1 != 0
                     THEN montant ELSE 0 END)                                     AS egalim_valeur,
            SUM(CASE WHEN bio = 1 AND clocal = 1 THEN montant ELSE 0 END)        AS bio_local_valeur
        FROM {table_liv59}
        GROUP BY
            annee, semaine, code_site, login_site,
            famille_article, sous_famille_article,
            lib_famille_article, lib_sous_famille_article
    """
    df = source.query(query)
    if df.empty:
        # Trino renvoie des colonnes en object sur 0 ligne — la division plus bas
        # échoue ("Expected numeric dtype, got object instead").
        logger.info("  Aucune donnée, STATS_LIV_EGALIM vide.")
        return df

    # Pourcentages (0 si montant total = 0)
    for prefix in ('local', 'bio', 'egalim', 'bio_local'):
        df[f'{prefix}_pct'] = (
            df[f'{prefix}_valeur'] / df['montant'].replace(0, float('nan')) * 100
        ).fillna(0).round(2)

    logger.info(f"  {len(df)} lignes STATS_LIV_EGALIM")
    return df

def compute_stats_liv_egalim_jour(source, date_debut=None, date_fin=None):
    """
    STATS_LIV_EGALIM_JOUR : même ventilation Egalim que stats_liv_egalim,
    mais agrégée par jour (une date) plutôt que par semaine ISO.
    Clé : (jour, code_site, login_site, famille_article, sous_famille_article)
    """
    logger.info("Calcul STATS_LIV_EGALIM_JOUR...")
    query = _enriched_cte(prefix_table, date_debut, date_fin) + f"""
        SELECT
            jour,
            code_site,
            login_site,
            famille_article,
            sous_famille_article,
            lib_famille_article,
            lib_sous_famille_article,
            SUM(montant)                                                          AS montant,
            SUM(montantht)                                                        AS montantht,
            SUM(qte)                                                              AS qte,
            SUM(CASE WHEN clocal = 1 THEN montant ELSE 0 END)                    AS local_valeur,
            SUM(CASE WHEN bio = 1 THEN montant ELSE 0 END)                       AS bio_valeur,
            SUM(CASE WHEN bio = 1 OR clocal = 1 OR label1 != 0
                     THEN montant ELSE 0 END)                                     AS egalim_valeur,
            SUM(CASE WHEN bio = 1 AND clocal = 1 THEN montant ELSE 0 END)        AS bio_local_valeur
        FROM enriched
        GROUP BY
            jour, code_site, login_site,
            famille_article, sous_famille_article,
            lib_famille_article, lib_sous_famille_article
    """
    df = source.query(query)
    if df.empty:
        # Trino renvoie des colonnes en object sur 0 ligne — la division plus bas
        # échoue ("Expected numeric dtype, got object instead").
        logger.info("  Aucune donnée, STATS_LIV_EGALIM_JOUR vide.")
        return df

    for prefix in ('local', 'bio', 'egalim', 'bio_local'):
        df[f'{prefix}_pct'] = (
            df[f'{prefix}_valeur'] / df['montant'].replace(0, float('nan')) * 100
        ).fillna(0).round(2)

    logger.info(f"  {len(df)} lignes STATS_LIV_EGALIM_JOUR")
    return df


def compute_stats_dashboard(source):
    """
    STATS_DASHBOARD : stats pour la vache et le poisson et le poivache du dashboard webgerest
    """
    logger.info("Calcul STATS_DASHBOARD...")
    query = f"""SELECT
        s.annee,
        lpc.logingroupe                                         AS login_group,
        COUNT(DISTINCT s.login_site)                            AS nb_site,
        CASE
            WHEN fa.type = 1 OR sf.type = 1 THEN 'viande'
            WHEN fa.type = 2 OR sf.type = 2 THEN 'poisson'
            ELSE 'autre'
        END                                                     AS type_produit,
        CASE WHEN s.clocal = 1 THEN 'local' ELSE 'non local' END   AS local_label,
        CASE WHEN s.bio = 1    THEN 'bio'   ELSE 'non bio'   END   AS bio_label,
        CASE
            WHEN s.bio = 1
            OR (s.label1 != 0 AND lb.egalim = true)
            THEN 'EGALIM'
            ELSE 'non EGALIM'
        END                                                     AS egalim_label,
        SUM(s.montant)                                          AS montant,
        SUM(s.montantht)                                        AS montantht
    FROM {prefix_table}stats_liv_annee s
    LEFT JOIN {prefix_table}login lpc
        ON s.login_site = lpc.login
    LEFT JOIN {prefix_table}descfic dfam
        ON dfam.nomfic = 'FAMART' AND dfam.login_group = lpc.logingroupe
    LEFT JOIN {prefix_table}famart fa
        ON TRIM(fa.codfamart) = TRIM(s.famille_article)
        AND fa.login_site = CASE WHEN dfam.statut = 2 THEN s.login_site ELSE lpc.logingroupe END
    LEFT JOIN {prefix_table}descfic dsfa
        ON dsfa.nomfic = 'SFAART' AND dsfa.login_group = lpc.logingroupe
    LEFT JOIN {prefix_table}sfaart sf
        ON TRIM(sf.codfamart) = TRIM(s.famille_article)
        AND TRIM(sf.sfaart) = TRIM(s.sous_famille_article)
        AND sf.login_site = CASE WHEN dsfa.statut = 2 THEN s.login_site ELSE lpc.logingroupe END
    LEFT JOIN {prefix_table}descfic dlb
        ON dlb.nomfic = 'LABEL' AND dlb.login_group = lpc.logingroupe
    LEFT JOIN {prefix_table}label lb
        ON lb.id_label = s.label1
        AND lb.login_site = CASE WHEN dlb.statut = 2 THEN s.login_site ELSE lpc.logingroupe END
    WHERE TRIM(fa.cptfam_1) = '6011'
      AND (TRY_CAST(TRIM(s.famille_article) AS INTEGER) < 90
           OR TRY_CAST(TRIM(s.famille_article) AS INTEGER) = 99)
    GROUP BY
        s.annee,
        lpc.logingroupe,
        CASE
            WHEN fa.type = 1 OR sf.type = 1 THEN 'viande'
            WHEN fa.type = 2 OR sf.type = 2 THEN 'poisson'
            ELSE 'autre'
        END,
        CASE WHEN s.clocal = 1 THEN 'local' ELSE 'non local' END,
        CASE WHEN s.bio = 1    THEN 'bio'   ELSE 'non bio'   END,
        CASE
            WHEN s.bio = 1
            OR (s.label1 != 0 AND lb.egalim = true)
            THEN 'EGALIM'
            ELSE 'non EGALIM'
        END
    """
    df = source.query(query)
    logger.info(f"  {len(df)} lignes STATS_DASHBOARD")
    return df

# ---------------------------------------------------------------------------
# Point d'entrée ForePaaS
# ---------------------------------------------------------------------------

def _run_with_retry(label: str, fn, max_retries: int = 5, retry_delay: int = 60):
    for attempt in range(1, max_retries + 2):
        try:
            return fn()
        except Exception as e:
            if attempt <= max_retries:
                logger.warning(
                    f"[retry] {label} : erreur transitoire (tentative {attempt}/{max_retries + 1}),"
                    f" retry dans {retry_delay}s — {e}"
                )
                time.sleep(retry_delay)
            else:
                logger.error(f"[retry] {label} : abandon après {max_retries} retries")
                raise


def _year_range(source) -> tuple:
    """Détecte MIN/MAX année depuis MVTART.dtemvt."""
    df = source.query(
        f"SELECT MIN(YEAR(DATE(CAST(dtemvt AS TIMESTAMP)))) AS min_year, "
        f"MAX(YEAR(DATE(CAST(dtemvt AS TIMESTAMP)))) AS max_year "
        f"FROM {prefix_table}mvtart WHERE dtemvt IS NOT NULL"
    )
    row = df.iloc[0]
    return int(row["min_year"]), int(row["max_year"])


def _process_year_liv(source, date_debut: str, date_fin: str, annee: str, now) -> None:
    """Calcule et écrit les tables LIV filtrables par année, une tranche à la fois
    (delete ciblé sur l'année — ne touche pas aux autres années déjà en base)."""
    logger.info(f"=== Année {annee} ===")

    logger.info("  Calcul STATS_LIV59...")
    df = compute_stats_liv59(source, date_debut, date_fin)
    if not df.empty:
        df['date_import'] = now
        df['date_modif']  = now
        def _insert_liv59():
            source.query(f"DELETE FROM {table_liv59} WHERE annee = '{annee}'")
            bulk_insert(source, table_liv59, df)
        _run_with_retry(table_liv59, _insert_liv59)
    del df

    logger.info("  Calcul STATS_LIV...")
    df = compute_stats_liv(source, date_debut, date_fin)
    if not df.empty:
        df['date_import'] = now
        df['date_modif']  = now
        def _insert_liv():
            source.query(f"DELETE FROM {table_liv} WHERE annee = '{annee}'")
            bulk_insert(source, table_liv, df)
        _run_with_retry(table_liv, _insert_liv)
    del df

    logger.info("  Calcul STATS_LIV_MOIS...")
    df = compute_stats_liv_mois(source, date_debut, date_fin)
    if not df.empty:
        df['date_import'] = now
        df['date_modif']  = now
        def _insert_liv_mois():
            source.query(f"DELETE FROM {table_liv_mois} WHERE annee = '{annee}'")
            bulk_insert(source, table_liv_mois, df)
        _run_with_retry(table_liv_mois, _insert_liv_mois)
    del df

    logger.info("  Calcul STATS_LIV_ANNEE...")
    df = compute_stats_liv_annee(source, date_debut, date_fin)
    if not df.empty:
        df['date_import'] = now
        df['date_modif']  = now
        def _insert_liv_annee():
            source.query(f"DELETE FROM {table_liv_annee} WHERE annee = '{annee}'")
            bulk_insert(source, table_liv_annee, df)
        _run_with_retry(table_liv_annee, _insert_liv_annee)
    del df

    logger.info("  Calcul STATS_LIV_EGALIM_JOUR...")
    df = compute_stats_liv_egalim_jour(source, date_debut, date_fin)
    if not df.empty:
        df['date_import'] = now
        df['date_modif']  = now
        def _insert_liv_egalim_jour():
            source.query(
                f"DELETE FROM {table_liv_egalim_jour} "
                f"WHERE jour >= DATE '{date_debut}' AND jour < DATE '{date_fin}'"
            )
            bulk_insert(source, table_liv_egalim_jour, df)
        _run_with_retry(table_liv_egalim_jour, _insert_liv_egalim_jour)
    del df

    _release_memory()


def _process_final_liv(source, now) -> None:
    """STATS_LIV_EGALIM et STATS_DASHBOARD agrègent tout l'historique (pas de filtre
    date possible) — calculés une seule fois, après la boucle sur les années."""
    logger.info("=== STATS_LIV_EGALIM / STATS_DASHBOARD (agrégats full-history) ===")

    logger.info("  Calcul STATS_LIV_EGALIM...")
    df = compute_stats_liv_egalim(source)
    if not df.empty:
        df['date_import'] = now
        df['date_modif']  = now
        def _insert_liv_egalim():
            source.query(f"DELETE FROM {table_liv_egalim}")
            bulk_insert(source, table_liv_egalim, df)
        _run_with_retry(table_liv_egalim, _insert_liv_egalim)
    del df
    _release_memory()

    logger.info("  Calcul STATS_DASHBOARD...")
    df = compute_stats_dashboard(source)
    if not df.empty:
        def _insert_dashboard():
            source.query(f"DELETE FROM {table_stats_dashboard}")
            bulk_insert(source, table_stats_dashboard, df)
        _run_with_retry(table_stats_dashboard, _insert_dashboard)
    del df
    _release_memory()


def customfunc(event):
    now = datetime.now()
    annee = PARAMS.get('ANNEE', None)
    annee_range = PARAMS.get('ANNEE_RANGE', None)

    if annee:
        # Un seul run = une seule année (comportement historique, inchangé).
        source = connect(dataset_cible)
        date_debut, date_fin = parse_annee(annee)
        logger.info(f"Filtre année : {annee} ({date_debut} → {date_fin})")
        _run_with_retry(
            f"année {annee}", lambda: _process_year_liv(source, date_debut, date_fin, annee, now),
            max_retries=2, retry_delay=30,
        )
    else:
        source = connect(dataset_cible)
        annee_min, annee_max = _run_with_retry(
            "détection plage d'années", lambda: _year_range(source),
            max_retries=2, retry_delay=30,
        )
        if annee_range:
            # ANNEE_RANGE=X : ne traite que les X dernières années (borné par le
            # minimum réel détecté si X dépasse la profondeur d'historique).
            n = int(annee_range)
            annee_min = max(annee_min, annee_max - n + 1)
            logger.info(f"ANNEE_RANGE={n} — traitement des {n} dernières années : {annee_min}-{annee_max}")
        else:
            # Pas d'ANNEE ni d'ANNEE_RANGE : traite tout l'historique, mais année
            # par année (plutôt qu'en un seul bloc pandas) pour borner la mémoire.
            logger.info(f"Pas de filtre année — traitement complet {annee_min}-{annee_max}, année par année")
        for year in range(annee_min, annee_max + 1):
            date_debut, date_fin = f"{year}-01-01", f"{year + 1}-01-01"
            # Reconnexion à chaque année : une connexion PolarData/Trino réutilisée sur
            # des dizaines de requêtes successives peut accumuler des buffers internes
            # au fil d'un job long multi-années — en repartir à zéro à chaque itération
            # limite ce risque d'accumulation mémoire indépendamment du garbage collector.
            source = connect(dataset_cible)
            # Chaque année est ré-essayée en entier en cas d'erreur transitoire Trino/Iceberg
            # (delete+insert par année étant idempotent, un retry complet est sûr).
            _run_with_retry(
                f"année {year}",
                lambda dd=date_debut, df=date_fin, y=year, s=source: _process_year_liv(s, dd, df, str(y), now),
                max_retries=2, retry_delay=30,
            )
            del source
            _release_memory()

    source = connect(dataset_cible)

    _run_with_retry(
        "agrégats full-history", lambda: _process_final_liv(source, now),
        max_retries=2, retry_delay=30,
    )

    logger.info("Job stats_liv terminé.")

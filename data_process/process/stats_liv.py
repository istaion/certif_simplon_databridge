import logging
from datetime import datetime

import pandas as pd

from data_process.db.trino_client import TrinoClient


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
# Calculs des tables
# ---------------------------------------------------------------------------

def compute_stats_liv59(db: TrinoClient, prefix: str, date_debut=None, date_fin=None) -> pd.DataFrame:
    logger.info("Calcul STATS_LIV59...")
    query = _enriched_cte(prefix, date_debut, date_fin) + """
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
    df = db.query_as_dataframe(query)
    logger.info(f"  {len(df)} lignes STATS_LIV59")
    return df


def compute_stats_liv(db: TrinoClient, prefix: str, date_debut=None, date_fin=None) -> pd.DataFrame:
    logger.info("Calcul STATS_LIV...")
    query = _enriched_cte(prefix, date_debut, date_fin) + """
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
    df = db.query_as_dataframe(query)
    logger.info(f"  {len(df)} lignes STATS_LIV")
    return df


def compute_stats_liv_mois(db: TrinoClient, prefix: str, date_debut=None, date_fin=None) -> pd.DataFrame:
    logger.info("Calcul STATS_LIV_MOIS...")
    query = _enriched_cte(prefix, date_debut, date_fin) + """
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
    df = db.query_as_dataframe(query)
    logger.info(f"  {len(df)} lignes STATS_LIV_MOIS")
    return df


def compute_stats_liv_annee(db: TrinoClient, prefix: str, date_debut=None, date_fin=None) -> pd.DataFrame:
    logger.info("Calcul STATS_LIV_ANNEE...")
    query = _enriched_cte(prefix, date_debut, date_fin) + """
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
    df = db.query_as_dataframe(query)
    logger.info(f"  {len(df)} lignes STATS_LIV_ANNEE")
    return df


def compute_stats_liv_egalim(db: TrinoClient, prefix: str) -> pd.DataFrame:
    logger.info("Calcul STATS_LIV_EGALIM depuis stats_liv59...")
    table_liv59 = f"{prefix}stats_liv59"
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
    df = db.query_as_dataframe(query)

    # Cast to float — Trino returns object dtype when the table is empty (0 rows)
    for col in ['montant', 'montantht', 'qte', 'local_valeur', 'bio_valeur', 'egalim_valeur', 'bio_local_valeur']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # Pourcentages (0 si montant total = 0)
    for pct_prefix in ('local', 'bio', 'egalim', 'bio_local'):
        df[f'{pct_prefix}_pct'] = (
            df[f'{pct_prefix}_valeur'] / df['montant'].replace(0, float('nan')) * 100
        ).fillna(0).round(2)

    logger.info(f"  {len(df)} lignes STATS_LIV_EGALIM")
    return df


def compute_stats_dashboard(db: TrinoClient, prefix: str) -> pd.DataFrame:
    logger.info("Calcul STATS_DASHBOARD...")
    query = f"""SELECT
        s.annee,
        lpc.logingroupe                                         AS login_group,
        COUNT(DISTINCT s.code_site)                             AS nb_site,
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
    FROM {prefix}stats_liv_annee s
    LEFT JOIN {prefix}login lpc
        ON s.login_site = lpc.login
    LEFT JOIN {prefix}descfic dfam
        ON dfam.nomfic = 'FAMART' AND dfam.login_group = lpc.logingroupe
    LEFT JOIN {prefix}famart fa
        ON TRIM(fa.codfamart) = TRIM(s.famille_article)
        AND fa.login_site = CASE WHEN dfam.statut = 2 THEN s.login_site ELSE lpc.logingroupe END
    LEFT JOIN {prefix}descfic dsfa
        ON dsfa.nomfic = 'SFAART' AND dsfa.login_group = lpc.logingroupe
    LEFT JOIN {prefix}sfaart sf
        ON TRIM(sf.codfamart) = TRIM(s.famille_article)
        AND TRIM(sf.sfaart) = TRIM(s.sous_famille_article)
        AND sf.login_site = CASE WHEN dsfa.statut = 2 THEN s.login_site ELSE lpc.logingroupe END
    LEFT JOIN {prefix}descfic dlb
        ON dlb.nomfic = 'LABEL' AND dlb.login_group = lpc.logingroupe
    LEFT JOIN {prefix}label lb
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
    df = db.query_as_dataframe(query)
    logger.info(f"  {len(df)} lignes STATS_DASHBOARD")
    return df

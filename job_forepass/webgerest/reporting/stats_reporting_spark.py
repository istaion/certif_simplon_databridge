"""
Custom PySpark Action — reporting Webgerest (FCJ + LIV), traitement année par année.

Reproduit exactement la logique de stats_fcj.py et stats_live.py (mêmes requêtes SQL,
mêmes transformations pandas, mêmes tables cibles), mais dans un unique job, en bouclant
sur chaque année calendaire et en libérant la mémoire (del + gc.collect()) entre deux
années plutôt que de charger tout l'historique en une seule fois.

Tables produites :
    FCJ  (par année) : stats_fcj59, stats_fcj59_detail, stats_recap_site,
                        stat_effect_cred_1, stats_dashboard_effect
    LIV  (par année) : stats_liv59, stats_liv, stats_liv_mois, stats_liv_annee,
                        stats_liv_egalim_jour
    LIV  (une fois, après la boucle — agrégats sur tout l'historique) :
                        stats_liv_egalim, stats_dashboard

PARAMS requis :
    ENVIRONNEMENT_CLIENT, PREFIX_TABLE
PARAMS optionnels :
    ZONE_SCOLAIRE   (défaut 'B')
    ANNEE_DEBUT, ANNEE_FIN  (format "2020" ; si absents, détectés automatiquement
                             depuis MIN/MAX(efdate) et MIN/MAX(dtemvt) en base)
    ANNEE_RANGE     (ex: "3" → ne traite que les 3 dernières années détectées ;
                     ignoré si ANNEE_DEBUT/ANNEE_FIN sont fournis)

Docs :
- https://docs.dataplatform.ovh.net/#/en/product/dpe/actions/custom-pyspark/
"""

import ctypes
import gc
import json
import logging
import time
from datetime import datetime, timedelta
from logging import getLogger

import numpy as np
import pandas as pd
from forepaas.core.settings import PARAMS
from forepaas.dwh import bulk_insert, connect
from pyspark.sql import SparkSession

logger = getLogger(__name__)


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

# ── Paramètres ─────────────────────────────────────────────────────────────────

prefix_table = PARAMS["PREFIX_TABLE"]
environement = PARAMS["ENVIRONNEMENT_CLIENT"]
dataset_cible = f"dwh/db_mg6jk45h_{environement}/"

table_fcj59            = f"{prefix_table}stats_fcj59"
table_fcj59_detail     = f"{prefix_table}stats_fcj59_detail"
table_recap_site       = f"{prefix_table}stats_recap_site"
table_effect_cred      = f"{prefix_table}stat_effect_cred_1"
table_dashboard_effect = f"{prefix_table}stats_dashboard_effect"

table_liv59            = f"{prefix_table}stats_liv59"
table_liv              = f"{prefix_table}stats_liv"
table_liv_mois         = f"{prefix_table}stats_liv_mois"
table_liv_annee        = f"{prefix_table}stats_liv_annee"
table_liv_egalim       = f"{prefix_table}stats_liv_egalim"
table_liv_egalim_jour  = f"{prefix_table}stats_liv_egalim_jour"
table_stats_dashboard  = f"{prefix_table}stats_dashboard"


# ── Utilitaires communs ─────────────────────────────────────────────────────────

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


def _replace_year(source, table: str, where: str, df: pd.DataFrame) -> None:
    """DELETE FROM {table} WHERE {where} puis bulk_insert(df) — remplace uniquement
    la tranche concernée (année ou plage de dates), sans toucher aux autres années
    déjà chargées dans la table."""
    if df is None or df.empty:
        logger.info(f"  {table} : DataFrame vide, tranche inchangée")
        return

    def _execute():
        source.query(f"DELETE FROM {table} WHERE {where}")
        bulk_insert(source, table, df)
        logger.info(f"  {table} : {len(df)} lignes insérées ({where})")

    _run_with_retry(table, _execute)


def _replace_all(source, table: str, df: pd.DataFrame) -> None:
    """DELETE FROM {table} (sans condition) puis bulk_insert(df) — recalcul complet,
    utilisé uniquement pour les tables agrégées sur tout l'historique."""
    if df is None or df.empty:
        logger.info(f"  {table} : DataFrame vide, table inchangée")
        return

    def _execute():
        source.query(f"DELETE FROM {table}")
        bulk_insert(source, table, df)
        logger.info(f"  {table} : {len(df)} lignes insérées (full)")

    _run_with_retry(table, _execute)


def _detect_year_range(source) -> tuple[int, int]:
    """Détecte MIN/MAX année depuis EFFECT.efdate et MVTART.dtemvt."""
    query = f"""
        SELECT MIN(y) AS min_year, MAX(y) AS max_year FROM (
            SELECT YEAR(CAST(efdate AS DATE)) AS y FROM {prefix_table}effect WHERE efdate IS NOT NULL
            UNION ALL
            SELECT YEAR(DATE(CAST(dtemvt AS TIMESTAMP))) AS y FROM {prefix_table}mvtart WHERE dtemvt IS NOT NULL
        )
    """
    df = source.query(query)
    row = df.iloc[0]
    return int(row["min_year"]), int(row["max_year"])


# =================================================================================
# FCJ — chargement des tables sources (identique à stats_fcj.py)
# =================================================================================

def load_effect_agg(source, date_debut=None, date_fin=None):
    logger.info("Agrégation EFFECT + CATEG...")
    query = f"""
        SELECT
            CAST(e.efdate AS DATE)   AS datej,
            CAST(CAST(TRIM(e.codss1) AS INTEGER) AS VARCHAR) AS codss1,
            CAST(CAST(TRIM(e.codss2) AS INTEGER) AS VARCHAR) AS codss2,
            lpc.code_site,
            e.login_site,
            SUM(e.efreel)            AS efreel,
            SUM(
                CASE CAST(e.typfac AS INTEGER)
                    WHEN 1 THEN CAST(e.eftheo AS DOUBLE) * CAST(e.credit AS DOUBLE)
                    WHEN 2 THEN CAST(e.efreel AS DOUBLE) * CAST(e.credit AS DOUBLE)
                    ELSE 0.0
                END
            )                        AS totcredit
        FROM {prefix_table}effect e
        JOIN {prefix_table}login lpc ON lpc.login = e.login_site
        JOIN {prefix_table}descfic d
            ON  d.nomfic     = 'CATEG'
            AND d.login_group = lpc.logingroupe
        JOIN {prefix_table}categ c
            ON  TRIM(c.codcat) = TRIM(e.codcat)
            AND c.login_site   = CASE WHEN d.statut = 2 THEN e.login_site ELSE lpc.logingroupe END
            AND COALESCE(c.noncompte, FALSE) = FALSE
        WHERE 1=1
          {f"AND CAST(e.efdate AS DATE) >= DATE '{date_debut}'" if date_debut else ""}
          {f"AND CAST(e.efdate AS DATE) <  DATE '{date_fin}'"  if date_fin  else ""}
        GROUP BY
            CAST(e.efdate AS DATE),
            TRIM(e.codss1),
            TRIM(e.codss2),
            lpc.code_site,
            e.login_site
    """
    df = source.query(query)
    df['datej']  = pd.to_datetime(df['datej']).dt.normalize()
    df['codss1'] = df['codss1'].astype(str).str.strip()
    df['codss2'] = df['codss2'].astype(str).str.strip()
    logger.info(f"  {len(df)} groupes depuis EFFECT")
    return df


def load_mvtart_agg(source, date_debut=None, date_fin=None):
    logger.info("Agrégation MVTART (typmvt=2)...")
    query = f"""
        SELECT
            DATE(CAST(m.dtemvt AS TIMESTAMP)) AS datej,
            CAST(CAST(TRIM(m.codss1) AS INTEGER) AS VARCHAR) AS codss1,
            CAST(CAST(TRIM(m.codss2) AS INTEGER) AS VARCHAR) AS codss2,
            lpc.code_site,
            m.login_site,
            SUM(CAST(m.totttc AS DOUBLE))     AS totsortie
        FROM {prefix_table}mvtart m
        LEFT JOIN {prefix_table}login lpc ON lpc.login = m.login_site
        WHERE CAST(m.typmvt AS INTEGER) = 2
          AND m.codss1 IS NOT NULL
          AND m.codss2 IS NOT NULL
          {f"AND DATE(CAST(m.dtemvt AS TIMESTAMP)) >= DATE '{date_debut}'" if date_debut else ""}
          {f"AND DATE(CAST(m.dtemvt AS TIMESTAMP)) <  DATE '{date_fin}'"  if date_fin  else ""}
        GROUP BY
            DATE(CAST(m.dtemvt AS TIMESTAMP)),
            TRIM(m.codss1),
            TRIM(m.codss2),
            lpc.code_site,
            m.login_site
    """
    df = source.query(query)
    df['datej']  = pd.to_datetime(df['datej']).dt.normalize()
    df['codss1'] = df['codss1'].astype(str).str.strip()
    df['codss2'] = df['codss2'].astype(str).str.strip()
    logger.info(f"  {len(df)} groupes depuis MVTART")
    return df


def load_gaspi_saisie_gen(source, date_debut=None, date_fin=None):
    logger.info("Chargement GASPI_SAISIE_GEN...")
    query = f"""
        SELECT
            CAST(g.datej AS DATE)        AS datej,
            CAST(CAST(TRIM(g.codss1) AS INTEGER) AS VARCHAR) AS codss1,
            CAST(CAST(TRIM(g.codss2) AS INTEGER) AS VARCHAR) AS codss2,
            g.login_site,
            MAX(CAST(g.eff_prev AS DOUBLE))         AS eff_prev,
            MAX(CAST(g.eff_prod AS DOUBLE))         AS eff_prod,
            MAX(CAST(g.eff_reel_service AS DOUBLE)) AS eff_reel_service
        FROM {prefix_table}gaspi_saisie_gen g
        WHERE 1=1
          {f"AND CAST(g.datej AS DATE) >= DATE '{date_debut}'" if date_debut else ""}
          {f"AND CAST(g.datej AS DATE) <  DATE '{date_fin}'"  if date_fin  else ""}
        GROUP BY
            CAST(g.datej AS DATE),
            TRIM(g.codss1),
            TRIM(g.codss2),
            g.login_site
    """
    df = source.query(query)
    df['datej']  = pd.to_datetime(df['datej']).dt.normalize()
    df['codss1'] = df['codss1'].astype(str).str.strip()
    df['codss2'] = df['codss2'].astype(str).str.strip()
    logger.info(f"  {len(df)} lignes depuis GASPI_SAISIE_GEN")
    return df


def load_feuille(source, date_debut=None, date_fin=None):
    logger.info("Chargement FEUILLE...")
    query = f"""
        SELECT
            CAST(f.efdate AS DATE)  AS datej,
            CAST(CAST(TRIM(f.codss1) AS INTEGER) AS VARCHAR) AS codss1,
            CAST(CAST(TRIM(f.codss2) AS INTEGER) AS VARCHAR) AS codss2,
            f.login_site,
            f.nb_vegetarien,
            f.id_animation,
            f.type_menu
        FROM {prefix_table}feuille f
        WHERE 1=1
          {f"AND CAST(f.efdate AS DATE) >= DATE '{date_debut}'" if date_debut else ""}
          {f"AND CAST(f.efdate AS DATE) <  DATE '{date_fin}'"  if date_fin  else ""}
    """
    df = source.query(query)
    df['datej']  = pd.to_datetime(df['datej']).dt.normalize()
    df['codss1'] = df['codss1'].astype(str).str.strip()
    df['codss2'] = df['codss2'].astype(str).str.strip()
    df = df.drop_duplicates(subset=['datej', 'codss1', 'codss2', 'login_site'])

    def expand_json_list(series, prefix, n=5, dtype=None):
        def parse(val):
            try:
                lst = json.loads(str(val).replace('False', 'false').replace('True', 'true'))
                return lst[:n] + [None] * max(0, n - len(lst))
            except (ValueError, TypeError, json.JSONDecodeError):
                return [None] * n
        expanded = pd.DataFrame(series.apply(parse).tolist(),
                                columns=[f"{prefix}_{i+1}" for i in range(n)],
                                index=series.index)
        if dtype:
            expanded = expanded.astype(dtype, errors='ignore')
        return expanded

    df = pd.concat([
        df.drop(columns=['id_animation', 'type_menu']),
        expand_json_list(df['id_animation'], 'id_animation', dtype='Int64'),
        expand_json_list(df['type_menu'],    'type_menu',    dtype='Int64'),
    ], axis=1)

    logger.info(f"  {len(df)} lignes depuis FEUILLE")
    return df


def load_logingroup(source):
    logger.info("Chargement logingroup depuis login...")
    query = f"""
        SELECT login, logingroupe AS logingroup
        FROM {prefix_table}login
        WHERE COALESCE(CAST(fictif AS BOOLEAN), FALSE) = FALSE
    """
    df = source.query(query)
    logger.info(f"  {len(df)} logins chargés")
    return df.set_index('login')['logingroup'].to_dict()


def get_descfic_statut_map(source, nomfic: str) -> dict:
    df = source.query(
        f"SELECT login_group, statut FROM {prefix_table}descfic WHERE UPPER(nomfic) = '{nomfic.upper()}'"
    )
    if isinstance(df, pd.DataFrame) and not df.empty:
        return dict(zip(df["login_group"], df["statut"]))
    return {}


def load_typss1(source):
    logger.info("Chargement TYPSS1...")
    query = f"""
        SELECT
            CAST(CAST(TRIM(t.codss1) AS INTEGER) AS VARCHAR) AS codss1,
            TRIM(t.codcpt) AS codcpt,
            t.login_site
        FROM {prefix_table}typss1 t
    """
    df = source.query(query)
    df['codss1'] = df['codss1'].astype(str).str.strip()
    df['codcpt'] = df['codcpt'].astype(str).str.strip()
    df = df[df['codcpt'] == '6011']
    logger.info(f"  {len(df)} lignes codcpt='6011' depuis TYPSS1")
    return df


def load_ntarif(source):
    logger.info("Chargement NTARIF...")
    query = f"""
        SELECT
            CAST(n.exercice AS VARCHAR)  AS annee,
            CAST(CAST(TRIM(n.prestation) AS INTEGER) AS VARCHAR) AS codss1,
            CAST(n.codcli AS VARCHAR)  AS codcli,
            TRIM(n.codcat)            AS codcat,
            n.login_site,
            n.creditbrut
        FROM {prefix_table}ntarif n
    """
    df = source.query(query)
    df['codss1'] = df['codss1'].astype(str).str.strip()
    df['codcli'] = df['codcli'].astype(str).str.strip()
    df['codcat'] = df['codcat'].astype(str).str.strip()
    logger.info(f"  {len(df)} lignes depuis NTARIF")
    return df


def load_trimestre(source):
    logger.info("Chargement TRIMESTRE...")
    query = f"""
        SELECT
            CAST(t.exercice AS VARCHAR)  AS exercice,
            CAST(t.datdeb AS DATE)       AS datdeb,
            CAST(t.datfin AS DATE)       AS datfin,
            CAST(t.notrim AS INTEGER)    AS notrim,
            t.login_site
        FROM {prefix_table}trimestre t
    """
    df = source.query(query)
    df['datdeb'] = pd.to_datetime(df['datdeb']).dt.normalize()
    df['datfin'] = pd.to_datetime(df['datfin']).dt.normalize()
    logger.info(f"  {len(df)} lignes depuis TRIMESTRE")
    return df


# =================================================================================
# FCJ — logique métier (identique à stats_fcj.py)
# =================================================================================

def apply_6011_consolidation(df_fcj59, df_typss1, lk_col: str = 'login_site'):
    logger.info("Consolidation 6011...")
    logger.info(f"  df_fcj59 entrant : {len(df_fcj59)} lignes, lk_col={lk_col!r}")
    df = df_fcj59.merge(
        df_typss1[['codss1', 'codcpt', lk_col]],
        on=['codss1', lk_col],
        how='left'
    )
    df['codcpt'] = df['codcpt'].fillna('').str.strip()

    mask_6011 = (df['codcpt'] == '6011') & (df['codss1'].str.strip() != '1')
    df_6011 = df[mask_6011].copy()
    logger.info(f"  Lignes 6011 éligibles (codss1 != '1') : {len(df_6011)}")

    df.drop(columns=['codcpt'], inplace=True)

    if df_6011.empty:
        logger.info("  Aucune ligne 6011 trouvée")
        return df

    keys = ['datej', 'codss2', 'code_site', 'login_site']
    df_6011_agg = (
        df_6011.groupby(keys)['totsortie']
        .sum()
        .reset_index()
        .rename(columns={'totsortie': 'totsortie_6011'})
    )

    mask_01 = df['codss1'].str.strip() == '1'
    df_01 = df[mask_01].copy()
    df_non_01 = df[~mask_01].copy()

    df_01 = df_01.merge(df_6011_agg, on=keys, how='left')
    df_01['totsortie'] = df_01['totsortie'] + df_01['totsortie_6011'].fillna(0)
    df_01.drop(columns=['totsortie_6011'], inplace=True)

    if not df_01.empty:
        merged_check = df_6011_agg[keys].merge(
            df_01[keys].drop_duplicates(),
            on=keys,
            how='left',
            indicator=True
        )
        df_new_01 = df_6011_agg[merged_check['_merge'].values == 'left_only'].copy()
    else:
        df_new_01 = df_6011_agg.copy()
    logger.info(f"  Nouvelles lignes codss1='1' à créer : {len(df_new_01)}")

    if not df_new_01.empty:
        df_new_01['codss1']    = '1'
        df_new_01['efreel']    = 0.0
        df_new_01['totcredit'] = 0.0
        df_new_01.rename(columns={'totsortie_6011': 'totsortie'}, inplace=True)
        for col in df.columns:
            if col not in df_new_01.columns:
                df_new_01[col] = np.nan
        df_new_01 = df_new_01[df.columns]

    parts = [df_non_01, df_01]
    if not df_new_01.empty:
        parts.append(df_new_01)
    result = pd.concat(parts, ignore_index=True)
    logger.info(f"  Résultat : {len(result)} lignes après consolidation 6011")
    return result


def compute_stats_fcj59(source, date_debut=None, date_fin=None):
    join_keys = ['datej', 'codss1', 'codss2', 'code_site', 'login_site']

    logingroup_map = load_logingroup(source)

    df_effect  = load_effect_agg(source, date_debut, date_fin)
    df_mvtart  = load_mvtart_agg(source, date_debut, date_fin)

    df = pd.merge(df_effect, df_mvtart, on=join_keys, how='outer')
    del df_effect, df_mvtart
    df['efreel']    = df['efreel'].fillna(0)
    df['totcredit'] = df['totcredit'].fillna(0)
    df['totsortie'] = df['totsortie'].fillna(0)

    df = df[(df['efreel'] != 0) | (df['totcredit'] != 0) | (df['totsortie'] != 0)].copy()
    logger.info(f"Après FULL JOIN + filtre : {len(df)} lignes")

    if df.empty:
        # Arrêt anticipé : évite les merges GASPI/FEUILLE/TYPSS1 (coûteux et inutiles
        # sur un DataFrame vide) et le bug de dtype (.apply() sur 0 ligne renvoie
        # float64 au lieu de object, ce qui fait échouer le merge sur les clés _lk).
        logger.info("Aucune donnée pour cette période — arrêt anticipé.")
        return df

    df['logingroup'] = df['login_site'].map(logingroup_map)

    gaspi_statut_map = get_descfic_statut_map(source, 'GASPI_SAISIE_GEN')
    df_gaspi = load_gaspi_saisie_gen(source, date_debut, date_fin)
    df['_gaspi_lk'] = df.apply(
        lambda r: r['login_site'] if gaspi_statut_map.get(r['logingroup']) == 2 else r['logingroup'], axis=1
    )
    df_gaspi['_gaspi_lk'] = df_gaspi['login_site']
    df = df.merge(df_gaspi.drop(columns=['login_site']), on=['datej', 'codss1', 'codss2', '_gaspi_lk'], how='left')
    df.drop(columns=['_gaspi_lk'], inplace=True)
    del df_gaspi
    df['eff_reel_service'] = np.where(
        df['eff_reel_service'].fillna(0) != 0,
        df['eff_reel_service'],
        df['efreel']
    )
    df['eff_prev'] = df.get('eff_prev', 0)
    df['eff_prev'] = df['eff_prev'].fillna(0)
    df['eff_prod'] = df.get('eff_prod', 0)
    df['eff_prod'] = df['eff_prod'].fillna(0)

    feuille_statut_map = get_descfic_statut_map(source, 'FEUILLE')
    df_feuille = load_feuille(source, date_debut, date_fin)
    df['_feuille_lk'] = df.apply(
        lambda r: r['login_site'] if feuille_statut_map.get(r['logingroup']) == 2 else r['logingroup'], axis=1
    )
    df_feuille['_feuille_lk'] = df_feuille['login_site']
    df = df.merge(df_feuille.drop(columns=['login_site']), on=['datej', 'codss1', 'codss2', '_feuille_lk'], how='left')
    df.drop(columns=['_feuille_lk'], inplace=True)
    del df_feuille

    typss1_statut_map = get_descfic_statut_map(source, 'TYPSS1')
    df_typss1 = load_typss1(source)
    df['_typss1_lk'] = df.apply(
        lambda r: r['login_site'] if typss1_statut_map.get(r['logingroup']) == 2 else r['logingroup'], axis=1
    )
    df_typss1_r = df_typss1.rename(columns={'login_site': '_typss1_lk'})
    df = apply_6011_consolidation(df, df_typss1_r, lk_col='_typss1_lk')
    df.drop(columns=['_typss1_lk'], errors='ignore', inplace=True)
    df['logingroup'] = df['logingroup'].fillna(df['login_site'].map(logingroup_map))
    del df_typss1, df_typss1_r

    if 'origine' not in df.columns:
        df['origine'] = np.nan

    fallback = pd.Timestamp('2015-01-01')
    df['date_import'] = fallback
    df['date_modif']  = fallback

    logger.info(f"STATS_FCJ59 : {len(df)} lignes")
    return df


def compute_stats_fcj59_detail(source, date_debut=None, date_fin=None):
    logger.info("Calcul STATS_FCJ59_DETAIL...")

    logingroup_map = load_logingroup(source)

    query = f"""
        SELECT
            CAST(e.id_effect AS BIGINT)  AS id_effect,
            CAST(e.efdate AS DATE)        AS datej,
            CAST(CAST(TRIM(e.codss1) AS INTEGER) AS VARCHAR) AS codss1,
            CAST(CAST(TRIM(e.codss2) AS INTEGER) AS VARCHAR) AS codss2,
            lpc.code_site,
            e.login_site,
            TRIM(e.codcat)                AS codcateg,
            c.libcat                      AS libcateg,
            CAST(e.efreel AS DOUBLE)      AS effreel,
            CAST(e.eftheo AS DOUBLE)      AS efftheo,
            c.famcat                      AS typecateg,
            CAST(e.credit AS DOUBLE)      AS tarifnet,
            CAST(e.typfac AS INTEGER)     AS typfac,
            e.origine,
            e.date_import,
            e.date_modif,
            CAST(e.codcli AS VARCHAR)      AS codcli,
            lpc.logingroupe                AS logingroupe
        FROM {prefix_table}effect e
        JOIN {prefix_table}login lpc ON lpc.login = e.login_site
        JOIN {prefix_table}descfic d
            ON  d.nomfic     = 'CATEG'
            AND d.login_group = lpc.logingroupe
        JOIN {prefix_table}categ c
            ON  TRIM(c.codcat) = TRIM(e.codcat)
            AND c.login_site   = CASE WHEN d.statut = 2 THEN e.login_site ELSE lpc.logingroupe END
            AND COALESCE(c.noncompte, FALSE) = FALSE
        WHERE 1=1
          {f"AND CAST(e.efdate AS DATE) >= DATE '{date_debut}'" if date_debut else ""}
          {f"AND CAST(e.efdate AS DATE) <  DATE '{date_fin}'"  if date_fin  else ""}
    """
    df = source.query(query)
    df['datej']    = pd.to_datetime(df['datej']).dt.normalize()
    df['codss1']   = df['codss1'].astype(str).str.strip()
    df['codss2']   = df['codss2'].astype(str).str.strip()
    df['codcateg'] = df['codcateg'].astype(str).str.strip()
    df['codcli']   = df['codcli'].astype(str).str.strip()
    logger.info(f"  {len(df)} lignes EFFECT valides")

    if df.empty:
        logger.info("Aucune donnée pour cette période — arrêt anticipé.")
        return df

    df_trim = load_trimestre(source)
    df['annee'] = df['datej'].apply(lambda d: str(d.year))

    df_trim['exercice'] = df_trim['exercice'].astype(str)
    trim_statut_map = get_descfic_statut_map(source, 'TRIMESTRE')
    df['_trim_lk'] = df.apply(
        lambda r: r['login_site'] if trim_statut_map.get(r['logingroupe']) == 2 else r['logingroupe'], axis=1
    )
    df_trim['_trim_lk'] = df_trim['login_site']
    df = df.merge(
        df_trim.drop(columns=['login_site']),
        left_on=['annee', '_trim_lk'],
        right_on=['exercice', '_trim_lk'],
        how='left'
    )
    df.drop(columns=['_trim_lk'], inplace=True)
    del df_trim
    df = df[(df['datdeb'].isna()) | ((df['datej'] >= df['datdeb']) & (df['datej'] <= df['datfin']))]
    df['notrim'] = df['notrim'].fillna(1).astype(int)

    df['codss2_int'] = pd.to_numeric(df['codss2'], errors='coerce').fillna(0).astype(int)
    df['k'] = 3 * (df['codss2_int'] - 1) + df['notrim']

    ntarif_statut_map = get_descfic_statut_map(source, 'NTARIF')
    df['ntarif_login'] = df.apply(
        lambda r: r['login_site'] if ntarif_statut_map.get(r['logingroupe']) == 2 else r['logingroupe'],
        axis=1
    )
    df_ntarif = load_ntarif(source)
    df_ntarif['ntarif_login'] = df_ntarif['login_site']
    df_ntarif.drop(columns=['login_site'], inplace=True)
    df = df.merge(
        df_ntarif,
        left_on=['annee', 'codss1', 'codcli', 'codcateg', 'ntarif_login'],
        right_on=['annee', 'codss1', 'codcli', 'codcat',  'ntarif_login'],
        how='left'
    )
    del df_ntarif

    def get_tarifbrut(row):
        try:
            if pd.isna(row['creditbrut']) or not row['creditbrut']:
                return 0.0
            lst = json.loads(row['creditbrut'])
            k = int(row['k'])
            if 1 <= k <= 15:
                return float(lst[k - 1])
        except (ValueError, IndexError, TypeError, json.JSONDecodeError):
            pass
        return 0.0

    df['tarifbrut'] = df.apply(get_tarifbrut, axis=1)

    df.drop(columns=['annee', 'codcli', 'codcat', 'creditbrut',
                     'exercice', 'datdeb', 'datfin', 'notrim',
                     'codss2_int', 'k', 'ntarif_login', 'login_site_y',
                     'logingroupe'], errors='ignore', inplace=True)

    df['logingroup'] = df['login_site'].map(logingroup_map)

    fallback = pd.Timestamp('2015-01-01')
    df['date_import'] = pd.to_datetime(df['date_import'], errors='coerce').fillna(fallback)
    df['date_modif']  = pd.to_datetime(df['date_modif'], errors='coerce').fillna(fallback)

    logger.info(f"STATS_FCJ59_DETAIL : {len(df)} lignes")
    return df


def compute_stats_recap_site(source, date_debut=None, date_fin=None):
    logger.info("Calcul STATS_RECAP_SITE...")

    logingroup_map = load_logingroup(source)

    logger.info("  Agrégation efreel par site/jour...")
    query_eff = f"""
        SELECT
            s.code_site,
            s.login_site,
            s.datej,
            s.codss1,
            s.codss2,
            s.efreel,
            s.logingroup
        FROM {table_fcj59} s
        WHERE 1=1
          {f"AND s.datej >= DATE '{date_debut}'" if date_debut else ""}
          {f"AND s.datej <  DATE '{date_fin}'"  if date_fin  else ""}
    """
    df_eff = source.query(query_eff)
    df_eff['datej'] = pd.to_datetime(df_eff['datej']).dt.normalize()

    df_eff['efreel_recap'] = df_eff['efreel']
    mask_cd93 = df_eff['logingroup'] == 'CD93'
    mask_cd93_excl = mask_cd93 & (
        ~df_eff['codss1'].isin(['1', '2', '8']) | (df_eff['codss2'] == '1')
    )
    df_eff.loc[mask_cd93_excl, 'efreel_recap'] = 0

    for z in range(1, 6):
        col = f'neffserv_{z}'
        df_eff[col] = np.where(df_eff['codss2'] == str(z), df_eff['efreel_recap'], 0)

    agg_eff = df_eff.groupby(['code_site', 'login_site', 'datej']).agg(
        neff_jour=('efreel_recap', 'sum'),
        neffserv_1_jour=('neffserv_1', 'sum'),
        neffserv_2_jour=('neffserv_2', 'sum'),
        neffserv_3_jour=('neffserv_3', 'sum'),
        neffserv_4_jour=('neffserv_4', 'sum'),
        neffserv_5_jour=('neffserv_5', 'sum'),
    ).reset_index()

    # nbjoursaisie : 1 si le cumul déjeuner (codss2='2') bouge ce jour-là (efreel != 0),
    # pas juste la présence d'une ligne codss2='2' (peut exister avec efreel=0, ex. sortie
    # stock seule) — confirmé empiriquement contre STATS_RECAP_SITE legacy (corrélation
    # quasi parfaite entre nbJourSaisie++ et le delta de nEffServ[2]).
    agg_eff['jour_saisie'] = (agg_eff['neffserv_2_jour'] != 0).astype(int)
    del df_eff

    logger.info("  Agrégation sorties 6011...")
    typss1_statut_map = get_descfic_statut_map(source, 'TYPSS1')
    df_typss1 = load_typss1(source)

    query_sor = f"""
        SELECT
            s.code_site,
            s.login_site,
            s.datej,
            s.codss1,
            s.codss2,
            s.totsortie
        FROM {table_fcj59} s
        WHERE 1=1
          {f"AND s.datej >= DATE '{date_debut}'" if date_debut else ""}
          {f"AND s.datej <  DATE '{date_fin}'"  if date_fin  else ""}
    """
    df_sor = source.query(query_sor)
    df_sor['datej'] = pd.to_datetime(df_sor['datej']).dt.normalize()

    df_sor['_typss1_lk'] = df_sor['login_site'].apply(
        lambda ls: ls if typss1_statut_map.get(logingroup_map.get(ls)) == 2 else logingroup_map.get(ls, ls)
    )
    df_typss1_r = df_typss1.rename(columns={'login_site': '_typss1_lk'})
    df_sor = df_sor.merge(
        df_typss1_r[['codss1', '_typss1_lk']].drop_duplicates(),
        on=['codss1', '_typss1_lk'],
        how='inner'
    )
    df_sor.drop(columns=['_typss1_lk'], inplace=True)
    del df_typss1, df_typss1_r

    for z in range(1, 6):
        df_sor[f'vtotsorserv_{z}'] = np.where(df_sor['codss2'] == str(z), df_sor['totsortie'], 0)

    agg_sor = df_sor.groupby(['code_site', 'login_site', 'datej']).agg(
        vtotsor_jour=('totsortie', 'sum'),
        vtotsorserv_1_jour=('vtotsorserv_1', 'sum'),
        vtotsorserv_2_jour=('vtotsorserv_2', 'sum'),
        vtotsorserv_3_jour=('vtotsorserv_3', 'sum'),
        vtotsorserv_4_jour=('vtotsorserv_4', 'sum'),
        vtotsorserv_5_jour=('vtotsorserv_5', 'sum'),
    ).reset_index()
    del df_sor

    logger.info("  Agrégation livraisons (typmvt=1)...")
    query_liv = f"""
        SELECT
            lpc.code_site,
            m.login_site,
            DATE(CAST(m.dtemvt AS TIMESTAMP)) AS datej,
            SUM(CAST(m.totttc AS DOUBLE))     AS vtotent_jour
        FROM {prefix_table}mvtart m
        LEFT JOIN {prefix_table}login lpc ON lpc.login = m.login_site
        WHERE CAST(m.typmvt AS INTEGER) = 1
          {f"AND DATE(CAST(m.dtemvt AS TIMESTAMP)) >= DATE '{date_debut}'" if date_debut else ""}
          {f"AND DATE(CAST(m.dtemvt AS TIMESTAMP)) <  DATE '{date_fin}'"  if date_fin  else ""}
        GROUP BY lpc.code_site, m.login_site, DATE(CAST(m.dtemvt AS TIMESTAMP))
    """
    agg_liv = source.query(query_liv)
    agg_liv['datej'] = pd.to_datetime(agg_liv['datej']).dt.normalize()

    logger.info("  Jointure et calcul cumul YTD...")
    df = agg_eff.merge(agg_sor, on=['code_site', 'login_site', 'datej'], how='left')
    df = df.merge(agg_liv, on=['code_site', 'login_site', 'datej'], how='left')
    del agg_eff, agg_sor, agg_liv

    fill_cols = ['vtotsor_jour', 'vtotent_jour'] + \
                [f'vtotsorserv_{z}_jour' for z in range(1, 6)]
    for col in fill_cols:
        df[col] = df[col].fillna(0)

    df['annee'] = df['datej'].dt.year
    df = df.sort_values(['code_site', 'login_site', 'annee', 'datej'])

    group_key = ['code_site', 'login_site', 'annee']
    cumul_cols = {
        'neff_jour': 'neff',
        'vtotsor_jour': 'vtotsor',
        'vtotent_jour': 'vtotent',
        'jour_saisie': 'nbjoursaisie',
    }
    for z in range(1, 6):
        cumul_cols[f'neffserv_{z}_jour'] = f'neffserv_{z}'
        cumul_cols[f'vtotsorserv_{z}_jour'] = f'vtotsorserv_{z}'

    for src_col, dst_col in cumul_cols.items():
        df[dst_col] = df.groupby(group_key)[src_col].cumsum()

    df['rpr'] = np.where(df['neff'] > 0, df['vtotsor'] / df['neff'], 0.0)

    df['logingroup'] = df['login_site'].map(logingroup_map)
    df.rename(columns={'datej': 'datestat'}, inplace=True)
    df['nbjoursaisie'] = df['nbjoursaisie'].astype('Int64')

    df['datestat'] = pd.to_datetime(df['datestat']).dt.normalize()

    result_cols = ['code_site', 'login_site', 'logingroup', 'datestat',
                   'neff', 'vtotent', 'vtotsor', 'rpr', 'nbjoursaisie',
                   'neffserv_1', 'neffserv_2', 'neffserv_3', 'neffserv_4', 'neffserv_5',
                   'vtotsorserv_1', 'vtotsorserv_2', 'vtotsorserv_3', 'vtotsorserv_4', 'vtotsorserv_5']
    df = df[result_cols]

    logger.info(f"STATS_RECAP_SITE : {len(df)} lignes")
    return df


def compute_stat_effect_cred_1(source, date_debut=None, date_fin=None):
    logger.info("Calcul STAT_EFFECT_CRED_1...")

    query = f"""
        SELECT
            s.datej,
            s.codss1,
            s.codss2,
            s.code_site,
            s.login_site,
            s.logingroup,
            s.efreel,
            s.totcredit,
            s.totsortie,
            s.nb_vegetarien,
            s.eff_prev,
            s.eff_reel_service,
            s.eff_prod,
            s.origine,
            s.id_animation_1,
            s.id_animation_2,
            s.id_animation_3,
            s.id_animation_4,
            s.id_animation_5,
            s.type_menu_1,
            s.type_menu_2,
            s.type_menu_3,
            s.type_menu_4,
            s.type_menu_5,
            l.nometabs,
            l.ville,
            l.id_arrondissement,
            l.id_canton,
            t.libss2,
            tt.libss1,
            d.detail_date_import,
            d.detail_date_modif,
            d.nb_eleves,
            d.nb_commensaux
        FROM {table_fcj59} s
        LEFT JOIN {prefix_table}login l
            ON s.login_site = l.login
            AND l.profil = 2
        LEFT JOIN {prefix_table}descfic dtypss2
            ON dtypss2.nomfic = 'TYPSS2' AND dtypss2.login_group = l.logingroupe
        LEFT JOIN {prefix_table}typss2 t
            ON s.codss2 = CAST(CAST(TRIM(t.codss2) AS INTEGER) AS VARCHAR)
            AND t.login_site = CASE WHEN dtypss2.statut = 2 THEN s.login_site ELSE l.logingroupe END
        LEFT JOIN {prefix_table}descfic dtypss1
            ON dtypss1.nomfic = 'TYPSS1' AND dtypss1.login_group = l.logingroupe
        LEFT JOIN {prefix_table}typss1 tt
            ON s.codss1 = CAST(CAST(TRIM(tt.codss1) AS INTEGER) AS VARCHAR)
            AND tt.login_site = CASE WHEN dtypss1.statut = 2 THEN s.login_site ELSE l.logingroupe END
        LEFT JOIN (
            SELECT datej, codss1, codss2, code_site, login_site,
                   SUM(CASE WHEN typecateg IN (0, 1) THEN effreel ELSE 0 END) AS nb_eleves,
                   SUM(CASE WHEN typecateg = 2 THEN effreel ELSE 0 END) AS nb_commensaux,
                   MAX(date_import) AS detail_date_import,
                   MAX(date_modif) AS detail_date_modif
            FROM {table_fcj59_detail}
            GROUP BY datej, codss1, codss2, code_site, login_site
        ) d
            ON  s.datej      = d.datej
            AND s.codss1     = d.codss1
            AND s.codss2     = d.codss2
            AND s.code_site  = d.code_site
            AND s.login_site = d.login_site
        WHERE 1=1
          {f"AND s.datej >= DATE '{date_debut}'" if date_debut else ""}
          {f"AND s.datej <  DATE '{date_fin}'"  if date_fin  else ""}
    """
    df = source.query(query)
    df['datej'] = pd.to_datetime(df['datej']).dt.normalize()
    df['detail_date_import'] = pd.to_datetime(df['detail_date_import'], errors='coerce')
    df['detail_date_modif'] = pd.to_datetime(df['detail_date_modif'], errors='coerce')
    logger.info(f"STAT_EFFECT_CRED_1 : {len(df)} lignes")
    return df


# ── Jours non travaillés (fériés + vacances scolaires) ─────────────────────────

def _load_dates_off(source, zone: str = 'B') -> set:
    df_feries = source.query("""
        SELECT date
        FROM db_mg6jk45h_default_dataset.default_dataset.jours_feries
        WHERE zone = 'Métropole'
    """)
    dates_off = set(pd.to_datetime(df_feries['date']).dt.date)

    df_vac = source.query(f"""
        SELECT date_debut, date_fin
        FROM db_mg6jk45h_default_dataset.default_dataset.vacances_scolaires
        WHERE zone = '{zone}'
    """)
    for _, row in df_vac.iterrows():
        current = pd.to_datetime(row['date_debut']).date()
        fin     = pd.to_datetime(row['date_fin']).date()
        while current <= fin:
            dates_off.add(current)
            current += timedelta(days=1)

    return dates_off


def _nb_jours_ouvres(d_debut, d_fin_inclu, dates_off):
    count = 0
    current = d_debut
    while current <= d_fin_inclu:
        if current.weekday() < 5 and current not in dates_off:
            count += 1
        current += timedelta(days=1)
    return count


def compute_stats_dashboard_effect(source, date_debut=None, date_fin=None, zone: str = 'B'):
    logger.info("Calcul STATS_DASHBOARD_EFFECT...")

    query = f"""
        SELECT
            logingroup,
            login_site,
            datej,
            codss2,
            efreel,
            totsortie
        FROM {table_fcj59}
        WHERE logingroup IS NOT NULL
          AND codss1 = '1'
          {f"AND datej >= DATE '{date_debut}'" if date_debut else ""}
          {f"AND datej <  DATE '{date_fin}'"  if date_fin  else ""}
    """
    df = source.query(query)
    if df.empty:
        logger.info("  Aucune donnée dans stats_fcj59, STATS_DASHBOARD_EFFECT vide.")
        return df

    df['datej']     = pd.to_datetime(df['datej']).dt.normalize()
    df['efreel']    = pd.to_numeric(df['efreel'],    errors='coerce').fillna(0.0)
    df['totsortie'] = pd.to_numeric(df['totsortie'], errors='coerce').fillna(0.0)
    df['annee']     = df['datej'].dt.year.astype(str)
    df['codss2']    = df['codss2'].astype(str).str.strip()

    grp = ['logingroup', 'annee']

    dates_off = _load_dates_off(source, zone=zone)

    logingroup_map = {row['login']: row['logingroupe']
                      for _, row in source.query(
                          f"SELECT login, logingroupe FROM {prefix_table}login"
                      ).iterrows()}
    query_recap = f"""
        SELECT login_site,
               CAST(YEAR(datestat) AS VARCHAR) AS annee,
               MAX(nbjoursaisie)               AS nbjoursaisie,
               MAX(neff)                       AS neff,
               MAX(neffserv_2)                 AS neffserv_2
        FROM {table_recap_site}
        WHERE 1=1
          {f"AND datestat >= DATE '{date_debut}'" if date_debut else ""}
          {f"AND datestat <  DATE '{date_fin}'"  if date_fin  else ""}
        GROUP BY login_site, YEAR(datestat)
    """
    df_recap = source.query(query_recap)
    df_recap['logingroup'] = df_recap['login_site'].map(logingroup_map)
    for col in ['nbjoursaisie', 'neff', 'neffserv_2']:
        df_recap[col] = pd.to_numeric(df_recap[col], errors='coerce').fillna(0)
    sum_jours_recap = (
        df_recap.groupby(grp)['nbjoursaisie']
        .sum()
        .reset_index(name='sum_jours')
    )
    sum_neff_recap = df_recap.groupby(grp)['neff'].sum().reset_index(name='neff_total')
    sum_neff_dej_recap = df_recap.groupby(grp)['neffserv_2'].sum().reset_index(name='neff_dej_total')
    del df_recap

    def _aggregate(df_src: pd.DataFrame, service_name: str) -> pd.DataFrame:
        if df_src.empty:
            return pd.DataFrame(columns=grp + ['service', 'repas_servis',
                                                'repas_par_jour', 'nb_sites', 'prix_revient',
                                                'repas_par_jour2'])

        agg = df_src.groupby(grp, as_index=False).agg(
            totsortie_sum =('totsortie',  'sum'),
            nb_sites      =('login_site', 'nunique'),
        )

        df_actif = df_src[df_src['efreel'] > 0]

        sum_jours = sum_jours_recap
        neff_for_jour = (
            sum_neff_dej_recap.rename(columns={'neff_dej_total': 'neff_recap'})
            if service_name == 'dejeuner'
            else sum_neff_recap.rename(columns={'neff_total': 'neff_recap'})
        )

        date_range_df = (
            df_actif
            .groupby(grp)['datej']
            .agg(d_min='min', d_max='max')
            .reset_index()
        )
        if date_range_df.empty:
            # .apply(axis=1) sur un DataFrame à 0 ligne renvoie un DataFrame au lieu
            # d'une Series ("Cannot set a DataFrame with multiple columns...").
            # Arrive quand efreel=0 partout (ex: année avec seulement des mouvements
            # MVTART, aucune donnée EFFECT réelle).
            date_range_df['nb_jours_ouvres'] = pd.Series(dtype='int64')
        else:
            date_range_df['nb_jours_ouvres'] = date_range_df.apply(
                lambda r: _nb_jours_ouvres(r['d_min'].date(), r['d_max'].date(), dates_off),
                axis=1
            )

        agg = agg.merge(sum_jours, on=grp, how='left')
        agg = agg.merge(neff_for_jour, on=grp, how='left')
        agg = agg.merge(date_range_df[grp + ['nb_jours_ouvres']], on=grp, how='left')
        agg['sum_jours']       = agg['sum_jours'].fillna(0)
        agg['neff_recap']      = agg['neff_recap'].fillna(0)
        agg['nb_jours_ouvres'] = agg['nb_jours_ouvres'].fillna(0)

        agg['repas_servis'] = agg['neff_recap']

        agg['repas_par_jour'] = np.where(
            agg['sum_jours'] > 0,
            agg['neff_recap'] / agg['sum_jours'],
            0.0
        )
        agg['repas_par_jour2'] = np.where(
            (agg['nb_jours_ouvres'] > 0) & (agg['nb_sites'] > 0),
            agg['repas_servis'] / agg['nb_sites'] / agg['nb_jours_ouvres'],
            0.0
        )
        agg['prix_revient'] = np.where(
            agg['repas_servis'] > 0,
            agg['totsortie_sum'] / agg['repas_servis'],
            0.0
        )
        agg['service'] = service_name
        agg.drop(columns=['totsortie_sum', 'sum_jours', 'neff_recap', 'nb_jours_ouvres'], inplace=True)
        return agg

    df_dejeuner = _aggregate(df[df['codss2'] == '2'].copy(), 'dejeuner')
    df_journee  = _aggregate(df.copy(),                       'journee')

    result = pd.concat([df_dejeuner, df_journee], ignore_index=True)

    result = result[['logingroup', 'annee', 'service',
                     'repas_servis', 'repas_par_jour', 'nb_sites', 'prix_revient',
                     'repas_par_jour2']]

    logger.info(f"STATS_DASHBOARD_EFFECT : {len(result)} lignes "
                f"({len(df_dejeuner)} déjeuner + {len(df_journee)} journée)")
    return result


# =================================================================================
# LIV — CTE enrichie, commune aux 5 tables (identique à stats_live.py, colonne
# 'jour' incluse pour STATS_LIV_EGALIM_JOUR)
# =================================================================================

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

            LPAD(CAST(year(date_trunc('week', DATE(CAST(m.dtemvt AS TIMESTAMP)))
                          + interval '3' day) AS VARCHAR), 4, '0')
                AS annee,

            LPAD(CAST(week(DATE(CAST(m.dtemvt AS TIMESTAMP))) AS VARCHAR), 2, '0')
                AS semaine,

            LPAD(CAST(month(DATE(CAST(m.dtemvt AS TIMESTAMP))) AS VARCHAR), 2, '0')
                AS mois,

            DATE(CAST(m.dtemvt AS TIMESTAMP)) AS jour,

            TRIM(a.codfamart) AS famille_article,
            TRIM(a.sfaart)    AS sous_famille_article,
            TRIM(COALESCE(fam.libfamart, '')) AS lib_famille_article,
            TRIM(COALESCE(sfa.libsfaart, '')) AS lib_sous_famille_article,

            TRIM(COALESCE(f.libfou, ''))           AS libfou,
            COALESCE(CAST(m.f_ocleunik AS INTEGER), 0) AS idfou,

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

            CASE
                WHEN COALESCE(CAST(m.id_origine  AS INTEGER), 0) != 0
                    THEN CAST(m.id_origine AS INTEGER)
                ELSE COALESCE(CAST(da.id_origine AS INTEGER), 0)
            END AS id_origine,

            CAST(m.qtef      AS DOUBLE)
                * CAST(m.puf    AS DOUBLE)
                * (1.0 + CAST(m.taux_tva AS DOUBLE) / 100.0) AS montant,

            CAST(m.qtef AS DOUBLE) * CAST(m.puf AS DOUBLE) AS montantht,

            CASE
                WHEN CAST(a.usart_vers_ufam AS DOUBLE) = 0.0
                THEN CASE
                    WHEN TRIM(a.usart) = 'KG' THEN CAST(m.qteusart AS DOUBLE)
                    ELSE 0.0
                END
                ELSE CAST(m.qteusart AS DOUBLE) * CAST(a.usart_vers_ufam AS DOUBLE)
            END AS qte

        FROM {p}mvtart m

        LEFT JOIN {p}login lpc
            ON  lpc.login = m.login_site

        LEFT JOIN {p}descfic dart
            ON  dart.nomfic     = 'ARTICLE'
            AND dart.login_group = lpc.logingroupe

        JOIN {p}article a
            ON  a.arcleunik  = m.arcleunik
            AND a.login_site = CASE WHEN dart.statut = 2 THEN m.login_site ELSE lpc.logingroupe END

        LEFT JOIN {p}descfic dmd
            ON  dmd.nomfic     = 'MVTART_DET'
            AND dmd.login_group = lpc.logingroupe

        LEFT JOIN (
            SELECT DISTINCT mvcleunik, login_site
            FROM {p}mvtart_det
        ) md
            ON  md.mvcleunik  = m.mvcleunik
            AND md.login_site = CASE WHEN dmd.statut = 2 THEN m.login_site ELSE lpc.logingroupe END

        LEFT JOIN {p}descfic dda
            ON  dda.nomfic     = 'DETAILARTICLE'
            AND dda.login_group = lpc.logingroupe

        LEFT JOIN {p}detail_article da
            ON  da.codart    = a.codart
            AND da.login_site = CASE WHEN dda.statut = 2 THEN m.login_site ELSE lpc.logingroupe END

        LEFT JOIN {p}descfic dfourn
            ON  dfourn.nomfic      = 'FOURN'
            AND dfourn.login_group = lpc.logingroupe

        LEFT JOIN {p}fourn f
            ON  f.f_ocleunik = m.f_ocleunik
            AND f.login_site = CASE WHEN dfourn.statut = 2 THEN m.login_site ELSE lpc.logingroupe END

        LEFT JOIN {p}descfic dfam
            ON  dfam.nomfic     = 'FAMART'
            AND dfam.login_group = lpc.logingroupe
        LEFT JOIN {p}famart fam
            ON  TRIM(fam.codfamart) = TRIM(a.codfamart)
            AND fam.login_site      = CASE WHEN dfam.statut = 2 THEN m.login_site ELSE lpc.logingroupe END

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


# ── LIV — calculs par table (identique à stats_live.py) ────────────────────────

def compute_stats_liv59(source, date_debut=None, date_fin=None) -> pd.DataFrame:
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


def compute_stats_liv(source, date_debut=None, date_fin=None) -> pd.DataFrame:
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


def compute_stats_liv_mois(source, date_debut=None, date_fin=None) -> pd.DataFrame:
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


def compute_stats_liv_annee(source, date_debut=None, date_fin=None) -> pd.DataFrame:
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


def compute_stats_liv_egalim_jour(source, date_debut=None, date_fin=None) -> pd.DataFrame:
    logger.info("Calcul STATS_LIV_EGALIM_JOUR...")
    query = _enriched_cte(prefix_table, date_debut, date_fin) + """
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

    for pct_prefix in ('local', 'bio', 'egalim', 'bio_local'):
        df[f'{pct_prefix}_pct'] = (
            df[f'{pct_prefix}_valeur'] / df['montant'].replace(0, float('nan')) * 100
        ).fillna(0).round(2)

    logger.info(f"  {len(df)} lignes STATS_LIV_EGALIM_JOUR")
    return df


# ── LIV — agrégats sur tout l'historique (exécutés une seule fois, hors boucle) ─

def compute_stats_liv_egalim(source) -> pd.DataFrame:
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

    for col in ['montant', 'montantht', 'qte', 'local_valeur', 'bio_valeur', 'egalim_valeur', 'bio_local_valeur']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    for pct_prefix in ('local', 'bio', 'egalim', 'bio_local'):
        df[f'{pct_prefix}_pct'] = (
            df[f'{pct_prefix}_valeur'] / df['montant'].replace(0, float('nan')) * 100
        ).fillna(0).round(2)

    logger.info(f"  {len(df)} lignes STATS_LIV_EGALIM")
    return df


def compute_stats_dashboard(source) -> pd.DataFrame:
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
    FROM {table_liv_annee} s
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


# =================================================================================
# Orchestration — une année
# =================================================================================

def _run_fcj_year(source, date_debut: str, date_fin: str, annee: str, zone: str) -> None:
    logger.info(f"--- FCJ {annee} ---")

    df_fcj59 = compute_stats_fcj59(source, date_debut, date_fin)
    _replace_year(source, table_fcj59, f"datej >= DATE '{date_debut}' AND datej < DATE '{date_fin}'", df_fcj59)
    del df_fcj59

    df_detail = compute_stats_fcj59_detail(source, date_debut, date_fin)
    _replace_year(source, table_fcj59_detail, f"datej >= DATE '{date_debut}' AND datej < DATE '{date_fin}'", df_detail)
    del df_detail

    df_recap = compute_stats_recap_site(source, date_debut, date_fin)
    _replace_year(source, table_recap_site, f"datestat >= DATE '{date_debut}' AND datestat < DATE '{date_fin}'", df_recap)
    del df_recap

    df_cred = compute_stat_effect_cred_1(source, date_debut, date_fin)
    _replace_year(source, table_effect_cred, f"datej >= DATE '{date_debut}' AND datej < DATE '{date_fin}'", df_cred)
    del df_cred

    df_dashboard = compute_stats_dashboard_effect(source, date_debut, date_fin, zone=zone)
    _replace_year(source, table_dashboard_effect, f"annee = '{annee}'", df_dashboard)
    del df_dashboard

    _release_memory()


def _run_liv_year(source, date_debut: str, date_fin: str, annee: str, now: datetime) -> None:
    logger.info(f"--- LIV {annee} ---")

    df = compute_stats_liv59(source, date_debut, date_fin)
    if not df.empty:
        df['date_import'] = now
        df['date_modif']  = now
    _replace_year(source, table_liv59, f"annee = '{annee}'", df)
    del df

    df = compute_stats_liv(source, date_debut, date_fin)
    if not df.empty:
        df['date_import'] = now
        df['date_modif']  = now
    _replace_year(source, table_liv, f"annee = '{annee}'", df)
    del df

    df = compute_stats_liv_mois(source, date_debut, date_fin)
    if not df.empty:
        df['date_import'] = now
        df['date_modif']  = now
    _replace_year(source, table_liv_mois, f"annee = '{annee}'", df)
    del df

    df = compute_stats_liv_annee(source, date_debut, date_fin)
    if not df.empty:
        df['date_import'] = now
        df['date_modif']  = now
    _replace_year(source, table_liv_annee, f"annee = '{annee}'", df)
    del df

    df = compute_stats_liv_egalim_jour(source, date_debut, date_fin)
    if not df.empty:
        df['date_import'] = now
        df['date_modif']  = now
    _replace_year(source, table_liv_egalim_jour, f"jour >= DATE '{date_debut}' AND jour < DATE '{date_fin}'", df)
    del df

    _release_memory()


def _run_liv_final(source, now: datetime) -> None:
    """Agrégats calculés sur tout l'historique — exécutés une seule fois, après la boucle."""
    logger.info("--- LIV — agrégats full-history (egalim, dashboard) ---")

    df = compute_stats_liv_egalim(source)
    if not df.empty:
        df['date_import'] = now
        df['date_modif']  = now
    _replace_all(source, table_liv_egalim, df)
    del df
    _release_memory()

    df = compute_stats_dashboard(source)
    _replace_all(source, table_stats_dashboard, df)
    del df
    _release_memory()


# =================================================================================
# Point d'entrée
# =================================================================================

def run_stats_reporting_job(
    zone_scolaire: str = 'B',
    annee_debut: int = None,
    annee_fin: int = None,
    annee_range: int = None,
) -> None:
    logger.info("START - run_stats_reporting_job")

    spark = SparkSession.builder.appName("stats_reporting_spark").getOrCreate()
    logger.info(f"Spark Version: {spark.version}")

    source = connect(dataset_cible)

    if annee_debut is None or annee_fin is None:
        annee_min, annee_max = _run_with_retry(
            "détection plage d'années", lambda: _detect_year_range(source),
            max_retries=2, retry_delay=30,
        )
        if annee_range:
            # ANNEE_RANGE=X : ne traite que les X dernières années (borné par le
            # minimum réel détecté si X dépasse la profondeur d'historique).
            annee_debut = max(annee_min, annee_max - annee_range + 1)
            annee_fin = annee_max
            logger.info(f"ANNEE_RANGE={annee_range} — traitement des {annee_range} dernières années : {annee_debut} → {annee_fin}")
        else:
            annee_debut, annee_fin = annee_min, annee_max
            logger.info(f"Plage d'années détectée automatiquement : {annee_debut} → {annee_fin}")
    else:
        logger.info(f"Plage d'années fournie : {annee_debut} → {annee_fin}")

    now = datetime.now()

    for year in range(annee_debut, annee_fin + 1):
        date_debut, date_fin = f"{year}-01-01", f"{year + 1}-01-01"
        annee_str = str(year)
        logger.info(f"=== Année {annee_str} ({date_debut} → {date_fin}) ===")

        # Reconnexion à chaque année : une connexion PolarData/Trino réutilisée sur
        # des dizaines de requêtes successives peut accumuler des buffers internes
        # au fil d'un job long multi-années — en repartir à zéro à chaque itération
        # limite ce risque d'accumulation mémoire indépendamment du garbage collector.
        source = connect(dataset_cible)

        # Chaque année est ré-essayée en entier en cas d'erreur transitoire Trino/Iceberg
        # (delete+insert par année étant idempotent, un retry complet est sûr).
        _run_with_retry(
            f"FCJ année {annee_str}",
            lambda dd=date_debut, df=date_fin, a=annee_str, s=source: _run_fcj_year(s, dd, df, a, zone_scolaire),
            max_retries=2, retry_delay=30,
        )
        _run_with_retry(
            f"LIV année {annee_str}",
            lambda dd=date_debut, df=date_fin, a=annee_str, s=source: _run_liv_year(s, dd, df, a, now),
            max_retries=2, retry_delay=30,
        )
        del source
        _release_memory()

    source = connect(dataset_cible)
    _run_with_retry(
        "agrégats full-history", lambda: _run_liv_final(source, now),
        max_retries=2, retry_delay=30,
    )

    del source
    spark.stop()
    logger.info("END - run_stats_reporting_job")


def customfunc(event):
    """Point d'entrée ForePaaS (custom action)."""
    zone = PARAMS.get('ZONE_SCOLAIRE', 'B')
    annee_debut = PARAMS.get('ANNEE_DEBUT')
    annee_fin = PARAMS.get('ANNEE_FIN')
    annee_range = PARAMS.get('ANNEE_RANGE')
    run_stats_reporting_job(
        zone_scolaire=zone,
        annee_debut=int(annee_debut) if annee_debut else None,
        annee_fin=int(annee_fin) if annee_fin else None,
        annee_range=int(annee_range) if annee_range else None,
    )


if __name__ == "__main__":
    customfunc(None)

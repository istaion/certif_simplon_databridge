import json
import logging
import csv
from datetime import timedelta

import numpy as np
import pandas as pd

from data_process.db.trino_client import TrinoClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chargement des tables sources
# ---------------------------------------------------------------------------

def load_effect_agg(db: TrinoClient, prefix: str, date_debut=None, date_fin=None):
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
        FROM {prefix}effect e
        JOIN {prefix}login lpc ON lpc.login = e.login_site
        JOIN {prefix}descfic d
            ON  d.nomfic     = 'CATEG'
            AND d.login_group = lpc.logingroupe
        JOIN {prefix}categ c
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
    df = db.query_as_dataframe(query)
    df['datej']  = pd.to_datetime(df['datej']).dt.normalize()
    df['codss1'] = df['codss1'].astype(str).str.strip()
    df['codss2'] = df['codss2'].astype(str).str.strip()
    logger.info(f"  {len(df)} groupes depuis EFFECT")
    return df


def load_mvtart_agg(db: TrinoClient, prefix: str, date_debut=None, date_fin=None):
    logger.info("Agrégation MVTART (typmvt=2)...")
    query = f"""
        SELECT
            DATE(CAST(m.dtemvt AS TIMESTAMP)) AS datej,
            CAST(CAST(TRIM(m.codss1) AS INTEGER) AS VARCHAR) AS codss1,
            CAST(CAST(TRIM(m.codss2) AS INTEGER) AS VARCHAR) AS codss2,
            lpc.code_site,
            m.login_site,
            SUM(CAST(m.totttc AS DOUBLE))     AS totsortie
        FROM {prefix}mvtart m
        LEFT JOIN {prefix}login lpc ON lpc.login = m.login_site
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
    df = db.query_as_dataframe(query)
    df['datej']  = pd.to_datetime(df['datej']).dt.normalize()
    df['codss1'] = df['codss1'].astype(str).str.strip()
    df['codss2'] = df['codss2'].astype(str).str.strip()
    logger.info(f"  {len(df)} groupes depuis MVTART")
    return df


def load_gaspi_saisie_gen(db: TrinoClient, prefix: str, date_debut=None, date_fin=None):
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
        FROM {prefix}gaspi_saisie_gen g
        WHERE 1=1
          {f"AND CAST(g.datej AS DATE) >= DATE '{date_debut}'" if date_debut else ""}
          {f"AND CAST(g.datej AS DATE) <  DATE '{date_fin}'"  if date_fin  else ""}
        GROUP BY
            CAST(g.datej AS DATE),
            TRIM(g.codss1),
            TRIM(g.codss2),
            g.login_site
    """
    df = db.query_as_dataframe(query)
    df['datej']  = pd.to_datetime(df['datej']).dt.normalize()
    df['codss1'] = df['codss1'].astype(str).str.strip()
    df['codss2'] = df['codss2'].astype(str).str.strip()
    logger.info(f"  {len(df)} lignes depuis GASPI_SAISIE_GEN")
    return df


def load_feuille(db: TrinoClient, prefix: str, date_debut=None, date_fin=None):
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
        FROM {prefix}feuille f
        WHERE 1=1
          {f"AND CAST(f.efdate AS DATE) >= DATE '{date_debut}'" if date_debut else ""}
          {f"AND CAST(f.efdate AS DATE) <  DATE '{date_fin}'"  if date_fin  else ""}
    """
    df = db.query_as_dataframe(query)
    df['datej']  = pd.to_datetime(df['datej']).dt.normalize()
    df['codss1'] = df['codss1'].astype(str).str.strip()
    df['codss2'] = df['codss2'].astype(str).str.strip()
    df = df.drop_duplicates(subset=['datej', 'codss1', 'codss2', 'login_site'])

    def expand_json_list(series, prefix_col, n=5, dtype=None):
        def parse(val):
            try:
                lst = json.loads(str(val).replace('False', 'false').replace('True', 'true'))
                return lst[:n] + [None] * max(0, n - len(lst))
            except (ValueError, TypeError, json.JSONDecodeError):
                return [None] * n
        expanded = pd.DataFrame(series.apply(parse).tolist(),
                                columns=[f"{prefix_col}_{i+1}" for i in range(n)],
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


def load_logingroup(db: TrinoClient, prefix: str):
    logger.info("Chargement logingroup depuis login...")
    query = f"""
        SELECT login, logingroupe AS logingroup
        FROM {prefix}login
        WHERE COALESCE(CAST(fictif AS BOOLEAN), FALSE) = FALSE
    """
    df = db.query_as_dataframe(query)
    logger.info(f"  {len(df)} logins chargés")
    return df.set_index('login')['logingroup'].to_dict()


def get_descfic_statut_map(db: TrinoClient, prefix: str, nomfic: str) -> dict:
    df = db.query_as_dataframe(
        f"SELECT login_group, statut FROM {prefix}descfic WHERE UPPER(nomfic) = '{nomfic.upper()}'"
    )
    if isinstance(df, pd.DataFrame) and not df.empty:
        return dict(zip(df["login_group"], df["statut"]))
    return {}


def load_typss1(db: TrinoClient, prefix: str):
    logger.info("Chargement TYPSS1...")
    query = f"""
        SELECT
            CAST(CAST(TRIM(t.codss1) AS INTEGER) AS VARCHAR) AS codss1,
            TRIM(t.codcpt) AS codcpt,
            t.login_site
        FROM {prefix}typss1 t
    """
    df = db.query_as_dataframe(query)
    df['codss1'] = df['codss1'].astype(str).str.strip()
    df['codcpt'] = df['codcpt'].astype(str).str.strip()
    df = df[df['codcpt'] == '6011']
    logger.info(f"  {len(df)} lignes codcpt='6011' depuis TYPSS1")
    return df


def load_ntarif(db: TrinoClient, prefix: str):
    logger.info("Chargement NTARIF...")
    query = f"""
        SELECT
            CAST(n.exercice AS VARCHAR)  AS annee,
            CAST(CAST(TRIM(n.prestation) AS INTEGER) AS VARCHAR) AS codss1,
            CAST(n.codcli AS VARCHAR)  AS codcli,
            TRIM(n.codcat)            AS codcat,
            n.login_site,
            n.creditbrut
        FROM {prefix}ntarif n
    """
    df = db.query_as_dataframe(query)
    df['codss1'] = df['codss1'].astype(str).str.strip()
    df['codcli'] = df['codcli'].astype(str).str.strip()
    df['codcat'] = df['codcat'].astype(str).str.strip()
    logger.info(f"  {len(df)} lignes depuis NTARIF")
    return df


def load_trimestre(db: TrinoClient, prefix: str):
    logger.info("Chargement TRIMESTRE...")
    query = f"""
        SELECT
            CAST(t.exercice AS VARCHAR)  AS exercice,
            CAST(t.datdeb AS DATE)       AS datdeb,
            CAST(t.datfin AS DATE)       AS datfin,
            CAST(t.notrim AS INTEGER)    AS notrim,
            t.login_site
        FROM {prefix}trimestre t
    """
    df = db.query_as_dataframe(query)
    df['datdeb'] = pd.to_datetime(df['datdeb']).dt.normalize()
    df['datfin'] = pd.to_datetime(df['datfin']).dt.normalize()
    logger.info(f"  {len(df)} lignes depuis TRIMESTRE")
    return df


# ---------------------------------------------------------------------------
# Logique métier
# ---------------------------------------------------------------------------

def apply_6011_consolidation(df_fcj59, df_typss1, lk_col: str = 'login_site'):
    logger.info("Consolidation 6011...")
    logger.info(f"  df_fcj59 entrant : {len(df_fcj59)} lignes, lk_col={lk_col!r}")
    logger.info(f"  df_typss1 : {len(df_typss1)} lignes, codss1 uniques : {sorted(df_typss1['codss1'].unique().tolist())}")
    df = df_fcj59.merge(
        df_typss1[['codss1', 'codcpt', lk_col]],
        on=['codss1', lk_col],
        how='left'
    )
    df['codcpt'] = df['codcpt'].fillna('').str.strip()

    n_matched = (df['codcpt'] == '6011').sum()
    logger.info(f"  Après merge TYPSS1 : {len(df)} lignes, dont {n_matched} avec codcpt='6011'")

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
    logger.info(f"  Clés distinctes dans df_6011_agg : {len(df_6011_agg)}")

    mask_01 = df['codss1'].str.strip() == '1'
    df_01 = df[mask_01].copy()
    df_non_01 = df[~mask_01].copy()
    logger.info(f"  Lignes codss1='1' existantes : {len(df_01)}")

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
        logger.info(f"  {len(df_new_01)} nouvelles lignes codss1='1' créées (sample datej/site : {df_new_01[['datej','code_site','codss2']].head(3).to_dict('records')})")

    parts = [df_non_01, df_01]
    if not df_new_01.empty:
        parts.append(df_new_01)
    result = pd.concat(parts, ignore_index=True)
    logger.info(f"  Résultat : {len(result)} lignes après consolidation 6011")
    return result


# ---------------------------------------------------------------------------
# Calcul STATS_FCJ59
# ---------------------------------------------------------------------------

def compute_stats_fcj59(db: TrinoClient, prefix: str, date_debut=None, date_fin=None):
    join_keys = ['datej', 'codss1', 'codss2', 'code_site', 'login_site']

    logingroup_map = load_logingroup(db, prefix)

    df_effect  = load_effect_agg(db, prefix, date_debut, date_fin)
    df_mvtart  = load_mvtart_agg(db, prefix, date_debut, date_fin)

    df = pd.merge(df_effect, df_mvtart, on=join_keys, how='outer')
    df['efreel']    = df['efreel'].fillna(0)
    df['totcredit'] = df['totcredit'].fillna(0)
    df['totsortie'] = df['totsortie'].fillna(0)

    df = df[(df['efreel'] != 0) | (df['totcredit'] != 0) | (df['totsortie'] != 0)].copy()
    logger.info(f"Après FULL JOIN + filtre : {len(df)} lignes")

    df['logingroup'] = df['login_site'].map(logingroup_map)

    gaspi_statut_map = get_descfic_statut_map(db, prefix, 'GASPI_SAISIE_GEN')
    df_gaspi = load_gaspi_saisie_gen(db, prefix, date_debut, date_fin)
    df['_gaspi_lk'] = df.apply(
        lambda r: r['login_site'] if gaspi_statut_map.get(r['logingroup']) == 2 else r['logingroup'], axis=1
    )
    df_gaspi['_gaspi_lk'] = df_gaspi['login_site']
    df = df.merge(df_gaspi.drop(columns=['login_site']), on=['datej', 'codss1', 'codss2', '_gaspi_lk'], how='left')
    df.drop(columns=['_gaspi_lk'], inplace=True)
    df['eff_reel_service'] = np.where(
        df['eff_reel_service'].fillna(0) != 0,
        df['eff_reel_service'],
        df['efreel']
    )
    df['eff_prev'] = df.get('eff_prev', 0)
    df['eff_prev'] = df['eff_prev'].fillna(0)
    df['eff_prod'] = df.get('eff_prod', 0)
    df['eff_prod'] = df['eff_prod'].fillna(0)

    feuille_statut_map = get_descfic_statut_map(db, prefix, 'FEUILLE')
    df_feuille = load_feuille(db, prefix, date_debut, date_fin)
    df['_feuille_lk'] = df.apply(
        lambda r: r['login_site'] if feuille_statut_map.get(r['logingroup']) == 2 else r['logingroup'], axis=1
    )
    df_feuille['_feuille_lk'] = df_feuille['login_site']
    df = df.merge(df_feuille.drop(columns=['login_site']), on=['datej', 'codss1', 'codss2', '_feuille_lk'], how='left')
    df.drop(columns=['_feuille_lk'], inplace=True)

    typss1_statut_map = get_descfic_statut_map(db, prefix, 'TYPSS1')
    df_typss1 = load_typss1(db, prefix)
    df['_typss1_lk'] = df.apply(
        lambda r: r['login_site'] if typss1_statut_map.get(r['logingroup']) == 2 else r['logingroup'], axis=1
    )
    df_typss1_r = df_typss1.rename(columns={'login_site': '_typss1_lk'})
    df = apply_6011_consolidation(df, df_typss1_r, lk_col='_typss1_lk')
    df.drop(columns=['_typss1_lk'], errors='ignore', inplace=True)
    df['logingroup'] = df['logingroup'].fillna(df['login_site'].map(logingroup_map))

    if 'origine' not in df.columns:
        df['origine'] = np.nan

    fallback = pd.Timestamp('2015-01-01')
    df['date_import'] = fallback
    df['date_modif']  = fallback

    logger.info(f"STATS_FCJ59 : {len(df)} lignes")
    return df


# ---------------------------------------------------------------------------
# Calcul STATS_FCJ59_DETAIL
# ---------------------------------------------------------------------------

def compute_stats_fcj59_detail(db: TrinoClient, prefix: str, date_debut=None, date_fin=None):
    logger.info("Calcul STATS_FCJ59_DETAIL...")

    logingroup_map = load_logingroup(db, prefix)

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
        FROM {prefix}effect e
        JOIN {prefix}login lpc ON lpc.login = e.login_site
        JOIN {prefix}descfic d
            ON  d.nomfic     = 'CATEG'
            AND d.login_group = lpc.logingroupe
        JOIN {prefix}categ c
            ON  TRIM(c.codcat) = TRIM(e.codcat)
            AND c.login_site   = CASE WHEN d.statut = 2 THEN e.login_site ELSE lpc.logingroupe END
            AND COALESCE(c.noncompte, FALSE) = FALSE
        WHERE 1=1
          {f"AND CAST(e.efdate AS DATE) >= DATE '{date_debut}'" if date_debut else ""}
          {f"AND CAST(e.efdate AS DATE) <  DATE '{date_fin}'"  if date_fin  else ""}
    """
    df = db.query_as_dataframe(query)
    df['datej']    = pd.to_datetime(df['datej']).dt.normalize()
    df['codss1']   = df['codss1'].astype(str).str.strip()
    df['codss2']   = df['codss2'].astype(str).str.strip()
    df['codcateg'] = df['codcateg'].astype(str).str.strip()
    df['codcli']   = df['codcli'].astype(str).str.strip()
    logger.info(f"  {len(df)} lignes EFFECT valides")

    df_trim = load_trimestre(db, prefix)
    df['annee'] = df['datej'].apply(lambda d: str(d.year))

    df_trim['exercice'] = df_trim['exercice'].astype(str)
    trim_statut_map = get_descfic_statut_map(db, prefix, 'TRIMESTRE')
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
    df = df[(df['datdeb'].isna()) | ((df['datej'] >= df['datdeb']) & (df['datej'] <= df['datfin']))]
    df['notrim'] = df['notrim'].fillna(1).astype(int)

    df['codss2_int'] = pd.to_numeric(df['codss2'], errors='coerce').fillna(0).astype(int)
    df['k'] = 3 * (df['codss2_int'] - 1) + df['notrim']

    ntarif_statut_map = get_descfic_statut_map(db, prefix, 'NTARIF')
    df['ntarif_login'] = df.apply(
        lambda r: r['login_site'] if ntarif_statut_map.get(r['logingroupe']) == 2 else r['logingroupe'],
        axis=1
    )
    df_ntarif = load_ntarif(db, prefix)
    df_ntarif['ntarif_login'] = df_ntarif['login_site']
    df_ntarif.drop(columns=['login_site'], inplace=True)
    df = df.merge(
        df_ntarif,
        left_on=['annee', 'codss1', 'codcli', 'codcateg', 'ntarif_login'],
        right_on=['annee', 'codss1', 'codcli', 'codcat',  'ntarif_login'],
        how='left'
    )

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


# ---------------------------------------------------------------------------
# Calcul STATS_RECAP_SITE
# ---------------------------------------------------------------------------

def compute_stats_recap_site(db: TrinoClient, prefix: str, date_debut=None, date_fin=None):
    logger.info("Calcul STATS_RECAP_SITE...")

    table_fcj59 = f"{prefix}stats_fcj59"
    logingroup_map = load_logingroup(db, prefix)

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
    df_eff = db.query_as_dataframe(query_eff)
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

    df_eff['jour_saisie'] = np.where(df_eff['codss2'] == '2', 1, 0)

    agg_eff = df_eff.groupby(['code_site', 'login_site', 'datej']).agg(
        neff_jour=('efreel_recap', 'sum'),
        jour_saisie=('jour_saisie', 'max'),
        neffserv_1_jour=('neffserv_1', 'sum'),
        neffserv_2_jour=('neffserv_2', 'sum'),
        neffserv_3_jour=('neffserv_3', 'sum'),
        neffserv_4_jour=('neffserv_4', 'sum'),
        neffserv_5_jour=('neffserv_5', 'sum'),
    ).reset_index()

    logger.info("  Agrégation sorties 6011...")
    typss1_statut_map = get_descfic_statut_map(db, prefix, 'TYPSS1')
    df_typss1 = load_typss1(db, prefix)

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
    df_sor = db.query_as_dataframe(query_sor)
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

    logger.info("  Agrégation livraisons (typmvt=1)...")
    query_liv = f"""
        SELECT
            lpc.code_site,
            m.login_site,
            DATE(CAST(m.dtemvt AS TIMESTAMP)) AS datej,
            SUM(CAST(m.totttc AS DOUBLE))     AS vtotent_jour
        FROM {prefix}mvtart m
        LEFT JOIN {prefix}login lpc ON lpc.login = m.login_site
        WHERE CAST(m.typmvt AS INTEGER) = 1
          {f"AND DATE(CAST(m.dtemvt AS TIMESTAMP)) >= DATE '{date_debut}'" if date_debut else ""}
          {f"AND DATE(CAST(m.dtemvt AS TIMESTAMP)) <  DATE '{date_fin}'"  if date_fin  else ""}
        GROUP BY lpc.code_site, m.login_site, DATE(CAST(m.dtemvt AS TIMESTAMP))
    """
    agg_liv = db.query_as_dataframe(query_liv)
    agg_liv['datej'] = pd.to_datetime(agg_liv['datej']).dt.normalize()

    logger.info("  Jointure et calcul cumul YTD...")
    df = agg_eff.merge(agg_sor, on=['code_site', 'login_site', 'datej'], how='left')
    df = df.merge(agg_liv, on=['code_site', 'login_site', 'datej'], how='left')

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


# ---------------------------------------------------------------------------
# Calcul STAT_EFFECT_CRED_1
# ---------------------------------------------------------------------------

def compute_stat_effect_cred_1(db: TrinoClient, prefix: str, date_debut=None, date_fin=None):
    logger.info("Calcul STAT_EFFECT_CRED_1...")

    table_fcj59        = f"{prefix}stats_fcj59"
    table_fcj59_detail = f"{prefix}stats_fcj59_detail"

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
        LEFT JOIN {prefix}login l
            ON s.login_site = l.login
            AND l.profil = 2
        LEFT JOIN {prefix}descfic dtypss2
            ON dtypss2.nomfic = 'TYPSS2' AND dtypss2.login_group = l.logingroupe
        LEFT JOIN {prefix}typss2 t
            ON s.codss2 = CAST(CAST(TRIM(t.codss2) AS INTEGER) AS VARCHAR)
            AND t.login_site = CASE WHEN dtypss2.statut = 2 THEN s.login_site ELSE l.logingroupe END
        LEFT JOIN {prefix}descfic dtypss1
            ON dtypss1.nomfic = 'TYPSS1' AND dtypss1.login_group = l.logingroupe
        LEFT JOIN {prefix}typss1 tt
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
    df = db.query_as_dataframe(query)
    df['datej'] = pd.to_datetime(df['datej']).dt.normalize()
    df['detail_date_import'] = pd.to_datetime(df['detail_date_import'], errors='coerce')
    df['detail_date_modif'] = pd.to_datetime(df['detail_date_modif'], errors='coerce')
    logger.info(f"STAT_EFFECT_CRED_1 : {len(df)} lignes")
    return df


# ---------------------------------------------------------------------------
# Chargement des jours non travaillés (fériés + vacances scolaires)
# ---------------------------------------------------------------------------

def _load_dates_off(ovh_api_key: str, ovh_secret_key: str, zone: str = 'B') -> set:
    db_default = TrinoClient("default_dataset", ovh_api_key, ovh_secret_key)

    df_feries = db_default.query_as_dataframe("""
        SELECT date
        FROM jours_feries
        WHERE zone = 'Métropole'
    """)
    dates_off = set(pd.to_datetime(df_feries['date']).dt.date)

    df_vac = db_default.query_as_dataframe(f"""
        SELECT date_debut, date_fin
        FROM vacances
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


# ---------------------------------------------------------------------------
# Calcul STATS_DASHBOARD_EFFECT
# ---------------------------------------------------------------------------

def compute_stats_dashboard_effect(
    db: TrinoClient,
    prefix: str,
    ovh_api_key: str,
    ovh_secret_key: str,
    date_debut=None,
    date_fin=None,
    zone: str = 'B',
):
    logger.info("Calcul STATS_DASHBOARD_EFFECT...")

    table_fcj59 = f"{prefix}stats_fcj59"

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
    df = db.query_as_dataframe(query)
    if df.empty:
        logger.info("  Aucune donnée dans stats_fcj59, STATS_DASHBOARD_EFFECT vide.")
        return df

    df['datej']     = pd.to_datetime(df['datej']).dt.normalize()
    df['efreel']    = pd.to_numeric(df['efreel'],    errors='coerce').fillna(0.0)
    df['totsortie'] = pd.to_numeric(df['totsortie'], errors='coerce').fillna(0.0)
    df['annee']     = df['datej'].dt.year.astype(str)
    df['codss2']    = df['codss2'].astype(str).str.strip()

    grp = ['logingroup', 'annee']

    dates_off = _load_dates_off(ovh_api_key, ovh_secret_key, zone=zone)

    # Somme de nbjoursaisie par (logingroup, annee) depuis stats_recap_site — même
    # logique que le legacy (MAX cumulatif par site = total annuel des jours de saisie codss2=2)
    logingroup_map = {row['login']: row['logingroupe']
                      for _, row in db.query_as_dataframe(
                          f"SELECT login, logingroupe FROM {prefix}login"
                      ).iterrows()}
    query_recap = f"""
        SELECT login_site,
               CAST(YEAR(datestat) AS VARCHAR) AS annee,
               MAX(nbjoursaisie)               AS nbjoursaisie,
               MAX(neff)                       AS neff,
               MAX(neffserv_2)                 AS neffserv_2
        FROM {prefix}stats_recap_site
        WHERE 1=1
          {f"AND datestat >= DATE '{date_debut}'" if date_debut else ""}
          {f"AND datestat <  DATE '{date_fin}'"  if date_fin  else ""}
        GROUP BY login_site, YEAR(datestat)
    """
    df_recap = db.query_as_dataframe(query_recap)
    df_recap['logingroup'] = df_recap['login_site'].map(logingroup_map)
    for col in ['nbjoursaisie', 'neff', 'neffserv_2']:
        df_recap[col] = pd.to_numeric(df_recap[col], errors='coerce').fillna(0)
    sum_jours_recap = (
        df_recap.groupby(grp)['nbjoursaisie']
        .sum()
        .reset_index(name='sum_jours')
    )
    # Numérateurs legacy : neff (journée = tous codss1) et neffserv_2 (déjeuner = service=2 tous codss1)
    sum_neff_recap = df_recap.groupby(grp)['neff'].sum().reset_index(name='neff_total')
    sum_neff_dej_recap = df_recap.groupby(grp)['neffserv_2'].sum().reset_index(name='neff_dej_total')

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

        # sum_jours depuis stats_recap_site (nbjoursaisie cumulatif annuel), identique au legacy
        sum_jours = sum_jours_recap
        # Numérateur legacy pour repas_servis / repas_par_jour : neff tous codss1 (journée,
        # cf. recupEFF.txt "nEffAnnée+=EFFECT.efreel" sans filtre codss1) ou neffserv_2 (déjeuner)
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

        # repas_servis = neff/neffserv_2 (tous codss1, comme WebGerest), pas efreel filtré codss1='1'
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

from forepaas.dwh import connect, bulk_insert
from forepaas.core.settings import PARAMS
import ctypes
import gc
import logging
import time
import pandas as pd
import numpy as np
import json
import csv
from datetime import timedelta

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
prefix_table = PARAMS['PREFIX_TABLE']
environement = PARAMS['ENVIRONNEMENT_CLIENT']
dataset_cible = f"dwh/db_mg6jk45h_{environement}/"


def parse_annee(annee: str):
    """
    Convertit "2024" en (date_debut, date_fin) = ("2024-01-01", "2025-01-01").
    Si None ou vide, retourne (None, None) → pas de filtre.
    """
    if not annee:
        return None, None
    annee = annee.strip()
    if not annee.isdigit() or len(annee) != 4:
        raise ValueError(f"ANNEE invalide : '{annee}'. Format attendu : '2024'")
    return f"{annee}-01-01", f"{int(annee) + 1}-01-01"

table_fcj59        = f"{prefix_table}stats_fcj59"
table_fcj59_detail = f"{prefix_table}stats_fcj59_detail"
table_recap_site   = f"{prefix_table}stats_recap_site"
table_effect_cred  = f"{prefix_table}stat_effect_cred_1"
table_dashboard_effect = f"{prefix_table}stats_dashboard_effect"

primary_keys_fcj59  = ["datej", "codss1", "codss2", "code_site", "login_site"]
primary_keys_detail = ["datej", "codss1", "codss2", "code_site", "login_site", "codcateg", "id_effect"]

column_trino_fcj59 = {
    "datej": "DATE",
    "codss1": "VARCHAR",
    "codss2": "VARCHAR",
    "code_site": "BIGINT",
    "login_site": "VARCHAR",
    "logingroup": "VARCHAR",
    "efreel": "DOUBLE",
    "totcredit": "DOUBLE",
    "totsortie": "DOUBLE",
    "id_animation_1": "BIGINT",
    "id_animation_2": "BIGINT",
    "id_animation_3": "BIGINT",
    "id_animation_4": "BIGINT",
    "id_animation_5": "BIGINT",
    "type_menu_1": "BIGINT",
    "type_menu_2": "BIGINT",
    "type_menu_3": "BIGINT",
    "type_menu_4": "BIGINT",
    "type_menu_5": "BIGINT",
    "nb_vegetarien": "BIGINT",
    "eff_prev": "DOUBLE",
    "eff_reel_service": "DOUBLE",
    "eff_prod": "DOUBLE",
    "origine": "VARCHAR",
    "date_import": "TIMESTAMP",
    "date_modif": "TIMESTAMP",
}

column_trino_detail = {
    "datej": "DATE",
    "codss1": "VARCHAR",
    "codss2": "VARCHAR",
    "code_site": "BIGINT",
    "login_site": "VARCHAR",
    "logingroup": "VARCHAR",
    "id_effect": "BIGINT",
    "codcateg": "VARCHAR",
    "libcateg": "VARCHAR",
    "effreel": "DOUBLE",
    "efftheo": "DOUBLE",
    "typecateg": "BIGINT",
    "tarifnet": "DOUBLE",
    "typfac": "BIGINT",
    "tarifbrut": "DOUBLE",
    "origine": "VARCHAR",
    "date_import": "TIMESTAMP",
    "date_modif": "TIMESTAMP",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def upsert(source, table, primary_keys, df, column_types=None):
    """MERGE (upsert) Trino par batch de 200 — avec CAST explicites et retry."""
    import time
    step = 200
    max_retries = 3
    retry_delay = 30  # secondes

    df = df.copy()
    df.columns = df.columns.str.replace(".", "_").str.lower()
    total_upserted = 0
    column_types = column_types or {}

    for start in range(0, len(df), step):
        df_batch = df[start:start + step].copy()
        cols = df_batch.columns.to_list()

        # Pré-traitement colonnes DATE : marqueurs DATEVAL pour Trino DATE literals
        for col in cols:
            if column_types.get(col) == "DATE":
                df_batch[col] = pd.to_datetime(df_batch[col], errors='coerce').apply(
                    lambda x: f"DATEVAL {x.strftime('%Y-%m-%d')} DATEVAL" if pd.notna(x) else np.nan
                )

        raw_csv = "(" + df_batch.to_csv(
            header=None,
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
            quotechar="'",
            na_rep="NULL",
            date_format="TIMESTAMP %Y-%m-%d %H:%M:%S%z TIMESTAMP"
        ).replace("'NULL'", "NULL") \
         .replace("'TIMESTAMP ", "TIMESTAMP '") \
         .replace(" TIMESTAMP'", "'") \
         .replace("'DATEVAL ", "DATE '") \
         .replace(" DATEVAL'", "'") \
         .strip("\n") \
         .replace("\n", "),(") + ")"

        sql = f"""
            MERGE INTO {table}
            USING (VALUES {raw_csv}) AS tmp ({', '.join(cols)})
            ON {' AND '.join([f'{table}.{f} IS NOT DISTINCT FROM tmp.{f}' for f in primary_keys])}
            WHEN MATCHED THEN
                UPDATE SET {', '.join([
                    f'{f}=CAST(tmp.{f} AS {column_types[f]})' if f in column_types else f'{f}=tmp.{f}'
                    for f in cols if f not in primary_keys
                ])}
            WHEN NOT MATCHED THEN
                INSERT ({', '.join(cols)})
                VALUES ({', '.join([
                    f'CAST(tmp.{f} AS {column_types[f]})' if f in column_types else f'tmp.{f}'
                    for f in cols
                ])})
        """
        sql_clean = sql.replace("\n", " ").replace("\r", " ")

        batch_num = start // step + 1
        for attempt in range(1, max_retries + 1):
            try:
                result = source.query(sql_clean)
                batch_count = result.iloc[0, 0] if isinstance(result, pd.DataFrame) and not result.empty else len(df_batch)
                total_upserted += batch_count
                logger.info(f"  Batch {batch_num}: {batch_count} lignes dans {table}")
                break
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"  Batch {batch_num} échoué (tentative {attempt}/{max_retries}): {e}. Retry dans {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"  Batch {batch_num} définitivement échoué après {max_retries} tentatives: {e}")
                    raise

    logger.info(f"  Total: {total_upserted} lignes upsertées dans {table}")
    return total_upserted


# ---------------------------------------------------------------------------
# Chargement des tables sources
# ---------------------------------------------------------------------------

def load_effect_agg(source, date_debut=None, date_fin=None):
    """
    Agrège effect par (datej, codss1, codss2, code_site, login_site).
    Filtre les catégories CATEG.NONCOMPTE=False (NULL traité comme False).
    Calcule efreel et totcredit (règle TYPFAC 1/2).

    Le join categ utilise login_site si CATEG est statut 2 dans descfic (unique au site),
    ou logingroupe si statut 1 (commun au groupe, ex: CD18, CD41).
    """
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
    """
    Agrège mvtart (typmvt=2) par (datej, codss1, codss2, code_site, login_site).
    """
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
    """
    Charge gaspi_saisie_gen pour les champs effPrevu / effProd / effReelService.
    """
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
    """
    Charge feuille pour nbVegetarien / idAnimation[1-5] / typeMenu[1-5].
    """
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
    # Dédoublonnage défensif : un doublon dans feuille provoquerait des lignes dupliquées
    df = df.drop_duplicates(subset=['datej', 'codss1', 'codss2', 'login_site'])

    # id_animation et type_menu sont des JSON strings "[0, 0, 0, 0, 0]"
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
    """
    Charge le mapping login_site → logingroup depuis login.
    """
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
    """Retourne {login_group: statut} pour un nomfic donné."""
    df = source.query(
        f"SELECT login_group, statut FROM {prefix_table}descfic WHERE UPPER(nomfic) = '{nomfic.upper()}'"
    )
    if isinstance(df, pd.DataFrame) and not df.empty:
        return dict(zip(df["login_group"], df["statut"]))
    return {}


def load_typss1(source):
    """
    Charge typss1 pour la règle de consolidation "6011" (L270-289).
    TYPSS1.codcpt est un string (ex: "6011 ").
    """
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
    # Seul codcpt='6011' est utilisé dans apply_6011_consolidation.
    # Filtrer ici évite les doublons si plusieurs codcpt par (codss1, login_site).
    df = df[df['codcpt'] == '6011']
    logger.info(f"  {len(df)} lignes codcpt='6011' depuis TYPSS1")
    return df


def load_ntarif(source):
    """
    Charge ntarif pour le calcul tarifbrut.
    ntarif.creditbrut est stocké comme JSON string "[0, 0, 4.28, ...]" (15 valeurs).
    Clé : (annee, codss1, codcli, codcat, login_site).
    """
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
    """
    Charge la table TRIMESTRE pour RecupTrimestre() (L199).
    Utilisée pour indexer ntarif.creditbrut[k] où k = 3*(codss2-1) + noTrim.
    DESCFIC statut 2 — unique au site, join sur login_site.
    """
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


# ---------------------------------------------------------------------------
# Logique métier
# ---------------------------------------------------------------------------

def apply_6011_consolidation(df_fcj59, df_typss1, lk_col: str = 'login_site'):
    """
    Règle WinDev :
    Pour chaque ligne avec codss1 != '1' ET TYPSS1.codcpt = '6011',
    ajouter son totsortie à la ligne codss1='1' (même datej/codss2/code_site/login_site).
    Si la ligne '1' n'existe pas, la créer (efreel=0, totcredit=0).
    Note : codss1 est normalisé sans zéro de tête ('1' et non '01').
    lk_col : colonne de clé login à utiliser pour le merge (routing descfic).
    """
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

    # Ne pas mettre à zéro : le webgerest conserve le totsortie original sur chaque ligne.
    # Le cumul dans stats_recap_site utilise uniquement les lignes 6011 directement (inner join
    # typss1), pas la ligne codss1='1' consolidée — donc pas de double comptage.

    # Supprimer la colonne codcpt du résultat
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

    # Lignes codss1="1" existantes
    mask_01 = df['codss1'].str.strip() == '1'
    df_01 = df[mask_01].copy()
    df_non_01 = df[~mask_01].copy()
    logger.info(f"  Lignes codss1='1' existantes : {len(df_01)}")

    # Ajouter totsortie_6011 aux lignes "1" existantes
    df_01 = df_01.merge(df_6011_agg, on=keys, how='left')
    df_01['totsortie'] = df_01['totsortie'] + df_01['totsortie_6011'].fillna(0)
    df_01.drop(columns=['totsortie_6011'], inplace=True)

    # Créer les lignes "1" manquantes — anti-join robuste (évite les problèmes de type Timestamp/NaN)
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
        # Colonnes manquantes → NaN
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

def compute_stats_fcj59(source, date_debut=None, date_fin=None):
    """
    Reproduit la logique pour STATS_FCJ59.
    """
    join_keys = ['datej', 'codss1', 'codss2', 'code_site', 'login_site']

    logingroup_map = load_logingroup(source)

    df_effect  = load_effect_agg(source, date_debut, date_fin)
    df_mvtart  = load_mvtart_agg(source, date_debut, date_fin)

    # Étape 3 : JOIN
    df = pd.merge(df_effect, df_mvtart, on=join_keys, how='outer')
    del df_effect, df_mvtart
    df['efreel']    = df['efreel'].fillna(0)
    df['totcredit'] = df['totcredit'].fillna(0)
    df['totsortie'] = df['totsortie'].fillna(0)

    # Filtre : au moins une valeur non nulle (L221)
    df = df[(df['efreel'] != 0) | (df['totcredit'] != 0) | (df['totsortie'] != 0)].copy()
    logger.info(f"Après FULL JOIN + filtre : {len(df)} lignes")

    if df.empty:
        # Arrêt anticipé : évite les merges GASPI/FEUILLE/TYPSS1 (coûteux et inutiles
        # sur un DataFrame vide) et le bug de dtype (.apply() sur 0 ligne renvoie
        # float64 au lieu de object, ce qui fait échouer le merge sur les clés _lk).
        logger.info("Aucune donnée pour cette période — arrêt anticipé.")
        return df

    # logingroup (nécessaire pour le routing descfic ci-dessous)
    df['logingroup'] = df['login_site'].map(logingroup_map)

    # Lookup GASPI_SAISIE_GEN avec routing descfic
    gaspi_statut_map = get_descfic_statut_map(source, 'GASPI_SAISIE_GEN')
    df_gaspi = load_gaspi_saisie_gen(source, date_debut, date_fin)
    df['_gaspi_lk'] = df.apply(
        lambda r: r['login_site'] if gaspi_statut_map.get(r['logingroup']) == 2 else r['logingroup'], axis=1
    )
    df_gaspi['_gaspi_lk'] = df_gaspi['login_site']
    df = df.merge(df_gaspi.drop(columns=['login_site']), on=['datej', 'codss1', 'codss2', '_gaspi_lk'], how='left')
    df.drop(columns=['_gaspi_lk'], inplace=True)
    del df_gaspi
    # eff_reel_service = eff_reel_service si != 0, sinon efreel
    df['eff_reel_service'] = np.where(
        df['eff_reel_service'].fillna(0) != 0,
        df['eff_reel_service'],
        df['efreel']
    )
    df['eff_prev'] = df.get('eff_prev', 0)
    df['eff_prev'] = df['eff_prev'].fillna(0)
    df['eff_prod'] = df.get('eff_prod', 0)
    df['eff_prod'] = df['eff_prod'].fillna(0)

    # Lookup FEUILLE avec routing descfic
    feuille_statut_map = get_descfic_statut_map(source, 'FEUILLE')
    df_feuille = load_feuille(source, date_debut, date_fin)
    df['_feuille_lk'] = df.apply(
        lambda r: r['login_site'] if feuille_statut_map.get(r['logingroup']) == 2 else r['logingroup'], axis=1
    )
    df_feuille['_feuille_lk'] = df_feuille['login_site']
    df = df.merge(df_feuille.drop(columns=['login_site']), on=['datej', 'codss1', 'codss2', '_feuille_lk'], how='left')
    df.drop(columns=['_feuille_lk'], inplace=True)
    del df_feuille

    # Consolidation 6011 avec routing descfic TYPSS1
    typss1_statut_map = get_descfic_statut_map(source, 'TYPSS1')
    df_typss1 = load_typss1(source)
    df['_typss1_lk'] = df.apply(
        lambda r: r['login_site'] if typss1_statut_map.get(r['logingroup']) == 2 else r['logingroup'], axis=1
    )
    df_typss1_r = df_typss1.rename(columns={'login_site': '_typss1_lk'})
    df = apply_6011_consolidation(df, df_typss1_r, lk_col='_typss1_lk')
    df.drop(columns=['_typss1_lk'], errors='ignore', inplace=True)
    # logingroup NaN pour les nouvelles lignes codss1='1' créées par consolidation
    df['logingroup'] = df['logingroup'].fillna(df['login_site'].map(logingroup_map))
    del df_typss1, df_typss1_r

    # !!!!! Voir comment on peut récupérer l'origin, mais pas l'impression d'en avoir besoin...
    if 'origine' not in df.columns:
        df['origine'] = np.nan

    # Timestamp d'import
    fallback = pd.Timestamp('2015-01-01')
    df['date_import'] = fallback
    df['date_modif']  = fallback

    logger.info(f"STATS_FCJ59 : {len(df)} lignes")
    return df


# ---------------------------------------------------------------------------
# Calcul STATS_FCJ59_DETAIL
# ---------------------------------------------------------------------------

def compute_stats_fcj59_detail(source, date_debut=None, date_fin=None):
    """
    Reproduit la logique de pour STATS_FCJ59_DETAIL.
    Une ligne par enregistrement EFFECT dont CATEG.NONCOMPTE = False.
    tarifbrut calculé via NTARIF.creditbrut[k] où k = 3*(codss2-1) + noTrim (table TRIMESTRE).
    """
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

    # Calcul noTrim via table TRIMESTRE
    # TRIMESTRE est DESCFIC statut 2 — join sur (login_site, exercice, datdeb<=datej<=datfin)
    df_trim = load_trimestre(source)
    df['annee'] = df['datej'].apply(lambda d: str(d.year))

    # merge via routing descfic TRIMESTRE
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

    # Calcul index k = 3*(int(codss2)-1) + noTrim  (1-indexé, bornes [1,15])
    df['codss2_int'] = pd.to_numeric(df['codss2'], errors='coerce').fillna(0).astype(int)
    df['k'] = 3 * (df['codss2_int'] - 1) + df['notrim']

    # Lookup NTARIF pour creditbrut
    # Si ntarif statut != 2 (commun au groupe), login_site stocké = logingroupe
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

    # Extraire tarifbrut depuis JSON string
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

    # Nettoyage colonnes intermédiaires
    df.drop(columns=['annee', 'codcli', 'codcat', 'creditbrut',
                     'exercice', 'datdeb', 'datfin', 'notrim',
                     'codss2_int', 'k', 'ntarif_login', 'login_site_y',
                     'logingroupe'], errors='ignore', inplace=True)

    # logingroup
    df['logingroup'] = df['login_site'].map(logingroup_map)

    # Timestamp si absent
    fallback = pd.Timestamp('2015-01-01')
    df['date_import'] = pd.to_datetime(df['date_import'], errors='coerce').fillna(fallback)
    df['date_modif']  = pd.to_datetime(df['date_modif'], errors='coerce').fillna(fallback)

    logger.info(f"STATS_FCJ59_DETAIL : {len(df)} lignes")
    return df


# ---------------------------------------------------------------------------
# Calcul STATS_RECAP_SITE
# ---------------------------------------------------------------------------

def compute_stats_recap_site(source, date_debut=None, date_fin=None):
    """
    Reproduit la logique de STATS_RECAP_SITE.
    Récapitulatif annuel par (code_site, login_site, datestat) avec cumul YTD.

    - neff : cumul YTD de efreel (noncompte=False), règle CD93
    - vtotsor : cumul YTD de totsortie 6011 (typmvt=2, codcpt='6011')
    - vtotent : cumul YTD de livraisons (typmvt=1, unique par jour)
    - rpr : vtotsor / neff
    - nbjoursaisie : cumul YTD du nb de jours où codss2='2' a des données
    - neffserv_1..5 : cumul YTD efreel par codss2 (1-5)
    - vtotsorserv_1..5 : cumul YTD totsortie 6011 par codss2 (1-5)
    """
    logger.info("Calcul STATS_RECAP_SITE...")

    logingroup_map = load_logingroup(source)

    # --- 1. Agréger efreel par (code_site, login_site, datej) depuis stats_fcj59 ---
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

    if df_eff.empty:
        # Arrêt anticipé : évite les .apply()/merges sur DataFrame vide plus bas
        # (dtype float64 au lieu de object sur 0 ligne → échec du merge).
        logger.info("Aucune donnée dans STATS_FCJ59 pour cette période — arrêt anticipé.")
        return df_eff

    # Règle CD93 : pour logingroup='CD93', neff ne compte que codss1 in (1,2,8) et codss2 != 1
    df_eff['efreel_recap'] = df_eff['efreel']
    mask_cd93 = df_eff['logingroup'] == 'CD93'
    mask_cd93_excl = mask_cd93 & (
        ~df_eff['codss1'].isin(['1', '2', '8']) | (df_eff['codss2'] == '1')
    )
    df_eff.loc[mask_cd93_excl, 'efreel_recap'] = 0

    # neffserv_z : efreel par codss2 (mêmes règles CD93)
    for z in range(1, 6):
        col = f'neffserv_{z}'
        df_eff[col] = np.where(df_eff['codss2'] == str(z), df_eff['efreel_recap'], 0)

    # Agréger par (code_site, login_site, datej)
    agg_eff = df_eff.groupby(['code_site', 'login_site', 'datej']).agg(
        neff_jour=('efreel_recap', 'sum'),
        neffserv_1_jour=('neffserv_1', 'sum'),
        neffserv_2_jour=('neffserv_2', 'sum'),
        neffserv_3_jour=('neffserv_3', 'sum'),
        neffserv_4_jour=('neffserv_4', 'sum'),
        neffserv_5_jour=('neffserv_5', 'sum'),
    ).reset_index()
    del df_eff

    # nbjoursaisie : 1 si le cumul déjeuner (codss2='2') bouge ce jour-là (efreel != 0),
    # pas juste la présence d'une ligne codss2='2' (peut exister avec efreel=0, ex. sortie
    # stock seule) — confirmé empiriquement contre STATS_RECAP_SITE legacy (corrélation
    # quasi parfaite entre nbJourSaisie++ et le delta de nEffServ[2]).
    agg_eff['jour_saisie'] = (agg_eff['neffserv_2_jour'] != 0).astype(int)

    # --- 2. Agréger sorties 6011 par (code_site, login_site, datej) ---
    logger.info("  Agrégation sorties 6011...")
    typss1_statut_map = get_descfic_statut_map(source, 'TYPSS1')
    df_typss1 = load_typss1(source)  # déjà filtré codcpt='6011'

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

    # Filtrer uniquement les lignes 6011 avec routing descfic TYPSS1
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

    # vtotsorserv_z par codss2
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

    # --- 3. Agréger livraisons (typmvt=1) par (code_site, login_site, datej) ---
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

    # --- 4. Joindre les 3 agrégations ---
    logger.info("  Jointure et calcul cumul YTD...")
    df = agg_eff.merge(agg_sor, on=['code_site', 'login_site', 'datej'], how='left')
    df = df.merge(agg_liv, on=['code_site', 'login_site', 'datej'], how='left')
    del agg_eff, agg_sor, agg_liv

    # Remplir les NaN
    fill_cols = ['vtotsor_jour', 'vtotent_jour'] + \
                [f'vtotsorserv_{z}_jour' for z in range(1, 6)]
    for col in fill_cols:
        df[col] = df[col].fillna(0)

    # --- 5. Cumul YTD par (code_site, login_site, année) ---
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

    # --- 6. Calculer rPR ---
    df['rpr'] = np.where(df['neff'] > 0, df['vtotsor'] / df['neff'], 0.0)

    # --- 7. Finaliser ---
    df['logingroup'] = df['login_site'].map(logingroup_map)
    df.rename(columns={'datej': 'datestat'}, inplace=True)
    df['nbjoursaisie'] = df['nbjoursaisie'].astype('Int64')

    # Pré-traitement datestat pour bulk_insert
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

def compute_stat_effect_cred_1(source, date_debut=None, date_fin=None):
    """
    Vue enrichie de stats_fcj59 avec jointures login, typss2, typss1 et détail.
    Ajoute nometabs, ville, libss2, libss1, nb_eleves, nb_commensaux, dates détail.
    """
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


def _load_dates_off(source, zone='B'):
    """
    Charge depuis Trino les jours fériés (Métropole) et les vacances scolaires
    pour la zone donnée. Retourne un set de datetime.date à exclure.
    """
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
    """
    Nb de jours lundi-vendredi hors les dates contenues dans dates_off
    entre d_debut et d_fin_inclu (bornes incluses).
    """
    count = 0
    current = d_debut
    while current <= d_fin_inclu:
        if current.weekday() < 5 and current not in dates_off:
            count += 1
        current += timedelta(days=1)
    return count


def compute_stats_dashboard_effect(source, date_debut=None, date_fin=None, zone='B'):
    """
    Agrégation par (logingroup, annee, service) depuis stats_fcj59.

    service :
      - 'dejeuner' → uniquement codss2 = '2'
      - 'journee'  → tous les codss2 (déjeuner inclus)

    Colonnes produites :
      repas_servis   = neff / neffserv_2 cumulé (stats_recap_site), tous codss1
                       (cf. recupEFF.txt "nEffAnnée+=EFFECT.efreel" sans filtre codss1)
      repas_par_jour = repas_servis / sum(nbjoursaisie) cumulé (stats_recap_site)
      nb_sites       = COUNT(DISTINCT login_site) ayant contribué (codss1='1')
      prix_revient   = SUM(totsortie, codss1='1' 6011-consolidé) / repas_servis
    """
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

    df['datej']    = pd.to_datetime(df['datej']).dt.normalize()
    df['efreel']   = pd.to_numeric(df['efreel'],   errors='coerce').fillna(0.0)
    df['totsortie']= pd.to_numeric(df['totsortie'], errors='coerce').fillna(0.0)
    df['annee']    = df['datej'].dt.year.astype(str)
    df['codss2']   = df['codss2'].astype(str).str.strip()

    grp = ['logingroup', 'annee']

    dates_off = _load_dates_off(source, zone=zone)

    # Somme de nbjoursaisie par (logingroup, annee) depuis stats_recap_site — même
    # logique que le legacy (MAX cumulatif par site = total annuel des jours de saisie codss2=2)
    df_login = source.query(f"SELECT login, logingroupe FROM {prefix_table}login")
    logingroup_map = {row['login']: row['logingroupe'] for _, row in df_login.iterrows()}
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
    # Numérateurs legacy : neff (journée = tous codss1) et neffserv_2 (déjeuner = service=2 tous codss1)
    sum_neff_recap = df_recap.groupby(grp)['neff'].sum().reset_index(name='neff_total')
    sum_neff_dej_recap = df_recap.groupby(grp)['neffserv_2'].sum().reset_index(name='neff_dej_total')

    def _aggregate(df_src: pd.DataFrame, service_name: str) -> pd.DataFrame:
        if df_src.empty:
            return pd.DataFrame(columns=grp + ['service', 'repas_servis',
                                                'repas_par_jour', 'nb_sites', 'prix_revient',
                                                'repas_par_jour2'])

        # Totaux bruts
        agg = df_src.groupby(grp, as_index=False).agg(
            totsortie_sum =('totsortie', 'sum'),
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

        # repas_par_jour2 : repas_servis / nb_sites / nb_jours_ouvres_ecoules
        # nb_jours_ouvres = jours lundi-vendredi hors fériés et vacances scolaires
        # sur la plage de dates réelles du groupe (zone via PARAM ZONE_SCOLAIRE)
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

    # Ordre des colonnes
    result = result[['logingroup', 'annee', 'service',
                     'repas_servis', 'repas_par_jour', 'nb_sites', 'prix_revient',
                     'repas_par_jour2']]

    logger.info(f"STATS_DASHBOARD_EFFECT : {len(result)} lignes "
                f"({len(df_dejeuner)} déjeuner + {len(df_journee)} journée)")
    return result

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


def _process_year(source, date_debut: str, date_fin: str, annee: str, zone: str) -> None:
    """Calcule et écrit les 5 tables FCJ pour une seule année (une tranche à la fois,
    delete ciblé sur cette tranche — ne touche pas aux autres années déjà en base)."""
    logger.info(f"=== Année {annee} ===")

    logger.info("  Calcul STATS_FCJ59...")
    df_fcj59 = compute_stats_fcj59(source, date_debut, date_fin)
    if not df_fcj59.empty:
        def _insert_fcj59():
            source.query(f"DELETE FROM {table_fcj59} WHERE datej >= DATE '{date_debut}' AND datej < DATE '{date_fin}'")
            bulk_insert(source, table_fcj59, df_fcj59)
        _run_with_retry(table_fcj59, _insert_fcj59)
    del df_fcj59

    logger.info("  Calcul STATS_FCJ59_DETAIL...")
    df_detail = compute_stats_fcj59_detail(source, date_debut, date_fin)
    if not df_detail.empty:
        def _insert_detail():
            source.query(f"DELETE FROM {table_fcj59_detail} WHERE datej >= DATE '{date_debut}' AND datej < DATE '{date_fin}'")
            bulk_insert(source, table_fcj59_detail, df_detail)
        _run_with_retry(table_fcj59_detail, _insert_detail)
    del df_detail

    logger.info("  Calcul STATS_RECAP_SITE...")
    df_recap = compute_stats_recap_site(source, date_debut, date_fin)
    if not df_recap.empty:
        def _insert_recap():
            source.query(f"DELETE FROM {table_recap_site} WHERE datestat >= DATE '{date_debut}' AND datestat < DATE '{date_fin}'")
            bulk_insert(source, table_recap_site, df_recap)
        _run_with_retry(table_recap_site, _insert_recap)
    del df_recap

    logger.info("  Calcul STAT_EFFECT_CRED_1...")
    df_cred = compute_stat_effect_cred_1(source, date_debut, date_fin)
    if not df_cred.empty:
        def _insert_cred():
            source.query(f"DELETE FROM {table_effect_cred} WHERE datej >= DATE '{date_debut}' AND datej < DATE '{date_fin}'")
            bulk_insert(source, table_effect_cred, df_cred)
        _run_with_retry(table_effect_cred, _insert_cred)
    del df_cred

    logger.info(f"  Calcul STATS_DASHBOARD_EFFECT (zone scolaire : {zone})...")
    df_dashboard = compute_stats_dashboard_effect(source, date_debut, date_fin, zone=zone)
    if not df_dashboard.empty:
        def _insert_dashboard():
            source.query(f"DELETE FROM {table_dashboard_effect} WHERE annee = '{annee}'")
            bulk_insert(source, table_dashboard_effect, df_dashboard)
        _run_with_retry(table_dashboard_effect, _insert_dashboard)
    del df_dashboard

    _release_memory()


def customfunc(event):
    zone = PARAMS.get('ZONE_SCOLAIRE', 'B')
    annee = PARAMS.get('ANNEE', None)
    annee_range = PARAMS.get('ANNEE_RANGE', None)

    if annee:
        # Un seul run = une seule année (comportement historique, inchangé).
        source = connect(dataset_cible)
        date_debut, date_fin = parse_annee(annee)
        logger.info(f"Filtre année : {annee} ({date_debut} → {date_fin})")
        _run_with_retry(
            f"année {annee}", lambda: _process_year(source, date_debut, date_fin, annee, zone),
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
                lambda dd=date_debut, df=date_fin, y=year, s=source: _process_year(s, dd, df, str(y), zone),
                max_retries=2, retry_delay=30,
            )
            del source
            _release_memory()

    logger.info("Job stats_fcj59 terminé.")

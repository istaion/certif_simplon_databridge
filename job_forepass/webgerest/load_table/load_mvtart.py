from forepaas.dwh import connect, bulk_insert 
from forepaas.core.settings import PARAMS
import logging
import requests
import pandas as pd
import numpy as np
import re
import unicodedata
import csv
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

table_source="mvtart"
prefix_table=PARAMS['PREFIX_TABLE']
table_cible=f"{prefix_table}mvtart"
login_table=f"{prefix_table}login"
primary_keys=["pk"]
column_updates = "dtemvt" # None pour tout recharger
environement=PARAMS['ENVIRONNEMENT_CLIENT']
dataset_cible = f"dwh/db_mg6jk45h_{environement}/"

logger = logging.getLogger(__name__)

DEFAULT_START_DATE = "2020-08-01"

column_trino = {
    "arcleunik": "BIGINT", "f_ocleunik": "BIGINT", "mvcleunik": "BIGINT",
    "typmvt": "BIGINT", "c0_cleunik": "BIGINT", "e0_cleunik": "BIGINT",
    "etat": "BIGINT", "e0_codart": "VARCHAR", "e0_libart": "VARCHAR",
    "id_article_lot": "BIGINT", "id_origine": "BIGINT", "id_label": "BIGINT",
    "pk": "VARCHAR",
    "descfic_statut": "BIGINT",
    "qteart": "DOUBLE", "poremise": "DOUBLE", "prixht": "DOUBLE",
    "taux_tva": "DOUBLE", "pmpart": "DOUBLE", "uatous": "DOUBLE",
    "qteusart": "DOUBLE", "pcb": "DOUBLE", "totht": "DOUBLE",
    "totttc": "DOUBLE", "pmpart_ttc": "DOUBLE", "qtefac": "DOUBLE",
    "pufac": "DOUBLE", "uatoufac": "DOUBLE", "ufac": "VARCHAR",
    "qtef": "DOUBLE", "puf": "DOUBLE", "stockavant": "DOUBLE", "pmp_avt": "DOUBLE",
    "dteimp": "DATE", "dtemvt": "TIMESTAMP",  # dtemvt=timestamp(6), dteimp=date
    "dlc": "DATE",
    "bio": "BOOLEAN", "circuit_court": "BOOLEAN", "valide": "BOOLEAN",
    "echantillon": "BOOLEAN", "statut_dlc": "BOOLEAN",
    "codss1": "VARCHAR", "codss2": "VARCHAR", "stypmvt": "VARCHAR",
    "nobon": "VARCHAR", "codun": "VARCHAR", "trv": "VARCHAR", "numlot": "VARCHAR",
    "libart": "VARCHAR", "usart": "VARCHAR", "codate": "VARCHAR",
    "reference": "VARCHAR", "commentaire": "VARCHAR", "chemin_pj": "VARCHAR",
    "login_site": "VARCHAR",
}

def transform_dataframe(df: pd.DataFrame, login_site, descfic_statut) -> pd.DataFrame:
    df["login_site"] = login_site

    for col in df.columns:
        df.rename(columns={col: to_snake_case(col)}, inplace=True)

    df = df[[
        "arcleunik","codss1","codss2","f_ocleunik","mvcleunik","typmvt","c0_cleunik","qteart",
        "poremise","prixht","taux_tva","pmpart","uatous","e0_cleunik","etat",
        "qteusart","pcb","totht","totttc","pmpart_ttc","qtefac","pufac",
        "e0_codart","e0_libart","uatoufac","ufac","qtef","puf","stockavant",
        "pmp_avt","id_article_lot","id_origine","id_label",
        "dteimp","dtemvt","dlc","bio","chemin_pj","circuit_court","valide",
        "echantillon","statut_dlc","stypmvt","nobon","codun","trv","numlot",
        "libart","usart","codate","reference","commentaire","login_site"
    ]]

    df["pk"] = str(login_site) + "_" + df["mvcleunik"].astype(str)
    df["descfic_statut"] = descfic_statut

    str_cols = df.select_dtypes(include='object').columns
    df[str_cols] = df[str_cols].replace(r'^\s*$', np.nan, regex=True)

    # Strip des espaces parasites sur les colonnes VARCHAR
    for col in ["codss1", "codss2", "stypmvt"]:
        if col in df.columns:
            df[col] = df[col].where(df[col].isna(), df[col].astype(str).str.strip())

    # BIGINT
    for col in ["arcleunik","f_ocleunik","mvcleunik","typmvt","c0_cleunik",
                "e0_cleunik","etat","id_article_lot","id_origine","id_label"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # DOUBLE
    for col in ["qteart","poremise","prixht","taux_tva","pmpart","uatous",
                "qteusart","pcb","totht","totttc","pmpart_ttc","qtefac","pufac",
                "uatoufac","qtef","puf","stockavant","pmp_avt"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # TIMESTAMP
    for col in ["dtemvt"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # DATE
    for col in ["dteimp", "dlc"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # BOOLEAN : forcer en bool numpy (pas quoted dans to_csv)
    for col in ["bio","circuit_court","valide","echantillon","statut_dlc"]:
        if col in df.columns:
            df[col] = df[col].map(
                lambda x: True if str(x).strip().lower() in ("true","1","yes") 
                else (False if str(x).strip().lower() in ("false","0","no") else np.nan)
            )

    return df

# !!!! A partir de là normalement rien à changer

def get_webgerest_table_data(table_name: str, login_request: str, from_date: str = None):
    base_url = PARAMS['BASE_URL']
    auth_url = f"{base_url}/auth"
    auth_params = {
        "client_id": PARAMS["CLIENT_WEBGEREST"],
        "client_secret": PARAMS["SECRET_KEY_WEBGEREST"]
    }

    # Authentification
    auth_response = requests.get(auth_url, params=auth_params)
    auth_response.raise_for_status()
    token = auth_response.json().get("token")

    if not token:
        raise Exception("Token non reçu")

    # Récupération des données avec gestion du retry pour l'erreur 500
    table_url = f"{base_url}/{table_name}"
    headers = {"Authorization": token}
    params = {"LOGIN": login_request}

    if from_date:
        params["from_date"] = from_date

    max_retries = 2
    for attempt in range(max_retries + 1):
        table_response = requests.get(table_url, headers=headers, params=params)

        if table_response.status_code == 500 and attempt < max_retries:
            logger.info(f"Erreur 500 détectée pour {login_request}. Nouvel essai dans 30s... ({attempt+1}/{max_retries})")
            time.sleep(30)
            continue

        table_response.raise_for_status()
        break

    json_data = table_response.json()
    message = (json_data.get("message") if isinstance(json_data, dict) else None) or {}
    data_list = message.get("data", [])

    if not data_list:
        return None

    return pd.DataFrame(data_list)


def to_snake_case(text: str) -> str:
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[\s\-]+', '_', text)
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', text)
    text = text.lower()
    text = re.sub(r'_+', '_', text)
    text = text.strip('_')
    text = re.sub(r'^id(?=[^_])', 'id_', text)
    return text


def upsert(source, table: str, primary_keys: list, df: pd.DataFrame):
    """Effectue un MERGE (upsert) sur la table cible."""
    step = 500
    df.columns = df.columns.str.replace(".", "_").str.lower()
    total_upserted = 0

    # Définition des types Trino par colonne
    column_types = column_trino

    for start in range(0, len(df), step):
        df_batch = df[start:start + step]
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

        values = raw_csv

        # Génération des CAST dans le SELECT englobant
        cols = df_batch.columns.to_list()
        cast_exprs = []
        for col in cols:
            t = column_types.get(col)
            if t:
                cast_exprs.append(f"CAST({col} AS {t}) AS {col}")
            else:
                cast_exprs.append(col)

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

        sql_upsert = sql.replace("\n", " ").replace("\r", " ")
        result = source.query(sql_upsert)

        if isinstance(result, pd.DataFrame) and not result.empty:
            batch_count = result.iloc[0, 0]
        else:
            batch_count = len(df_batch)

        total_upserted += batch_count
        logger.info(f"Batch {start // step + 1}: {batch_count} lignes upsertées")

    logger.info(f"Total: {total_upserted} lignes upsertées")
    return total_upserted


def get_all_last_dates(source) -> dict:
    """Récupère en une seule requête la dernière date_modif pour tous les logins.

    Filtre sur les 2 dernières années pour permettre le partition pruning Iceberg
    et éviter le scan complet de la table (qui provoque un timeout forepaas et
    des exécutions concurrentes).
    Sites avec données plus anciennes tombent sur DEFAULT_START_DATE côté appelant.
    """
    if column_updates is None:
        return {}

    cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    scan_from = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    try:
        t0 = time.time()
        result = source.query(
            f"SELECT login_site, MAX({column_updates}) as last_update "
            f"FROM {table_cible} "
            f"WHERE {column_updates} >= TIMESTAMP '{scan_from} 00:00:00' "
            f"GROUP BY login_site"
        )
        logger.info(f"  [ALL] get_all_last_dates (requête groupée, fenêtre 2 ans): {time.time() - t0:.2f}s")
        if not isinstance(result, pd.DataFrame) or result.empty:
            return {}
        out = {}
        for _, row in result.iterrows():
            login = row["login_site"]
            last_update = row["last_update"]
            if pd.notna(last_update):
                max_date = last_update if isinstance(last_update, str) else last_update.strftime("%Y-%m-%d")
                out[login] = min(max_date, cutoff)
        return out
    except Exception as e:
        logger.warning(f"Erreur récupération des dernières dates: {e}")
        return {}


def fetch_for_login(login: str, updated_since: str, descfic_statut=None):
    """Fetch API + transform uniquement — pas d'écriture Trino (thread-safe)."""
    try:
        logger.info(f"  [{login}] Fetch API depuis: {updated_since}")
        if column_updates is None:
            df = get_webgerest_table_data(table_source, login)
        else:
            df = get_webgerest_table_data(table_source, login, from_date=updated_since)

        if df is not None and not df.empty:
            return login, transform_dataframe(df, login, descfic_statut)
        return login, None
    except Exception as e:
        logger.error(f"  [{login}] Echec fetch: {e}")
        return login, None

_LOCK_TABLE = f"{prefix_table}job_lock"
_LOCK_KEY = "mvtart"
_LOCK_MAX_AGE_MINUTES = 240  # libère le verrou si une instance est bloquée depuis > 4h


def _try_acquire_lock(source) -> bool:
    """Tente d'acquérir le verrou distribué. Retourne True si acquis."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        existing = source.query(
            f"SELECT started_at FROM {_LOCK_TABLE} WHERE job_key = '{_LOCK_KEY}'"
        )
        if isinstance(existing, pd.DataFrame) and not existing.empty:
            started = pd.to_datetime(existing.iloc[0]["started_at"])
            age_min = (datetime.now() - started).total_seconds() / 60
            if age_min < _LOCK_MAX_AGE_MINUTES:
                logger.warning(
                    f"[mvtart] Verrou détecté (démarré il y a {age_min:.0f} min) — "
                    f"instance concurrente en cours, sortie."
                )
                return False
            logger.warning(f"[mvtart] Verrou périmé ({age_min:.0f} min) — reprise du contrôle.")
        source.query(f"DELETE FROM {_LOCK_TABLE} WHERE job_key = '{_LOCK_KEY}'")
        source.query(
            f"INSERT INTO {_LOCK_TABLE} (job_key, started_at) "
            f"VALUES ('{_LOCK_KEY}', TIMESTAMP '{now_str}')"
        )
        return True
    except Exception as e:
        logger.warning(f"[mvtart] Impossible d'acquérir le verrou ({e}) — on continue sans verrou.")
        return True  # fail-open : mieux vaut tourner sans verrou que bloquer indéfiniment


def _release_lock(source) -> None:
    try:
        source.query(f"DELETE FROM {_LOCK_TABLE} WHERE job_key = '{_LOCK_KEY}'")
    except Exception as e:
        logger.warning(f"[mvtart] Échec libération verrou: {e}")


def customfunc(event):
    logger.info("job : mvtart")
    source = connect(dataset_cible)

    if not _try_acquire_lock(source):
        return

    df_login_site = source.select(login_table)
    df_login_site = df_login_site[df_login_site["profil"] == 2]
    df_login_site = df_login_site[df_login_site["fictif"] == False]

    # Routing via descfic
    df_descfic = source.query(
        f"SELECT login_group, statut FROM {prefix_table}descfic WHERE UPPER(nomfic) = 'MVTART'"
    )
    statut_by_group = {}
    if isinstance(df_descfic, pd.DataFrame) and not df_descfic.empty:
        statut_by_group = dict(zip(df_descfic["login_group"], df_descfic["statut"]))

    # Récupération de toutes les dates en une seule requête
    last_dates = get_all_last_dates(source)

    # Construire la liste des (login, last_date, descfic_statut) à fetcher
    rows = list(df_login_site.itertuples())
    login_params = []
    groupes_traites = set()
    for row in rows:
        login_group = row.logingroupe
        statut = statut_by_group.get(login_group)
        if statut == 1:
            if login_group in groupes_traites:
                continue
            groupes_traites.add(login_group)
            last_date = last_dates.get(login_group, DEFAULT_START_DATE)
            login_params.append((login_group, last_date, 1))
        else:
            last_date = last_dates.get(row.login, DEFAULT_START_DATE)
            login_params.append((row.login, last_date, statut))

    # Phase 1 : fetch API en parallèle (I/O-bound → threads)
    MAX_WORKERS = 5
    resultats = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_for_login, login, last_date, descfic_statut): login
            for login, last_date, descfic_statut in login_params
        }
        for future in as_completed(futures):
            login, df = future.result()
            resultats[login] = df

    # Phase 2 : écriture Trino séquentielle
    MAX_WRITE_RETRIES = 3
    COMMIT_RETRY_DELAYS = [30, 60, 120]

    import random

    def _is_retriable(e: Exception) -> bool:
        s = str(e)
        return (
            "CommitFailedException" in s
            or "branch main has changed" in s
            or "ICEBERG_CATALOG_ERROR" in s
            or "Failed to load view" in s
            or "RESTError 503" in s
            or "Response ended prematurely" in s
            or "no healthy upstream" in s
        )

    sites_en_echec = []
    for login, last_date, _ in login_params:
        df_transformed = resultats.get(login)
        if df_transformed is not None and not df_transformed.empty:
            success = False
            for attempt in range(MAX_WRITE_RETRIES + 1):
                try:
                    min_date = df_transformed[column_updates].dropna().min()
                    if pd.notna(min_date):
                        min_date_str = min_date.strftime("%Y-%m-%d")
                        logger.info(f"  [{login}] Suppression des données existantes depuis {min_date_str}...")
                        t_del = time.time()
                        source.query(
                            f"DELETE FROM {table_cible} "
                            f"WHERE login_site = '{login}' AND {column_updates} >= TIMESTAMP '{min_date_str} 00:00:00'"
                        )
                        logger.info(f"  [{login}] DELETE: {time.time() - t_del:.2f}s")
                    t_ins = time.time()
                    bulk_insert(source, table_cible, df_transformed)
                    logger.info(f"  [{login}] INSERT ({len(df_transformed)} lignes): {time.time() - t_ins:.2f}s")
                    success = True
                    break
                except Exception as e:
                    if _is_retriable(e) and attempt < MAX_WRITE_RETRIES:
                        base_delay = COMMIT_RETRY_DELAYS[attempt]
                        delay = base_delay + random.randint(0, base_delay // 2)
                        logger.warning(f"  [{login}] Erreur transitoire (tentative {attempt+1}/{MAX_WRITE_RETRIES}), retry dans {delay}s: {type(e).__name__}")
                        time.sleep(delay)
                    else:
                        logger.error(f"  [{login}] Echec écriture Trino: {e}")
                        sites_en_echec.append(login)
                        break
            if not success and login not in sites_en_echec:
                sites_en_echec.append(login)
        else:
            logger.info(f"  [{login}] Aucune donnée à charger")
            if login not in resultats:
                sites_en_echec.append(login)

    _release_lock(source)

    if sites_en_echec:
        logger.warning(
            f"Les données n'ont pas entièrement été récupérées pour les sites: "
            f"{', '.join(sites_en_echec)}. Elles le seront au prochain appel du job."
        )
    else:
        logger.info("Tous les sites ont été synchronisés avec succès.")
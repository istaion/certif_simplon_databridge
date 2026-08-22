from forepaas.dwh import connect, bulk_insert
from forepaas.core.settings import PARAMS
import logging
import requests
import pandas as pd
import numpy as np
import re
import unicodedata
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

table_source = "plandis"
prefix_table = PARAMS['PREFIX_TABLE']
table_cible = f"{prefix_table}plandis"
login_table = f"{prefix_table}login"
primary_keys = ['pk']
column_updates = "datdis"
environement = PARAMS['ENVIRONNEMENT_CLIENT']
dataset_cible = f"dwh/db_mg6jk45h_{environement}/"

logger = logging.getLogger(__name__)

DEFAULT_START_DATE = "2016-08-01"


def transform_dataframe(df: pd.DataFrame, login_site, descfic_statut) -> pd.DataFrame:
    for col in df.columns:
        df.rename(columns={col: to_snake_case(col)}, inplace=True)

    for col in df.select_dtypes(include=['object']).columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip()

    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.replace('', np.nan)
    df = df.replace('nan', np.nan)

    # Note: 'code_site' vient de l'API (CodeSite) — indispensable quand statut=1
    # pour identifier l'établissement au sein d'un fetch de groupe
    columns_to_keep = [
        'plcleunik', 'an', 'semaine', 'jour', 'service',
        'datdis', 'effectif', 'prestation', 'code_site'
    ]
    columns_to_keep = [c for c in columns_to_keep if c in df.columns]
    df = df[columns_to_keep]

    integer_columns = ['plcleunik', 'effectif', 'code_site']
    for col in integer_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    if 'datdis' in df.columns:
        df['datdis'] = pd.to_datetime(df['datdis'], format='%Y%m%d', errors='coerce')

    df["login_site"] = str(login_site)
    df["pk"] = str(login_site) + "_" + df["plcleunik"].astype(str)
    df["descfic_statut"] = descfic_statut

    return df


# !!!! A partir de là normalement rien à changer

def get_webgerest_table_data(table_name: str, login_request: str, from_date: str = None):
    base_url = PARAMS['BASE_URL']
    auth_url = f"{base_url}/auth"
    auth_params = {
        "client_id": PARAMS["CLIENT_WEBGEREST"],
        "client_secret": PARAMS["SECRET_KEY_WEBGEREST"]
    }

    auth_response = requests.get(auth_url, params=auth_params)
    auth_response.raise_for_status()
    token = auth_response.json().get("token")

    if not token:
        raise Exception("Token non reçu")

    table_url = f"{base_url}/{table_name}"
    headers = {"Authorization": token}
    params = {"LOGIN": login_request}

    if from_date:
        params["from_date"] = from_date

    max_retries = 1
    for attempt in range(max_retries + 1):
        table_response = requests.get(table_url, headers=headers, params=params)

        if table_response.status_code == 500 and attempt < max_retries:
            logger.info(f"Erreur 500 détectée pour {login_request}. Nouvel essai dans 60s...")
            time.sleep(60)
            continue

        table_response.raise_for_status()
        break

    json_data = table_response.json()
    data_list = json_data.get("message", {}).get("data", [])

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


def get_all_last_dates(source) -> dict:
    """Récupère en une seule requête la dernière date_modif pour tous les logins."""
    cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    try:
        t0 = time.time()
        result = source.query(
            f"SELECT login_site, MAX({column_updates}) as last_update FROM {table_cible} GROUP BY login_site"
        )
        logger.info(f"  [ALL] get_all_last_dates (requête groupée): {time.time() - t0:.2f}s")
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
        df = get_webgerest_table_data(table_source, login, from_date=updated_since)

        if df is not None and not df.empty:
            return login, transform_dataframe(df, login, descfic_statut)
        return login, None
    except Exception as e:
        logger.error(f"  [{login}] Echec fetch: {e}")
        return login, None


def customfunc(event):
    source = connect(dataset_cible)
    df_login_site = source.select(login_table)
    df_login_site = df_login_site[df_login_site["profil"] == 2]
    df_login_site = df_login_site[df_login_site["fictif"] == False]

    df_descfic = source.query(
        f"SELECT login_group, statut FROM {prefix_table}descfic WHERE nomfic = 'PLANDIS'"
    )
    statut_by_group = {}
    if isinstance(df_descfic, pd.DataFrame) and not df_descfic.empty:
        statut_by_group = dict(zip(df_descfic["login_group"], df_descfic["statut"]))

    last_dates = get_all_last_dates(source)

    rows = list(df_login_site.itertuples())
    login_params = []
    groupes_traites = set()
    for row in rows:
        login_group = row.logingroupe
        statut = statut_by_group.get(login_group)
        if statut != 2:
            if login_group in groupes_traites:
                continue
            groupes_traites.add(login_group)
            login_params.append((login_group, DEFAULT_START_DATE, statut))
        else:
            last_date = last_dates.get(row.login, DEFAULT_START_DATE)
            login_params.append((row.login, last_date, 2))

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
    sites_en_echec = []
    for login, last_date, _ in login_params:
        df_transformed = resultats.get(login)
        if df_transformed is not None and not df_transformed.empty:
            try:
                min_date = df_transformed[column_updates].dropna().min()
                if pd.notna(min_date):
                    min_date_str = min_date.strftime("%Y-%m-%d")
                    logger.info(f"  [{login}] Suppression des données existantes depuis {min_date_str}...")
                    source.query(
                        f"DELETE FROM {table_cible} "
                        f"WHERE login_site = '{login}' AND {column_updates} >= DATE '{min_date_str}'"
                    )
                bulk_insert(source, table_cible, df_transformed)
                logger.info(f"  [{login}] {len(df_transformed)} lignes insérées")
            except Exception as e:
                logger.error(f"  [{login}] Echec écriture Trino: {e}")
                sites_en_echec.append(login)
        else:
            logger.info(f"  [{login}] Aucune donnée à charger")
            if login not in resultats:
                sites_en_echec.append(login)

    if sites_en_echec:
        logger.warning(
            f"Les données n'ont pas entièrement été récupérées pour les sites/groupes: "
            f"{', '.join(str(x) for x in sites_en_echec)}. Elles le seront au prochain appel du job."
        )
    else:
        logger.info("Tous les sites ont été synchronisés avec succès.")

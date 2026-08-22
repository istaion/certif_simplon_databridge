from forepaas.dwh import connect, bulk_insert
from forepaas.core.settings import PARAMS
import logging
import requests
import pandas as pd
import numpy as np
import re
import unicodedata
import json
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

table_source = "article"
prefix_table = PARAMS['PREFIX_TABLE']
table_cible = f"{prefix_table}article"
login_table = f"{prefix_table}login"
primary_keys = ["pk"]
column_updates = "datmod"
environement = PARAMS['ENVIRONNEMENT_CLIENT']
dataset_cible = f"dwh/db_mg6jk45h_{environement}/"

logger = logging.getLogger(__name__)

DEFAULT_START_DATE = "2020-01-01"

column_trino = {
    "pk": "VARCHAR",
    "arcleunik": "BIGINT",
    "codart": "BIGINT",
    "percde": "BIGINT",
    "delai": "BIGINT",
    "reglecalc": "BIGINT",
    "reglearr": "BIGINT",
    "nature": "BIGINT",
    "statut": "BIGINT",
    "id_marche": "BIGINT",
    "id_lots": "BIGINT",
    "dpaart": "DOUBLE",
    "pmpart": "DOUBLE",
    "qtethe": "DOUBLE",
    "qtecom": "DOUBLE",
    "slqte": "DOUBLE",
    "pdsbrut": "DOUBLE",
    "glucides": "DOUBLE",
    "lipides": "DOUBLE",
    "protides": "DOUBLE",
    "energie": "DOUBLE",
    "qtecomus": "DOUBLE",
    "seuilmaxi": "DOUBLE",
    "pdsvol": "DOUBLE",
    "usart_vers_ufam": "DOUBLE",
    "dcreart": "DATE",
    "datmod": "TIMESTAMP",
    "conditionne": "BOOLEAN",
    "suivi": "BOOLEAN",
    "bio": "BOOLEAN",
    "login_site": "VARCHAR",
    "descfic_statut": "BIGINT",
    "libart": "VARCHAR",
    "codfamart": "VARCHAR",
    "sfaart": "VARCHAR",
    "usart": "VARCHAR",
    "codstk": "VARCHAR",
    "codtva": "VARCHAR",
    "catol": "VARCHAR",
    "uniteft": "VARCHAR",
    "designation_externe": "VARCHAR",
    "allergenes": "VARCHAR",
    "codarticle": "VARCHAR",
    "logincrea": "VARCHAR",
}


def transform_dataframe(df: pd.DataFrame, login_site, descfic_statut) -> pd.DataFrame:
    df["login_site"] = login_site

    for col in df.columns:
        df.rename(columns={col: to_snake_case(col)}, inplace=True)

    df.rename(columns={"usart_vers_ufa": "usart_vers_ufam"}, inplace=True)

    cols_cible = [
        "login_site", "arcleunik", "codart", "libart", "codfamart",
        "sfaart", "usart", "dpaart", "pmpart", "qtethe", "qtecom", "dcreart",
        "slqte", "codstk", "codtva", "conditionne", "suivi", "percde", "delai",
        "reglecalc", "catol", "pdsbrut", "glucides", "lipides", "protides",
        "energie", "qtecomus", "reglearr", "uniteft", "seuilmaxi", "nature",
        "pdsvol", "statut", "datmod", "id_marche", "id_lots", "bio",
        "designation_externe", "allergenes", "codarticle", "usart_vers_ufam",
        "logincrea",
    ]
    cols_present = [c for c in cols_cible if c in df.columns]
    df = df[cols_present]

    str_cols = df.select_dtypes(include='object').columns
    df[str_cols] = df[str_cols].replace(r'^\s*$', np.nan, regex=True)

    def _to_str_safe(x):
        if isinstance(x, (list, dict)):
            return json.dumps(x)
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return str(x) if not isinstance(x, str) else x

    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df[col] = df[col].apply(_to_str_safe)

    df["pk"] = str(login_site) + "_" + df["arcleunik"].astype(str)
    df["descfic_statut"] = descfic_statut

    # BIGINT
    for col in ["arcleunik", "codart", "percde", "delai",
                "reglecalc", "reglearr", "nature", "statut", "id_marche", "id_lots"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # DOUBLE
    for col in ["dpaart", "pmpart", "qtethe", "qtecom", "slqte", "pdsbrut",
                "glucides", "lipides", "protides", "energie", "qtecomus",
                "seuilmaxi", "pdsvol", "usart_vers_ufam"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # TIMESTAMP
    for col in ["datmod"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # DATE (format source "YYYYMMDD")
    for col in ["dcreart"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")

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


def get_all_last_dates(source) -> dict:
    """Récupère en une seule requête la dernière date_modif pour tous les logins."""
    if column_updates is None:
        return {}

    cutoff = (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d")
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


def customfunc(event):
    logger.info("job : article")
    source = connect(dataset_cible)
    df_login_site = source.select(login_table)
    df_login_site = df_login_site[df_login_site["profil"] == 2]
    df_login_site = df_login_site[df_login_site["fictif"] == False]

    df_descfic = source.query(
        f"SELECT login_group, statut FROM {prefix_table}descfic WHERE UPPER(nomfic) = 'ARTICLE'"
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
                        f"WHERE login_site = '{login}' AND {column_updates} >= TIMESTAMP '{min_date_str} 00:00:00'"
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
            f"Les données n'ont pas entièrement été récupérées pour les sites: "
            f"{', '.join(sites_en_echec)}. Elles le seront au prochain appel du job."
        )
    else:
        logger.info("Tous les sites ont été synchronisés avec succès.")

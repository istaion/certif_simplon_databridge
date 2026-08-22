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

table_source = "descfic"
prefix_table = PARAMS['PREFIX_TABLE']
table_cible = f"{prefix_table}descfic"
login_table = f"{prefix_table}login"
primary_keys = ["nomfic", "login_group"]
column_updates = None
environement = PARAMS['ENVIRONNEMENT_CLIENT']
dataset_cible = f"dwh/db_mg6jk45h_{environement}/"

logger = logging.getLogger(__name__)


def transform_dataframe(df: pd.DataFrame, login_group: str) -> pd.DataFrame:
    for col in df.columns:
        df.rename(columns={col: to_snake_case(col)}, inplace=True)

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()

    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.replace('', np.nan)
    df = df.replace('nan', np.nan)

    df = df[["nomfic", "description", "statut"]]

    df["statut"] = pd.to_numeric(df["statut"], errors="coerce")
    df["login_group"] = str(login_group)

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


def fetch_and_load_for_login_group(source, login_group: str) -> bool:
    try:
        logger.info(f"  [{login_group}] Récupération descfic...")
        df = get_webgerest_table_data(table_source, login_group)

        if df is not None and not df.empty:
            df_transformed = transform_dataframe(df, login_group)
            if not df_transformed.empty:
                logger.info(f"  [{login_group}] Suppression des données existantes...")
                source.query(f"DELETE FROM {table_cible} WHERE login_group = '{login_group}'")
                bulk_insert(source, table_cible, df_transformed)
                logger.info(f"  [{login_group}] {len(df_transformed)} lignes insérées")
            else:
                logger.info(f"  [{login_group}] Aucune donnée après transformation")
        else:
            logger.info(f"  [{login_group}] Aucune donnée reçue")

        return True

    except Exception as e:
        logger.error(f"  [{login_group}] Echec de la récupération: {e}")
        return False


def customfunc(event):
    source = connect(dataset_cible)
    df_login_site = source.select(login_table)
    df_login_site = df_login_site[df_login_site["profil"] == 2]
    df_login_site = df_login_site[df_login_site["fictif"] == False]

    login_groups = df_login_site["logingroupe"].dropna().unique().tolist()
    logger.info(f"{len(login_groups)} login_group(s) distincts trouvés.")

    groupes_en_echec = []

    for login_group in login_groups:
        logger.info(f"Chargement logingroupe: {login_group}...")
        success = fetch_and_load_for_login_group(source, login_group)

        if not success:
            groupes_en_echec.append(login_group)

    if groupes_en_echec:
        logger.warning(
            f"Les données n'ont pas entièrement été récupérées pour les groupes: "
            f"{', '.join(groupes_en_echec)}. Elles le seront au prochain appel du job."
        )
    else:
        logger.info("Tous les login_groups ont été synchronisés avec succès.")

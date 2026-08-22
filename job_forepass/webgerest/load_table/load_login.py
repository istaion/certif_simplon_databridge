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
from datetime import datetime

table_source="login"
prefix_table=PARAMS['PREFIX_TABLE']
table_cible=f"{prefix_table}login"
primary_keys=["login"]
environement=PARAMS['ENVIRONNEMENT_CLIENT']
login_group = None
region=None
dataset_cible = f"dwh/db_mg6jk45h_{environement}/"
logger = logging.getLogger(__name__)

if environement == "prodcentre":
    login_group = "REG-CENT"
elif environement == "prodrhone":
    login_group = "CD38"
    
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

def old_to_snake_case(text: str) -> str:
    """Ancienne fonction to_snake_case mais je fait trop souvent appel à login pour la changer..."""
    text = re.sub(r'[\s\-]+', '_', text)
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', text)
    text = text.lower()
    text = re.sub(r'_+', '_', text)
    text = text.strip('_')
    return text

def customfunc(event):
    df = get_webgerest_table_data(table_source, login_group)
    df["DATACTIF"] = pd.to_datetime(df["DATACTIF"], format="%Y%m%d")
    df["DATINACTIF"] = pd.to_datetime(df["DATINACTIF"], format="%Y%m%d")
    colunms = df.columns
    for col in colunms:
        df.rename(columns={col:old_to_snake_case(col)}, inplace=True)
    df=df[df["profil"]==2]
    logger.info(df.dtypes)
    login_test=["0410000A","0280000A","0190000A","0181111Z","0180000X"]
    for login in login_test:
        df.loc[df["login"] == login, "fictif"] = True
    df["code_postal"]= pd.to_numeric(df["code_postal"], errors="coerce")
    source = connect(dataset_cible)
    source.query(f"DELETE FROM {table_cible}")
    bulk_insert(source, table_cible, df)

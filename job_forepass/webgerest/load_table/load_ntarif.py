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

table_source="ntarif"
prefix_table=PARAMS['PREFIX_TABLE']
table_cible=f"{prefix_table}ntarif"
login_table=f"{prefix_table}login"
primary_keys=["pk"]
column_updates = None
environement=PARAMS['ENVIRONNEMENT_CLIENT']
dataset_cible = f"dwh/db_mg6jk45h_{environement}/"

logger = logging.getLogger(__name__)

DEFAULT_START_DATE = "2016-08-01"

def transform_dataframe(df: pd.DataFrame, login_site, descfic_statut) -> pd.DataFrame:
    """Transforme le DataFrame brut API en format cible."""
    # Nettoyage des colonnes (Snake Case) D'ABORD
    for col in df.columns:
        df.rename(columns={col: to_snake_case(col)}, inplace=True)

    # CRITIQUE: Nettoyage des espaces avant/après pour toutes les colonnes string
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip()

    # Nettoyage des chaînes vides/espaces avant conversion
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.replace('', np.nan)
    df = df.replace('nan', np.nan)

    # Sélection des colonnes nécessaires
    df = df[[
        'id_ntarif', 'exercice', 'codcli', 'codcat', 'typfac', 
        'codimp', 'creditbrut', 'creditnet', 'forfaitan', 'forfaittrim', 
        'coeftrim', 'nbjan', 'nbjtrim', 'nbjsem', 'coefserv', 'statut', 
        'prestation'
    ]]
    
    numerical_colunms = [
        'id_ntarif','codcli','typfac'
    ]
    date_colunms = []

    # Conversion numérique
    for col in numerical_colunms:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Conversion date
    for col in date_colunms:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Ajout des colonnes métier À LA FIN
    df["login_site"] = str(login_site)
    df["pk"] = str(login_site) + "_" + df["id_ntarif"].astype(str)
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

    for start in range(0, len(df), step):
        df_batch = df[start:start + step]

        values = "(" + "{}".format(
            df_batch.to_csv(
                header=None,
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
                quotechar="'",
                na_rep="NULL",
                date_format="TIMESTAMP %Y-%m-%d %H:%M:%S%z TIMESTAMP"
            )
        ).replace("'NULL'", "NULL") \
         .replace("'TIMESTAMP ", "TIMESTAMP '") \
         .replace(" TIMESTAMP'", "'") \
         .strip("\n") \
         .replace("\n", "),(") + ")"

        sql = f"""
            MERGE INTO {table}
            USING (
                VALUES {values}
            ) AS tmp ({','.join(df_batch.columns.to_list())})
            ON {' AND '.join([f'{table}.{f} IS NOT DISTINCT FROM tmp.{f}' for f in primary_keys])}
            WHEN MATCHED THEN
                UPDATE SET {','.join([f'{f}=tmp.{f}' for f in df_batch.columns.to_list() if f not in primary_keys])}
            WHEN NOT MATCHED THEN
                INSERT ({','.join(df_batch.columns.to_list())})
                VALUES ({','.join([f'tmp.{f}' for f in df_batch.columns.to_list()])})
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


def get_last_date_modif_for_login(source, login: str) -> str:
    """Récupère la dernière date_modif pour un login donné."""
    if column_updates is None:
        # Si pas de colonne de mise à jour, on recharge tout depuis DEFAULT_START_DATE
        return DEFAULT_START_DATE
    
    try:
        result = source.query(
            f"SELECT MAX({column_updates}) as last_update FROM {table_cible} WHERE login_site = '{login}'"
        )
        if isinstance(result, pd.DataFrame) and not result.empty:
            last_update = result.iloc[0, 0]
            if pd.notna(last_update):
                if isinstance(last_update, str):
                    return last_update
                return last_update.strftime("%Y-%m-%d")
        return DEFAULT_START_DATE
    except Exception as e:
        logger.warning(f"Erreur récupération dernière date pour {login}: {e}")
        return DEFAULT_START_DATE


def fetch_and_load_for_login(source, login: str, updated_since: str, descfic_statut=None) -> bool:
    """
    Récupère et charge les données pour un login.
    - Si column_updates est None : utilise bulk_insert (écrase tout)
    - Sinon : utilise upsert (mise à jour incrémentale)
    Retourne True si tout a réussi, False si un fetch a échoué.
    """
    try:
        logger.info(f"  [{login}] Récupération depuis: {updated_since}")
        if column_updates is None:
            df = get_webgerest_table_data(
                table_source,
                login
            )
        else:
            df = get_webgerest_table_data(
                table_source,
                login,
                from_date=updated_since
            )
        if df is not None and not df.empty:
            df_transformed = transform_dataframe(df, login, descfic_statut)
            if not df_transformed.empty:
                if column_updates is None:
                    # Mode bulk_insert : supprime les anciennes données du login puis insère
                    logger.info(f"  [{login}] Mode bulk_insert - Suppression des données existantes...")
                    source.query(f"DELETE FROM {table_cible} WHERE login_site = '{login}'")
                    bulk_insert(source, table_cible, df_transformed)
                    logger.info(f"  [{login}] {len(df_transformed)} lignes insérées")
                else:
                    # Mode upsert : mise à jour incrémentale
                    upsert(source, table_cible, primary_keys, df_transformed)
                    logger.info(f"  [{login}] {len(df_transformed)} lignes upsertées")
            else:
                logger.info(f"  [{login}] Aucune donnée après transformation")
        else:
            logger.info(f"  [{login}] Aucune donnée depuis {updated_since}")

        return True

    except Exception as e:
        logger.error(f"  [{login}] Echec de la récupération: {e}")
        return False


def customfunc(event):
    source = connect(dataset_cible)
    df_login_site = source.select(login_table)
    df_login_site = df_login_site[df_login_site["profil"] == 2]
    df_login_site = df_login_site[df_login_site["fictif"] == False]

    # Récupération du statut de ntarif par logingroupe depuis descfic
    df_descfic = source.query(
        f"SELECT login_group, statut FROM {prefix_table}descfic WHERE nomfic = 'NTARIF'"
    )
    statut_by_group = {}
    if isinstance(df_descfic, pd.DataFrame) and not df_descfic.empty:
        statut_by_group = dict(zip(df_descfic["login_group"], df_descfic["statut"]))

    sites_en_echec = []
    groupes_traites = set()

    for row in df_login_site.itertuples():
        login_group = row.logingroupe
        statut = statut_by_group.get(login_group)

        if statut != 2:
            # Statut != 2 = commun au groupe : fetch une seule fois avec logingroupe
            if login_group in groupes_traites:
                continue
            groupes_traites.add(login_group)
            logger.info(f"Chargement groupe {login_group} (statut={statut}, commun au groupe)...")
            success = fetch_and_load_for_login(source, login_group, DEFAULT_START_DATE, descfic_statut=statut)
            if not success:
                sites_en_echec.append(login_group)
        else:
            # Statut 2 = unique au site : fetch par login_site
            logger.info(f"Chargement {row.login}...")
            last_date = get_last_date_modif_for_login(source, row.login)
            logger.info(f"  Dernière date_modif en base: {last_date}")
            success = fetch_and_load_for_login(source, row.login, last_date, descfic_statut=2)
            if not success:
                sites_en_echec.append(row.login)

    if sites_en_echec:
        logger.warning(
            f"Les données n'ont pas entièrement été récupérées pour les sites/groupes: "
            f"{', '.join(str(x) for x in sites_en_echec)}. Elles le seront au prochain appel du job."
        )
    else:
        logger.info("Tous les sites ont été synchronisés avec succès.")
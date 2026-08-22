"""
Synchronisation mvtart - version Spark
=======================================
Differences vs load_mvtart.py :
  - Fetch parallele avec 20 workers ThreadPoolExecutor (vs 5)
  - Toutes les donnees unifiees en un seul DataFrame Spark
  - 1 seul MERGE INTO Iceberg (vs ~200 DELETE+INSERT sequentiels)
    -> elimine les CommitFailedException et reduit ~3h a quelques minutes
  - Pas de probleme d'instance concurrente : MERGE est une operation atomique

Docs forepaas Spark : https://docs.dataplatform.ovh.net/#/en/product/dpe/actions/custom-pyspark/
"""

# --------------------------------------------------------------------- #
# 0. Imports
# --------------------------------------------------------------------- #
import logging
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from forepaas.core.settings import CONFIG, PARAMS
from forepaas.dwh import connect
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# 1. Configuration (identique a load_mvtart.py)
# --------------------------------------------------------------------- #
table_source   = "mvtart"
prefix_table   = PARAMS["PREFIX_TABLE"]
table_cible    = f"{prefix_table}mvtart"
login_table    = f"{prefix_table}login"
column_updates = "dtemvt"
environement   = PARAMS["ENVIRONNEMENT_CLIENT"]
dataset_cible  = f"dwh/db_mg6jk45h_{environement}/"

MAX_FETCH_WORKERS  = 20
FETCH_WINDOW_DAYS  = 45  # fenetre glissante - MERGE INTO est idempotent


# --------------------------------------------------------------------- #
# 2. Schema Spark (deduit de column_trino)
# --------------------------------------------------------------------- #
_TRINO_TO_SPARK = {
    "BIGINT":    LongType(),
    "DOUBLE":    DoubleType(),
    "VARCHAR":   StringType(),
    "BOOLEAN":   BooleanType(),
    "DATE":      DateType(),
    "TIMESTAMP": TimestampType(),
}

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
    "dteimp": "DATE", "dtemvt": "TIMESTAMP",
    "dlc": "DATE",
    "bio": "BOOLEAN", "circuit_court": "BOOLEAN", "valide": "BOOLEAN",
    "echantillon": "BOOLEAN", "statut_dlc": "BOOLEAN",
    "codss1": "VARCHAR", "codss2": "VARCHAR", "stypmvt": "VARCHAR",
    "nobon": "VARCHAR", "codun": "VARCHAR", "trv": "VARCHAR", "numlot": "VARCHAR",
    "libart": "VARCHAR", "usart": "VARCHAR", "codate": "VARCHAR",
    "reference": "VARCHAR", "commentaire": "VARCHAR", "chemin_pj": "VARCHAR",
    "login_site": "VARCHAR",
}

SPARK_SCHEMA = StructType([
    StructField(col, _TRINO_TO_SPARK[typ], nullable=True)
    for col, typ in column_trino.items()
])


# --------------------------------------------------------------------- #
# 3. Helpers
# --------------------------------------------------------------------- #
def to_snake_case(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = text.lower()
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    text = re.sub(r"^id(?=[^_])", "id_", text)
    return text


def transform_dataframe(df: pd.DataFrame, login_site, descfic_statut) -> pd.DataFrame:
    df["login_site"] = login_site

    for col in df.columns:
        df.rename(columns={col: to_snake_case(col)}, inplace=True)

    df = df[[
        "arcleunik", "codss1", "codss2", "f_ocleunik", "mvcleunik", "typmvt", "c0_cleunik",
        "qteart", "poremise", "prixht", "taux_tva", "pmpart", "uatous", "e0_cleunik", "etat",
        "qteusart", "pcb", "totht", "totttc", "pmpart_ttc", "qtefac", "pufac",
        "e0_codart", "e0_libart", "uatoufac", "ufac", "qtef", "puf", "stockavant",
        "pmp_avt", "id_article_lot", "id_origine", "id_label",
        "dteimp", "dtemvt", "dlc", "bio", "chemin_pj", "circuit_court", "valide",
        "echantillon", "statut_dlc", "stypmvt", "nobon", "codun", "trv", "numlot",
        "libart", "usart", "codate", "reference", "commentaire", "login_site",
    ]]

    df["pk"] = str(login_site) + "_" + df["mvcleunik"].astype(str)
    df["descfic_statut"] = descfic_statut

    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].replace(r"^\s*$", np.nan, regex=True)

    for col in ["codss1", "codss2", "stypmvt"]:
        if col in df.columns:
            df[col] = df[col].where(df[col].isna(), df[col].astype(str).str.strip())

    # BIGINT -> pandas Int64 nullable (pd.NA, pas float NaN) pour compat Spark LongType
    for col in ["arcleunik", "f_ocleunik", "mvcleunik", "typmvt", "c0_cleunik",
                "e0_cleunik", "etat", "id_article_lot", "id_origine", "id_label",
                "descfic_statut"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ["qteart", "poremise", "prixht", "taux_tva", "pmpart", "uatous",
                "qteusart", "pcb", "totht", "totttc", "pmpart_ttc", "qtefac", "pufac",
                "uatoufac", "qtef", "puf", "stockavant", "pmp_avt"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    for col in ["dtemvt"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in ["dteimp", "dlc"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # None (pas np.nan float) pour que PyArrow/Spark accepte BooleanType nullable
    for col in ["bio", "circuit_court", "valide", "echantillon", "statut_dlc"]:
        if col in df.columns:
            df[col] = df[col].map(
                lambda x: True if str(x).strip().lower() in ("true", "1", "yes")
                else (False if str(x).strip().lower() in ("false", "0", "no") else None)
            )

    return df


def get_webgerest_table_data(table_name: str, login_request: str, from_date: str = None):
    base_url  = PARAMS["BASE_URL"]
    auth_resp = requests.get(
        f"{base_url}/auth",
        params={"client_id": PARAMS["CLIENT_WEBGEREST"], "client_secret": PARAMS["SECRET_KEY_WEBGEREST"]},
        timeout=30,
    )
    auth_resp.raise_for_status()
    token = auth_resp.json().get("token")
    if not token:
        raise Exception("Token non recu")

    params = {"LOGIN": login_request}
    if from_date:
        params["from_date"] = from_date

    max_retries = 2
    for attempt in range(max_retries + 1):
        resp = requests.get(
            f"{base_url}/{table_name}",
            headers={"Authorization": token},
            params=params,
            timeout=240,
        )
        if resp.status_code == 500 and attempt < max_retries:
            logger.info(f"Erreur 500 pour {login_request}. Retry dans 30s... ({attempt+1}/{max_retries})")
            time.sleep(30)
            continue
        resp.raise_for_status()
        break

    message   = (resp.json().get("message") if isinstance(resp.json(), dict) else None) or {}
    data_list = message.get("data", [])
    return pd.DataFrame(data_list) if data_list else None


def get_default_start_date() -> str:
    """Rolling window - no Iceberg scan needed since MERGE INTO is idempotent."""
    return (datetime.now() - timedelta(days=FETCH_WINDOW_DAYS)).strftime("%Y-%m-%d")


def fetch_for_login(login: str, updated_since: str, descfic_statut=None):
    """Fetch + transform for one login (thread-safe, no write)."""
    t0 = time.time()
    try:
        df = get_webgerest_table_data(table_source, login, from_date=updated_since)
        if df is not None and not df.empty:
            result = transform_dataframe(df, login, descfic_statut)
            logger.info(f"  [{login}] {len(result)} lignes ({time.time() - t0:.1f}s)")
            return login, result
        logger.info(f"  [{login}] vide ({time.time() - t0:.1f}s)")
        return login, None
    except Exception as e:
        logger.error(f"  [{login}] Echec: {e} ({time.time() - t0:.1f}s)")
        return login, None


# --------------------------------------------------------------------- #
# 4. Job principal
# --------------------------------------------------------------------- #
def customfunc(event):
    logger.info("job : mvtart (Spark)")

    spark        = SparkSession.builder.appName("forepass_mvtart").getOrCreate()
    dataplant_id = CONFIG.get("dataplant_id", "mg6jk45h")
    logger.info(f"Spark {spark.version} - dataplant_id={dataplant_id}")

    # Full Iceberg table name : {catalog}.{schema}.{table}
    # Example : db_mg6jk45h_prodcentre.prodcentre.wg_test_mvtart
    table_iceberg = f"db_{dataplant_id}_{environement}.{environement}.{table_cible}"
    logger.info(f"Table Iceberg cible : {table_iceberg}")

    source = connect(dataset_cible)

    df_login = source.select(login_table).toPandas()
    logger.info(f"df_login: {len(df_login)} lignes chargees")
    df_login = df_login[(df_login["profil"] == 2) & (df_login["fictif"] == False)]

    df_descfic = source.select(f"{prefix_table}descfic").toPandas()
    df_descfic = df_descfic[df_descfic["nomfic"].str.upper() == "MVTART"][["login_group", "statut"]]
    statut_by_group = (
        dict(zip(df_descfic["login_group"], df_descfic["statut"]))
        if not df_descfic.empty else {}
    )
    logger.info(f"df_descfic: {len(statut_by_group)} groupes statut MVTART")

    fetch_from = get_default_start_date()
    logger.info(f"Fetch depuis : {fetch_from} (fenetre 45 jours)")

    login_params    = []
    groupes_traites = set()
    for row in df_login.itertuples():
        login_group = getattr(row, "logingroupe", None)
        login_val   = getattr(row, "login", None)
        statut      = statut_by_group.get(login_group)
        if statut == 1:
            if login_group in groupes_traites:
                continue
            groupes_traites.add(login_group)
            login_params.append((login_group, fetch_from, 1))
        else:
            login_params.append((login_val, fetch_from, statut))
    login_params = login_params[:1]
    logger.info(f"{len(login_params)} logins a synchroniser : {login_params}")

    t_fetch     = time.time()
    all_dfs     = []
    sites_echec = []

    logger.info("Demarrage fetch API...")
    with ThreadPoolExecutor(max_workers=MAX_FETCH_WORKERS) as executor:
        futures = {
            executor.submit(fetch_for_login, login, last_date, descfic_statut): login
            for login, last_date, descfic_statut in login_params
        }
        for future in as_completed(futures):
            login, df_pd = future.result()
            if df_pd is not None and not df_pd.empty:
                all_dfs.append(df_pd)
            elif df_pd is None and login in [l for l, _, _ in login_params]:
                sites_echec.append(login)

    logger.info(
        f"Fetch termine en {time.time() - t_fetch:.1f}s - "
        f"{len(all_dfs)} sites avec donnees, {len(sites_echec)} en echec"
    )

    if not all_dfs:
        logger.warning("Aucune donnee recuperee - fin du job.")
        del source
        return

    df_all_pd = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"{len(df_all_pd)} lignes au total avant ecriture Iceberg")

    logger.info("Conversion pandas -> Spark DataFrame...")
    # createDataFrame mappe par position (chemin non-Arrow) : aligner les colonnes sur le schema
    schema_cols = [f.name for f in SPARK_SCHEMA.fields]
    df_all_pd = df_all_pd[schema_cols]
    df_spark  = spark.createDataFrame(df_all_pd, schema=SPARK_SCHEMA)
    df_spark.cache()
    row_count = df_spark.count()
    logger.info(f"Spark DataFrame : {row_count} lignes, {len(SPARK_SCHEMA.fields)} colonnes")

    df_spark.createOrReplaceTempView("_mvtart_updates")

    cols        = [f.name for f in SPARK_SCHEMA.fields]
    update_set  = ", ".join(f"t.{c} = s.{c}" for c in cols if c != "pk")
    insert_cols = ", ".join(cols)
    insert_vals = ", ".join(f"s.{c}" for c in cols)

    logger.info(f"Demarrage MERGE INTO {table_iceberg}...")
    t_merge = time.time()
    spark.sql(f"""
        MERGE INTO {table_iceberg} AS t
        USING _mvtart_updates AS s
        ON t.pk = s.pk
        WHEN MATCHED THEN UPDATE SET {update_set}
        WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})
    """)
    logger.info(f"MERGE termine en {time.time() - t_merge:.1f}s - 1 commit Iceberg pour {row_count} lignes")

    df_spark.unpersist()
    del source

    if sites_echec:
        logger.warning(
            f"Donnees non recuperees pour : {', '.join(sites_echec)}. "
            f"Elles le seront au prochain appel du job."
        )
    else:
        logger.info("Tous les sites ont ete synchronises avec succes.")


# forepaas PySpark soumet via spark-submit -> script est __main__, pas importe
if __name__ == "__main__":
    customfunc(None)

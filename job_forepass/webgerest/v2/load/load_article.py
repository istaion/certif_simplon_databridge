"""
Chargement incrémental de article en architecture V2.

Pour chaque table cible {PREFIX_TABLE}{safe_id(identifier)}_article :
  - SELECT MAX(datmod) → from_date (ou DEFAULT_START_DATE si vide)
  - Fetch API avec from_date
  - DELETE WHERE datmod >= min(données reçues)
  - bulk_insert

PARAMS requis :
    BASE_URL, ENVIRONNEMENT_CLIENT, PREFIX_TABLE,
    CLIENT_WEBGEREST, SECRET_KEY_WEBGEREST, WEBGEREST_LOGIN_GROUPS
"""

import json
import logging
import random
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from forepaas.core.settings import PARAMS
from forepaas.dwh import bulk_insert, connect

logger = logging.getLogger(__name__)

# ── Paramètres ─────────────────────────────────────────────────────────────────

WEBGEREST_BASE_URL = PARAMS["BASE_URL"]
DATASET_CIBLE      = f"dwh/{PARAMS['ENVIRONNEMENT_CLIENT']}/"
SERVER_PREFIX      = PARAMS["PREFIX_TABLE"]
CLIENT_WEBGEREST   = PARAMS["CLIENT_WEBGEREST"]
SECRET_WEBGEREST   = PARAMS["SECRET_KEY_WEBGEREST"]
LOGIN_GROUPS       = json.loads(PARAMS["WEBGEREST_LOGIN_GROUPS"])

DEFAULT_START_DATE = "2016-08-01"
MAX_WORKERS        = 5
MAX_WRITE_RETRIES  = 3
RETRY_DELAYS       = [30, 60, 120]

# Fenêtre de rattrapage : from_date ne dépasse jamais now() - LOOKBACK_DAYS,
# même si MAX(datmod) est dans le futur (dates de planification, anomalies).
# Relancer ponctuellement à 90 pour rattraper un historique de corrections tardives,
# puis remettre à 7 ou 30 pour le job quotidien.
LOOKBACK_DAYS = 7

# ── Colonnes finales V2 (sans login_site / pk / descfic_statut) ────────────────

_FINAL_COLUMNS = [
    "arcleunik", "codart", "libart", "codfamart", "sfaart", "usart",
    "dpaart", "pmpart", "qtethe", "qtecom", "dcreart", "slqte",
    "codstk", "codtva", "conditionne", "suivi", "percde", "delai",
    "reglecalc", "catol", "pdsbrut", "glucides", "lipides", "protides",
    "energie", "qtecomus", "reglearr", "uniteft", "seuilmaxi", "nature",
    "pdsvol", "statut", "datmod", "id_marche", "id_lots", "bio",
    "designation_externe", "allergenes", "codarticle", "usart_vers_ufam",
    "logincrea",
]

# ── Utilitaires ────────────────────────────────────────────────────────────────

def _safe_id(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).lower().strip("_")


def _to_snake_case(text: str) -> str:
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = text.lower()
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    text = re.sub(r"^id(?=[^_])", "id_", text)
    return text


def _is_retriable(e: Exception) -> bool:
    s = str(e)
    return any(k in s for k in (
        "CommitFailedException", "branch main has changed",
        "ICEBERG_CATALOG_ERROR", "Failed to load view",
        "RESTError 503", "Response ended prematurely", "no healthy upstream",
    ))


# ── Transform V2 ──────────────────────────────────────────────────────────────

def _transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={col: _to_snake_case(col) for col in df.columns})
    df = df.rename(columns={"usart_vers_ufa": "usart_vers_ufam"})

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
    df.replace("nan", np.nan, inplace=True)

    for col in ["arcleunik", "codart", "percde", "delai", "reglecalc",
                "reglearr", "nature", "statut", "id_marche", "id_lots"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ["dpaart", "pmpart", "qtethe", "qtecom", "slqte", "pdsbrut",
                "glucides", "lipides", "protides", "energie", "qtecomus",
                "seuilmaxi", "pdsvol", "usart_vers_ufam"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # TIMESTAMP → garde pd.Timestamp pour strftime dans le DELETE
    if "datmod" in df.columns:
        df["datmod"] = pd.to_datetime(df["datmod"], errors="coerce")

    # DATE
    if "dcreart" in df.columns:
        df["dcreart"] = pd.to_datetime(df["dcreart"], errors="coerce").apply(
            lambda x: x.date() if pd.notna(x) else None
        )

    for col in ["conditionne", "suivi", "bio"]:
        if col in df.columns:
            df[col] = df[col].map(
                lambda x: True if str(x).strip().lower() in ("true", "1", "yes")
                else (False if str(x).strip().lower() in ("false", "0", "no") else None)
            )

    return df[[c for c in _FINAL_COLUMNS if c in df.columns]]


# ── Client Webgerest ───────────────────────────────────────────────────────────

def _get_token() -> str:
    resp = requests.get(
        f"{WEBGEREST_BASE_URL}/auth",
        params={"client_id": CLIENT_WEBGEREST, "client_secret": SECRET_WEBGEREST},
        timeout=30,
    )
    resp.raise_for_status()
    token = resp.json().get("token")
    if not token:
        raise RuntimeError("Token non reçu depuis /auth")
    return token


def _fetch(identifier: str, from_date: str) -> pd.DataFrame | None:
    token = _get_token()
    for attempt in range(3):
        resp = requests.get(
            f"{WEBGEREST_BASE_URL}/article",
            headers={"Authorization": token},
            params={"LOGIN": identifier, "from_date": from_date},
            timeout=300,
        )
        if resp.status_code == 500 and attempt < 2:
            logger.warning(f"  [{identifier}] 500 — retry dans 30s ({attempt+1}/2)")
            time.sleep(30)
            token = _get_token()
            continue
        resp.raise_for_status()
        break
    data_list = (resp.json().get("message") or {}).get("data", [])
    return pd.DataFrame(data_list) if data_list else None


# ── Dernière date par table cible ─────────────────────────────────────────────

def _get_last_date(source, target_table: str) -> str:
    cutoff = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    try:
        result = source.query(f"SELECT MAX(datmod) AS last_date FROM {target_table}")
        if isinstance(result, pd.DataFrame) and not result.empty:
            val = result.iloc[0]["last_date"]
            if pd.notna(val):
                date_str = val if isinstance(val, str) else val.strftime("%Y-%m-%d")
                return min(date_str, cutoff)
    except Exception as e:
        logger.warning(f"  [{target_table}] impossible de lire MAX(datmod): {e}")
    return DEFAULT_START_DATE


# ── Entry point ────────────────────────────────────────────────────────────────

def customfunc(event):
    logger.info("job : load_article V2")
    source = connect(DATASET_CIBLE)
    login_table = f"{SERVER_PREFIX}login"

    # 1. login_map depuis centre_login
    df_login = source.select(login_table)
    df_login = df_login[df_login["profil"] == 2]
    if "fictif" in df_login.columns:
        df_login = df_login[df_login["fictif"] != True]
    if "nometabs" in df_login.columns:
        df_login = df_login[~df_login["nometabs"].str.upper().str.contains("DEMO]", na=False)]

    login_map: dict[str, list[str]] = {}
    for _, row in df_login.iterrows():
        login_map.setdefault(row["logingroupe"], []).append(row["login"])

    # 2. statut article par groupe
    statut_by_group: dict[str, int] = {}
    for grp in LOGIN_GROUPS:
        descfic_table = f"{SERVER_PREFIX}{_safe_id(grp)}_descfic"
        try:
            df_d = source.query(
                f"SELECT statut FROM {descfic_table} WHERE UPPER(nomfic) = 'ARTICLE'"
            )
            if isinstance(df_d, pd.DataFrame) and not df_d.empty:
                statut_by_group[grp] = int(df_d.iloc[0]["statut"])
        except Exception as e:
            logger.warning(f"  [{grp}] impossible de lire descfic article: {e}")

    # 3. Tâches (identifier, target_table)
    tasks: list[tuple[str, str]] = []
    for grp in LOGIN_GROUPS:
        statut = statut_by_group.get(grp)
        if statut is None:
            logger.warning(f"[{grp}] pas d'entrée descfic pour article, ignoré")
            continue
        if statut == 2:
            for site in login_map.get(grp, []):
                tasks.append((site, f"{SERVER_PREFIX}{_safe_id(site)}_article"))
        else:
            tasks.append((grp, f"{SERVER_PREFIX}{_safe_id(grp)}_article"))

    logger.info(f"{len(tasks)} table(s) article à synchroniser")

    # 4. Dernières dates en parallèle
    last_dates: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_get_last_date, source, target): (identifier, target)
            for identifier, target in tasks
        }
        for future in as_completed(futures):
            identifier, target = futures[future]
            last_dates[identifier] = future.result()
            logger.info(f"  [{identifier}] from_date = {last_dates[identifier]}")

    # 5. Fetch API en parallèle
    fetched: dict[str, pd.DataFrame | None] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_fetch, identifier, last_dates[identifier]): identifier
            for identifier, _ in tasks
        }
        for future in as_completed(futures):
            identifier = futures[future]
            try:
                fetched[identifier] = future.result()
            except Exception as e:
                logger.error(f"  [{identifier}] fetch échoué: {e}")

    # 6. Écriture séquentielle avec retry Iceberg
    echecs = []
    for identifier, target in tasks:
        df_raw = fetched.get(identifier)
        if df_raw is None or df_raw.empty:
            logger.info(f"  [{identifier}] aucune donnée depuis {last_dates[identifier]}")
            continue

        df = _transform(df_raw)
        if df.empty:
            continue

        min_date = df["datmod"].dropna().min()
        if pd.isna(min_date):
            logger.warning(f"  [{identifier}] datmod toutes nulles, skip")
            continue
        min_date_str = min_date.strftime("%Y-%m-%d")

        for attempt in range(MAX_WRITE_RETRIES + 1):
            try:
                source.query(
                    f"DELETE FROM {target} "
                    f"WHERE datmod >= TIMESTAMP '{min_date_str} 00:00:00'"
                )
                bulk_insert(source, target, df)
                logger.info(f"  [{identifier}] → {target} : {len(df)} lignes (depuis {min_date_str})")
                break
            except Exception as e:
                if _is_retriable(e) and attempt < MAX_WRITE_RETRIES:
                    delay = RETRY_DELAYS[attempt] + random.randint(0, RETRY_DELAYS[attempt] // 2)
                    logger.warning(
                        f"  [{identifier}] erreur transitoire "
                        f"(tentative {attempt+1}/{MAX_WRITE_RETRIES}), retry dans {delay}s: {type(e).__name__}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"  [{identifier}] écriture échouée: {e}")
                    echecs.append(identifier)
                    break

    if echecs:
        logger.warning(f"Échecs : {', '.join(echecs)}. Seront rechargés au prochain appel.")
    else:
        logger.info("load_article V2 terminé avec succès.")

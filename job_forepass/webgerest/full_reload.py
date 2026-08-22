"""
Chargement initial complet des tables Webgerest via appel direct à l'API Webgerest.
Utilise bulk_insert ForePaaS — ne passe PAS par l'application databridge.

Tables chargées :
  - login        : 1 appel avec WEBGEREST_LOGIN_GROUP
  - descfic      : 1 appel par groupe dans WEBGEREST_LOGIN_GROUPS
  - fourn, article, categ, detail_article, famart, label, ntarif, sfaart,
    trimestre, typss1, typss2, fitech  (tables de référence — full reload)
  - effect, feuille, gaspi_saisie_gen, mvtart, mvtart_det, plandis, detpland
    (tables transactionnelles — full reload)

Pour chaque table hors login/descfic, la donnée est récupérée login par login
(tous les logins actifs profil=2 issus de la table login).
"""

import json
import logging
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import requests
from forepaas.core.settings import PARAMS
from forepaas.dwh import bulk_insert, connect

logger = logging.getLogger(__name__)

# ── Paramètres ─────────────────────────────────────────────────────────────────

WEBGEREST_BASE_URL    = PARAMS["BASE_URL"]
WEBGEREST_DATASET     = PARAMS["ENVIRONNEMENT_CLIENT"]
WEBGEREST_PREFIX      = PARAMS["PREFIX_TABLE"]
WEBGEREST_LOGIN_GROUP = PARAMS["WEBGEREST_LOGIN_GROUP"]
CLIENT_WEBGEREST      = PARAMS["CLIENT_WEBGEREST"]
SECRET_KEY_WEBGEREST  = PARAMS["SECRET_KEY_WEBGEREST"]

dataset_cible = f"dwh/{WEBGEREST_DATASET}/"
p = WEBGEREST_PREFIX

# Mode test : True = 1 seul groupe/login par table (pour valider le pipeline rapidement)
TEST_SESSION = False

# Route API Webgerest quand différente du nom de table cible
_API_ROUTE = {
    "detail_article": "detailarticle",
}

# Clé descfic (UPPER) quand différente de table_name.upper()
_DESCFIC_KEY = {
    "detail_article": "DETAILARTICLE",
}

_REF_TABLES = [
    "fourn", "article", "categ", "detail_article", "famart",
    "label", "ntarif", "sfaart", "trimestre", "typss1", "typss2", "fitech",
]

_TXN_TABLES = [
    "effect", "feuille", "gaspi_saisie_gen", "mvtart", "mvtart_det",
    "plandis", "detpland",
]

# {table: {colonne: format_strptime_ou_None}} → converties en datetime.date
# None = auto-detect pandas, "%Y%m%d" = format YYYYMMDD source Webgerest
_DATE_COLUMNS: dict[str, dict[str, str | None]] = {
    "trimestre": {"datdeb": "%Y%m%d", "datfin": "%Y%m%d"},
    "effect":    {"efdate": None, "date_import": None, "date_modif": None},
    "article":   {"dcreart": "%Y%m%d"},
    "plandis":   {"datdis": "%Y%m%d"},
    "mvtart":    {"dteimp": "%Y%m%d", "dlc": "%Y%m%d"},
    "detpland":  {"datfab": "%Y%m%d", "datbes": "%Y%m%d"},
}

# {table: [colonnes]} → converties en datetime (timestamp, sans .date())
_TIMESTAMP_COLUMNS: dict[str, list[str]] = {
    "article": ["datmod"],
    "mvtart":  ["dtemvt"],
}

# {table: {colonne_snake: préfixe}} → liste explodée en préfixe_1, préfixe_2, ... (1-indexé)
_ARRAY_EXPAND: dict[str, dict[str, str]] = {
    "famart": {"cptfam": "cptfam"},
}

_COLUMN_RENAME: dict[str, dict[str, str]] = {
    "effect":         {"efcleunik": "id_effect"},
    "article":        {"usart_vers_ufa": "usart_vers_ufam"},
    "label":          {"lib_label": "label"},
}

# Colonnes business (hors login_site) concaténées pour former pk
_PK_SOURCE_COLUMNS: dict[str, list[str]] = {
    "fourn":            ["f_ocleunik"],
    "article":          ["arcleunik"],
    "categ":            ["codcat"],
    "detail_article":   ["id_detailarticle"],
    "famart":           ["codfamart"],
    "label":            ["id_label"],
    "ntarif":           ["id_ntarif"],
    "sfaart":           ["codfamart", "sfaart"],
    "trimestre":        ["notrim", "exercice"],
    "typss1":           ["codss1"],
    "typss2":           ["codss2"],
    "fitech":           ["codft"],
    "effect":           ["id_effect"],
    "feuille":          ["fecleunik"],
    "gaspi_saisie_gen": ["id_gaspi_saisie_gen"],
    "mvtart":           ["mvcleunik"],
    "mvtart_det":       ["id_mvtart_det"],
    "plandis":          ["plcleunik"],
    "detpland":         ["d1_cleunik"],
}

# ── Utilitaires ────────────────────────────────────────────────────────────────

def _to_snake_case(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = text.lower()
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    text = re.sub(r"^id(?=[^_])", "id_", text)
    return text


def _snake_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={col: _to_snake_case(col) for col in df.columns})


def _expand_arrays(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Explose les colonnes contenant des listes en colonnes individuelles (1-indexé)."""
    for col, prefix in _ARRAY_EXPAND.get(table_name, {}).items():
        if col not in df.columns:
            continue
        max_len = df[col].apply(lambda x: len(x) if isinstance(x, list) else 0).max()
        for i in range(1, int(max_len) + 1):
            df[f"{prefix}_{i}"] = df[col].apply(
                lambda x, idx=i: x[idx - 1] if isinstance(x, list) and len(x) >= idx else None
            )
        df.drop(columns=[col], inplace=True)
    return df


def _clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Sérialise les listes/dicts en JSON, strip les strings, remplace vides par NaN."""
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df[col] = df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
            )
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
    df.replace("nan", np.nan, inplace=True)
    return df

def _parse_date_columns(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    for col, fmt in _DATE_COLUMNS.get(table_name, {}).items():
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col].astype(str).str.strip(),
                format=fmt,
                errors="coerce"
            ).apply(lambda x: x.date() if pd.notna(x) else None)
    for col in _TIMESTAMP_COLUMNS.get(table_name, []):
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col].astype(str).str.strip(),
                errors="coerce"
            )
    return df

# ── Client Webgerest ───────────────────────────────────────────────────────────

def _get_token() -> str:
    resp = requests.get(
        f"{WEBGEREST_BASE_URL}/auth",
        params={"client_id": CLIENT_WEBGEREST, "client_secret": SECRET_KEY_WEBGEREST},
        timeout=30,
    )
    resp.raise_for_status()
    token = resp.json().get("token")
    if not token:
        raise RuntimeError("Token non reçu depuis /auth")
    return token


def _fetch_table(api_name: str, login: str, from_date: str = None) -> pd.DataFrame | None:
    """Appelle GET /{api_name}?LOGIN={login} avec retry automatique sur 500."""
    token = _get_token()
    url = f"{WEBGEREST_BASE_URL}/{api_name}"
    params: dict = {"LOGIN": login}
    if from_date:
        params["from_date"] = from_date

    for attempt in range(2):
        resp = requests.get(
            url,
            headers={"Authorization": token},
            params=params,
            timeout=300,
        )
        if resp.status_code == 500 and attempt == 0:
            logger.warning(f"[{login}/{api_name}] 500 — retry dans 60s")
            time.sleep(60)
            token = _get_token()
            continue
        resp.raise_for_status()
        break

    json_data = resp.json()
    if not json_data:
        return None
    data_list = (json_data.get("message") or {}).get("data", [])
    if not data_list:
        return None
    return pd.DataFrame(data_list)


# ── Sync login ─────────────────────────────────────────────────────────────────

def _sync_login(source) -> pd.DataFrame:
    """Charge la table login. Retourne le DataFrame complet pour extraction des logins."""
    logger.info("Sync login...")
    df = _fetch_table("login", WEBGEREST_LOGIN_GROUP)
    if df is None or df.empty:
        logger.warning("login : aucune donnée reçue")
        return pd.DataFrame()

    df = _snake_columns(df)
    df = _clean_strings(df)

    for col in ("datactif", "datinactif"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce").apply(
                lambda x: x.date() if pd.notna(x) else None
            )

    for col in ("statut", "code_site", "profil", "secteur",
                "id_arrondissement", "id_canton", "code_postal"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ("fictif", "gestionnaire"):
        if col in df.columns:
            df[col] = df[col].map(
                lambda x: True if str(x).lower() in ("true", "1", "yes")
                else (False if str(x).lower() in ("false", "0", "no") else None)
            )
    df=df[df["profil"]==2]
    df = df[~df["nometabs"].str.upper().str.contains("DEMO]", na=False)]
    source.query(f"DELETE FROM {p}login")
    bulk_insert(source, f"{p}login", df)
    logger.info(f"login : {len(df)} lignes insérées")
    return df


# ── Sync descfic ───────────────────────────────────────────────────────────────

def _sync_descfic(source, login_groups: list[str]) -> dict[str, dict[str, int]]:
    """Charge descfic et retourne {login_group: {TABLE_UPPER: statut}}."""
    logger.info("Sync descfic...")
    source.query(f"DELETE FROM {p}descfic")

    statut_map: dict[str, dict[str, int]] = {}
    for grp in login_groups:
        df = _fetch_table("descfic", grp)
        if df is None or df.empty:
            logger.warning(f"descfic [{grp}] : aucune donnée")
            continue
        df = _snake_columns(df)
        df = _clean_strings(df)
        df["login_group"] = grp
        if "statut" in df.columns:
            df["statut"] = pd.to_numeric(df["statut"], errors="coerce").astype("Int64")
        bulk_insert(source, f"{p}descfic", df)
        logger.info(f"descfic [{grp}] : {len(df)} lignes insérées")

        statut_map[grp] = {
            str(row["nomfic"]).upper(): int(row["statut"])
            for _, row in df.iterrows()
            if pd.notna(row.get("statut")) and pd.notna(row.get("nomfic"))
        }

    return statut_map


# ── Bulk insert avec retry ─────────────────────────────────────────────────────

_BULK_INSERT_MAX_RETRIES = 5
_BULK_INSERT_RETRY_DELAY = 120  # secondes

def _bulk_insert_with_retry(source, table: str, df, login: str) -> None:
    for attempt in range(1, _BULK_INSERT_MAX_RETRIES + 2):
        try:
            bulk_insert(source, table, df)
            return
        except Exception as e:
            if attempt > _BULK_INSERT_MAX_RETRIES:
                raise
            logger.warning(
                f"  [{login}] bulk_insert échoué (tentative {attempt}/{_BULK_INSERT_MAX_RETRIES}) : {e} "
                f"— retry dans {_BULK_INSERT_RETRY_DELAY}s"
            )
            time.sleep(_BULK_INSERT_RETRY_DELAY)


# ── Sync tables génériques (par login) ────────────────────────────────────────

def _sync_table(
    source,
    table_name: str,
    logins: list[str],
    login_to_group: dict[str, str],
    descfic_statut_map: dict[str, dict[str, int]],
) -> None:
    """Full reload d'une table.
    - statut descfic != 2 : 1 fetch par groupe (identifier = logingroupe)
    - statut descfic == 2 : 1 fetch par login individuel
    """
    api_name = _API_ROUTE.get(table_name, table_name)
    logger.info(f"Sync {table_name} (via route /{api_name})...")

    source.query(f"DELETE FROM {p}{table_name}")

    pk_cols = _PK_SOURCE_COLUMNS.get(table_name, [])

    # Regrouper les logins par login_group en préservant l'ordre de première apparition
    group_to_logins: dict[str, list[str]] = {}
    for login in logins:
        grp = login_to_group.get(login, login)
        group_to_logins.setdefault(grp, []).append(login)

    ok, ko = 0, 0

    def _fetch_and_insert(identifier: str, statut) -> None:
        nonlocal ok, ko
        try:
            df = _fetch_table(api_name, identifier)
            if df is None or df.empty:
                logger.debug(f"  [{identifier}] aucune donnée")
                return
            df = _snake_columns(df)
            renames = _COLUMN_RENAME.get(table_name, {})
            if renames:
                df.rename(columns=renames, inplace=True)
            df = _expand_arrays(df, table_name)
            df = _clean_strings(df)
            df = _parse_date_columns(df, table_name)
            df["login_site"] = identifier
            if pk_cols:
                df["pk"] = (
                    df[[col for col in pk_cols if col in df.columns]]
                    .astype(str)
                    .agg(lambda row: identifier + "_" + "_".join(row), axis=1)
                )
            df["descfic_statut"] = statut
            _bulk_insert_with_retry(source, f"{p}{table_name}", df, identifier)
            ok += 1
            logger.info(f"  [{identifier}] {len(df)} lignes insérées")
        except Exception as e:
            logger.error(f"  [{identifier}] Erreur sur {table_name} : {e}")
            ko += 1

    descfic_key = _DESCFIC_KEY.get(table_name, table_name.upper())

    for grp, grp_logins in group_to_logins.items():
        statut = descfic_statut_map.get(grp, {}).get(descfic_key)
        ok_before = ok
        if statut == 2:
            for login in grp_logins:
                _fetch_and_insert(login, statut)
                if TEST_SESSION and ok > ok_before:
                    break
        else:
            _fetch_and_insert(grp, statut)
        if TEST_SESSION and ok > ok_before:
            logger.info(f"  [TEST_SESSION] arrêt après le premier groupe avec données ({grp})")
            break

    logger.info(f"{table_name} : {ok} insertions OK, {ko} en erreur")


# ── Entry point ────────────────────────────────────────────────────────────────

def customfunc(event):
    source = connect(dataset_cible)

    # 1. Login
    df_login = _sync_login(source)
    if df_login.empty:
        raise RuntimeError("Table login vide — impossible de continuer")

    # 2. Descfic — groupes dérivés de la table login
    login_groups = df_login["logingroupe"].dropna().unique().tolist()
    descfic_statut_map = _sync_descfic(source, login_groups)

    # 3. Extraction des logins actifs (profil=2, non fictifs, non DEMO)
    mask = (df_login.get("profil", pd.Series(dtype=object)) == 2)
    if "fictif" in df_login.columns:
        mask = mask & (df_login["fictif"] != True)
    if "nometabs" in df_login.columns:
        mask = mask & (~df_login["nometabs"].str.upper().str.contains("DEMO]", na=False))
    logins = df_login.loc[mask, "login"].dropna().tolist()
    logger.info(f"{len(logins)} logins actifs à traiter")

    # Mapping login_site → login_group pour injection descfic_statut
    login_to_group: dict[str, str] = {}
    if "logingroupe" in df_login.columns:
        login_to_group = dict(
            zip(df_login["login"].dropna(), df_login["logingroupe"].fillna(""))
        )

    # 4. Tables de référence
    for tbl in _REF_TABLES:
        _sync_table(source, tbl, logins, login_to_group, descfic_statut_map)

    # 5. Tables transactionnelles
    for tbl in _TXN_TABLES:
        _sync_table(source, tbl, logins, login_to_group, descfic_statut_map)

    logger.info("Synchronisation Webgerest directe terminée.")

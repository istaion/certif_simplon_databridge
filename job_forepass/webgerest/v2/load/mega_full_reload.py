"""
Rechargement complet de toutes les tables de données Webgerest V2.

Architecture V2 : une table physique par login_group (statut descfic=1) ou par
login_site (statut=2). Les tables login et descfic V2 doivent exister et être
remplies (cf. init_webgerest_v2_tables.py).

Pour chaque table de données : DELETE FROM + bulk_insert.
Pas de login_site / pk / descfic_statut dans les colonnes V2.

Usage ForePaaS : customfunc(event) est l'entry point.

PARAMS requis :
    BASE_URL              : URL de base Webgerest
    ENVIRONNEMENT_CLIENT  : ex "prodcentre"  → dataset dwh/db_mg6jk45h_prodcentre/
    PREFIX_TABLE          : ex "centre_"     → centre_login, centre_cd28_article...
    CLIENT_WEBGEREST      : client_key API
    SECRET_KEY_WEBGEREST  : client_secret API
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

WEBGEREST_BASE_URL = PARAMS["BASE_URL"]
DATASET_CIBLE      = f"dwh/{PARAMS['ENVIRONNEMENT_CLIENT']}/"
SERVER_PREFIX      = PARAMS["PREFIX_TABLE"]
CLIENT_WEBGEREST   = PARAMS["CLIENT_WEBGEREST"]
SECRET_WEBGEREST   = PARAMS["SECRET_KEY_WEBGEREST"]
LOGIN_GROUPS       = json.loads(PARAMS["WEBGEREST_LOGIN_GROUPS"])

# None = toutes les tables de données V2 ; sinon ex: ["article", "mvtart"]
TABLES_FILTER = None

# ── Métadonnées des tables V2 ──────────────────────────────────────────────────

# Route API quand différente du nom de table
_API_ROUTE = {
    "detail_article": "detailarticle",
}

# Clé descfic (UPPER) quand différente de table_name.upper()
_DESCFIC_KEY = {
    "detail_article": "DETAILARTICLE",
}

# Renommages post-snake_case {col_api: col_trino}
_COLUMN_RENAME = {
    "article": {"usart_vers_ufa": "usart_vers_ufam"},
    "fourn":   {"label": "id_label"},
    "effect":  {"efcleunik": "id_effect"},
}

# Colonnes DATE {col: format_strptime|None}
_DATE_COLUMNS = {
    "article":          {"dcreart": "%Y%m%d"},
    "trimestre":        {"datdeb": "%Y%m%d", "datfin": "%Y%m%d"},
    "effect":           {"efdate": None, "date_import": None, "date_modif": None},
    "feuille":          {"efdate": None},
    "gaspi_saisie_gen": {"datej": None},
    "mvtart":           {"dlc": None, "dteimp": None},
    "plandis":          {"datdis": "%Y%m%d"},
    "detpland":         {"datfab": None, "datbes": None},
}

# Colonnes TIMESTAMP(6)
_TIMESTAMP_COLUMNS = {
    "article": ["datmod"],
    "mvtart":  ["dtemvt"],
}

# Colonnes listes à exploser en col_1, col_2, ...
_ARRAY_EXPAND = {
    "famart": {"cptfam": "cptfam"},
}

# Colonnes finales dans l'ordre du schéma V2 (sans login_site / pk / descfic_statut)
_FINAL_COLUMNS: dict[str, list[str]] = {
    "article": [
        "arcleunik", "codart", "libart", "codfamart", "sfaart", "usart",
        "dpaart", "pmpart", "qtethe", "qtecom", "dcreart", "slqte", "codstk",
        "codtva", "conditionne", "suivi", "percde", "delai", "reglecalc",
        "catol", "pdsbrut", "glucides", "lipides", "protides", "energie",
        "qtecomus", "reglearr", "uniteft", "seuilmaxi", "nature", "pdsvol",
        "statut", "datmod", "id_marche", "id_lots", "bio", "designation_externe",
        "allergenes", "codarticle", "usart_vers_ufam", "logincrea",
    ],
    "categ": [
        "codimp", "famcat", "libcat", "nbjsem", "noncompte", "onilait",
        "ordre", "regime", "service", "statut", "typfac", "codcat",
    ],
    "detail_article": [
        "id_detailarticle", "arcleunik", "circuit_court", "saison_deb",
        "saison_fin", "id_label", "id_origine", "codart", "zinc", "fer",
        "iode", "selenium", "dha", "epa", "vit_b3", "vit_b6", "vit_b12",
        "vit_d", "id_calories", "calcium", "sodium", "sucre", "glucose",
        "amidon", "fibre", "vit_c", "magnesium", "vit_e", "vit_b2", "cuivre",
        "vit_a_retinol", "vit_k", "vit_b9", "potassium", "phosphore",
        "vit_b1", "vit_a_beta_caro", "vit_b5", "favori", "jamais_local",
        "acide_gras_sature",
    ],
    "famart": [
        "codfamart", "libfamart", "typart",
        "cptfam_1", "cptfam_2", "cptfam_3", "cptfam_4", "cptfam_5",
        "cptfam_6", "cptfam_7", "cptfam_8", "cptfam_9", "cptfam_10",
        "type", "statut", "ordre", "jamais_local",
    ],
    "fourn": [
        "f_ocleunik", "codfou", "libfou", "codfamfou", "cpfou", "minfac",
        "jours", "delai", "numcli", "percde", "statut", "login",
        "circuit_court", "num_agrement", "bio", "no_engagement",
        "id_fourn_ext", "id_origine", "id_label", "tags", "agrimer_fl",
        "agrimer_lait", "circuit_court_strict",
    ],
    "label": ["code", "id_label", "label", "egalim"],
    "ntarif": [
        "id_ntarif", "exercice", "codcli", "codcat", "typfac", "codimp",
        "creditbrut", "creditnet", "forfaitan", "forfaittrim", "coeftrim",
        "nbjan", "nbjtrim", "nbjsem", "coefserv", "statut", "prestation",
    ],
    "sfaart": [
        "codfamart", "sfaart", "libsfaart", "typart", "unite", "type",
        "statut", "ordre", "jamais_local",
    ],
    "trimestre": ["exercice", "datdeb", "datfin", "notrim", "nbjtrim"],
    "typss1": [
        "libss1", "onilait", "codcpt", "boolsh", "id_catpresta",
        "no_variante", "ordre", "codss1",
    ],
    "typss2": ["libss2", "bool_supprimable", "statut", "codss2"],
    "effect": [
        "efdate", "id_effect", "eftheo", "efreel", "efprev", "credit",
        "codcat", "typfac", "codcli", "effautre", "creditbrut", "origine",
        "date_import", "date_modif", "codss1", "codss2",
    ],
    "feuille": [
        "fecleunik", "efdate", "codss1", "menu", "commentaire", "codss2",
        "cle_jps", "nb_vegetarien", "type_menu", "id_animation",
    ],
    "gaspi_saisie_gen": [
        "id_gaspi_saisie_gen", "datej", "codss1", "codss2", "qte_gen",
        "cle_jps", "eff_prev", "pas_de_tri", "commentaire",
        "eff_prod", "eff_reel_service",
    ],
    "mvtart": [
        "mvcleunik", "dtemvt", "typmvt", "stypmvt", "nobon", "f_ocleunik",
        "arcleunik", "c0_cleunik", "qteart", "codun", "prixht", "poremise",
        "taux_tva", "pmpart", "uatous", "e0_cleunik", "etat", "trv",
        "numlot", "dlc", "libart", "usart", "qteusart", "valide", "pcb",
        "totht", "totttc", "pmpart_ttc", "qtefac", "pufac", "codate",
        "e0_codart", "e0_libart", "uatoufac", "ufac", "qtef", "puf",
        "stockavant", "pmp_avt", "dteimp", "id_article_lot", "reference",
        "circuit_court", "id_origine", "id_label", "echantillon",
        "commentaire", "bio", "chemin_pj", "statut_dlc", "codss1", "codss2",
    ],
    "mvtart_det": [
        "id_mvtart_det", "mvcleunik", "modif_local", "modif_bio",
        "modif_label1", "modif_origine",
    ],
    "plandis": [
        "plcleunik", "an", "semaine", "jour", "service", "datdis",
        "effectif", "prestation", "code_site",
    ],
    "detpland": [
        "d1_cleunik", "typrec", "ordre", "codrec", "plcleunik", "datfab",
        "datbes", "codft", "typord", "recleuniq", "id_mvtpre", "puht",
        "puttc", "typpart", "valsor", "options", "potage", "equillibre",
        "horsgamme", "intolerance", "no_variante", "recleunik_variante",
        "qtefab", "menus", "qtemodif", "qteprev",
        "qtefabr1", "qtefabr2", "qtefabr3",
        "qtemodifr1", "qtemodifr2", "qtemodifr3", "taux_prise",
    ],
    "fitech": [
        "codft", "codrec", "librec", "libred", "typrec", "famrec", "sfamrec",
        "fsfcleuniq", "codate", "codftg4", "typlib", "datcre", "datmaj",
        "dateval", "derfab", "dersor", "nbparts", "pu", "puht", "pvht",
        "pvttc", "coefpv", "conserv", "delliv", "image", "clarec",
        "id_frequence", "id_matos", "codftfab", "pcb", "pcbliv", "tempcons",
        "tconserv_min", "id_enseignant", "id_objetft", "etat", "statut",
        "c0upe", "typcondtucl", "lipides", "glucides", "protides", "energie",
        "usart", "codtva", "tprep", "tcuis", "trefr", "tremi", "etapes",
        "saison", "libmenu", "sel", "reglearr", "porc", "valide", "validedef",
        "intolerance", "sh", "circuitcourt", "bio", "vegetarien", "maison",
        "antigaspi", "grammages", "typcond", "regime", "menus", "condtucl",
        "allergenes", "id_label",
        "poids_ingredient_principal_calcul", "poids_ingredient_principal_saisi",
    ],
}

_REF_TABLES = [
    "fourn", "article", "categ", "detail_article", "famart",
    "label", "ntarif", "sfaart", "trimestre", "typss1", "typss2", "fitech",
]
_TXN_TABLES = [
    "effect", "feuille", "gaspi_saisie_gen", "mvtart", "mvtart_det",
    "plandis", "detpland",
]

# ── Utilitaires ────────────────────────────────────────────────────────────────

def _safe_id(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).lower().strip("_")


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


def _expand_arrays(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
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


def _parse_dates(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    for col, fmt in _DATE_COLUMNS.get(table_name, {}).items():
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col].astype(str).str.strip(), format=fmt, errors="coerce"
            ).apply(lambda x: x.date() if pd.notna(x) else None)
    for col in _TIMESTAMP_COLUMNS.get(table_name, []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col].astype(str).str.strip(), errors="coerce")
    return df


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


def _fetch_table(api_name: str, login: str) -> pd.DataFrame | None:
    token = _get_token()
    url = f"{WEBGEREST_BASE_URL}/{api_name}"
    for attempt in range(2):
        resp = requests.get(
            url,
            headers={"Authorization": token},
            params={"LOGIN": login},
            timeout=300,
        )
        if resp.status_code == 500 and attempt == 0:
            logger.warning(f"[{login}/{api_name}] 500 — retry dans 60s")
            time.sleep(60)
            token = _get_token()
            continue
        resp.raise_for_status()
        break
    data_list = (resp.json().get("message") or {}).get("data", [])
    if not data_list:
        return None
    return pd.DataFrame(data_list)


# ── bulk_insert avec retry ─────────────────────────────────────────────────────

_BULK_RETRIES = 5
_BULK_RETRY_DELAY = 120

def _bulk_insert_with_retry(source, table: str, df: pd.DataFrame, label: str) -> None:
    for attempt in range(1, _BULK_RETRIES + 2):
        try:
            bulk_insert(source, table, df)
            return
        except Exception as e:
            if attempt > _BULK_RETRIES:
                raise
            logger.warning(
                f"  [{label}] bulk_insert échoué (tentative {attempt}/{_BULK_RETRIES}): {e}"
                f" — retry dans {_BULK_RETRY_DELAY}s"
            )
            time.sleep(_BULK_RETRY_DELAY)


# ── Transform V2 ──────────────────────────────────────────────────────────────

def _transform_v2(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    df = df.rename(columns={col: _to_snake_case(col) for col in df.columns})
    renames = _COLUMN_RENAME.get(table_name, {})
    if renames:
        df = df.rename(columns=renames)
    df = _expand_arrays(df, table_name)
    df = _clean_strings(df)
    df = _parse_dates(df, table_name)
    final_cols = _FINAL_COLUMNS.get(table_name, [])
    if final_cols:
        df = df[[c for c in final_cols if c in df.columns]]
    return df


# ── Sync une table V2 ─────────────────────────────────────────────────────────

def _sync_table_v2(
    source,
    table_name: str,
    login_groups: list[str],
    descfic_map: dict[str, dict[str, int]],
    login_map: dict[str, list[str]],
) -> None:
    api_name    = _API_ROUTE.get(table_name, table_name)
    descfic_key = _DESCFIC_KEY.get(table_name, table_name.upper())
    logger.info(f"=== {table_name} (/{api_name}) ===")
    ok = ko = 0

    for grp in login_groups:
        statut = descfic_map.get(grp, {}).get(descfic_key)
        if statut is None:
            logger.warning(f"  [{grp}] pas d'entrée descfic pour {table_name!r}, ignoré")
            continue
        identifiers = login_map.get(grp, []) if statut == 2 else [grp]

        for identifier in identifiers:
            target = f"{SERVER_PREFIX}{_safe_id(identifier)}_{table_name}"
            try:
                df_raw = _fetch_table(api_name, identifier)
                if df_raw is None or df_raw.empty:
                    logger.info(f"  [{identifier}] aucune donnée API")
                    ok += 1
                    continue
                df = _transform_v2(df_raw, table_name)
                if df.empty:
                    logger.info(f"  [{identifier}] vide après transform")
                    ok += 1
                    continue
                source.query(f"DELETE FROM {target}")
                _bulk_insert_with_retry(source, target, df, identifier)
                ok += 1
                logger.info(f"  [{identifier}] → {target} : {len(df)} lignes")
            except Exception as e:
                logger.error(f"  [{identifier}] erreur {table_name}: {e}")
                ko += 1

    logger.info(f"{table_name}: {ok} OK, {ko} erreur(s)")


# ── Entry point ────────────────────────────────────────────────────────────────

def customfunc(event):
    source = connect(DATASET_CIBLE)
    login_table = f"{SERVER_PREFIX}login"

    # 1. login_map depuis la table login V2
    df_login = source.select(login_table)
    df_login = df_login[df_login["profil"] == 2]
    if "fictif" in df_login.columns:
        df_login = df_login[df_login["fictif"] != True]
    if "nometabs" in df_login.columns:
        df_login = df_login[~df_login["nometabs"].str.upper().str.contains("DEMO]", na=False)]

    login_map: dict[str, list[str]] = {}
    for _, row in df_login.iterrows():
        login_map.setdefault(row["logingroupe"], []).append(row["login"])

    logger.info(
        f"{sum(len(v) for v in login_map.values())} logins actifs "
        f"dans {len(LOGIN_GROUPS)} groupe(s) configurés"
    )

    # 2. descfic_map depuis les tables descfic V2 (une par groupe)
    descfic_map: dict[str, dict[str, int]] = {}
    for grp in LOGIN_GROUPS:
        descfic_table = f"{SERVER_PREFIX}{_safe_id(grp)}_descfic"
        try:
            df = source.select(descfic_table)
            descfic_map[grp] = {
                str(row["nomfic"]).upper(): int(row["statut"])
                for _, row in df.iterrows()
                if pd.notna(row.get("nomfic")) and pd.notna(row.get("statut"))
            }
        except Exception as e:
            logger.error(f"Impossible de lire {descfic_table}: {e}")

    # 3. Tables à traiter
    all_tables = _REF_TABLES + _TXN_TABLES
    if TABLES_FILTER:
        all_tables = [t for t in all_tables if t in TABLES_FILTER]
    logger.info(f"{len(all_tables)} table(s) à synchroniser")

    # 4. Chargement table par table
    for table_name in all_tables:
        _sync_table_v2(source, table_name, LOGIN_GROUPS, descfic_map, login_map)

    logger.info("mega_full_reload V2 terminé.")

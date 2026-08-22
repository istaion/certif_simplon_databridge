"""
Création des objets logiques V2 dans l'UI DataPlatform.

Lit login + descfic depuis Trino pour découvrir dynamiquement toutes les tables
V2 et les enregistrer avec le bon chemin de dossier :

    Webgerest/{serveur}/              → login
    Webgerest/{serveur}/{groupe}/     → descfic, tables statut=1
    Webgerest/{serveur}/{groupe}/{site}/  → tables statut=2

PARAMS requis :
    ENVIRONNEMENT_CLIENT, PREFIX_TABLE, WEBGEREST_LOGIN_GROUPS
"""

import json
import logging
import re
import unicodedata

from forepaas.core.settings import PARAMS
from forepaas.dwh import connect
from forepaas.dwh.logical import LogicalObject

logger = logging.getLogger(__name__)

# ── Paramètres ─────────────────────────────────────────────────────────────────

DATASET_CIBLE  = f"dwh/{PARAMS['ENVIRONNEMENT_CLIENT']}/"
DATASET        = PARAMS["ENVIRONNEMENT_CLIENT"]
SERVER_PREFIX  = PARAMS["PREFIX_TABLE"]
LOGIN_GROUPS   = json.loads(PARAMS["WEBGEREST_LOGIN_GROUPS"])

# Mapping nomfic (tel que stocké dans descfic, uppercase) → suffixe de table physique
NOMFIC_TO_TABLE = {
    "ARTICLE":         "article",
    "CATEG":           "categ",
    "DETAILARTICLE":   "detail_article",
    "FAMART":          "famart",
    "FOURN":           "fourn",
    "LABEL":           "label",
    "NTARIF":          "ntarif",
    "SFAART":          "sfaart",
    "TRIMESTRE":       "trimestre",
    "TYPSS1":          "typss1",
    "TYPSS2":          "typss2",
    "EFFECT":          "effect",
    "FEUILLE":         "feuille",
    "GASPI_SAISIE_GEN": "gaspi_saisie_gen",
    "MVTART":          "mvtart",
    "MVTART_DET":      "mvtart_det",
    "PLANDIS":         "plandis",
    "DETPLAND":        "detpland",
    "FITECH":          "fitech",
}

# ── Utilitaires ────────────────────────────────────────────────────────────────

def _safe_id(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).lower().strip("_")


def _register(table: str, path: str, created: list, skipped: list) -> None:
    try:
        lo = LogicalObject()
        lo.create_from_physical(table=table, dataset=DATASET)
        lo.update(table, {"tags": {"path": path}})
        logger.info(f"  [OK] {table} → {path}")
        created.append(table)
    except Exception as e:
        logger.info(f"  [skip] {table}: {e}")
        skipped.append(table)


# ── Entry point ────────────────────────────────────────────────────────────────

def customfunc(event):
    logger.info("job : maj_physical — création des objets logiques V2")
    source = connect(DATASET_CIBLE)

    # "centre_" → "centre"
    serveur = SERVER_PREFIX.rstrip("_")

    # 1. login_map depuis centre_login (identifiants originaux pour les chemins)
    df_login = source.select(f"{SERVER_PREFIX}login")
    df_login = df_login[df_login["profil"] == 2]
    if "fictif" in df_login.columns:
        df_login = df_login[df_login["fictif"] != True]
    if "nometabs" in df_login.columns:
        df_login = df_login[~df_login["nometabs"].str.upper().str.contains("DEMO]", na=False)]

    login_map: dict[str, list[str]] = {}
    for _, row in df_login.iterrows():
        login_map.setdefault(row["logingroupe"], []).append(row["login"])

    created: list[str] = []
    skipped: list[str] = []

    # 2. Table login elle-même
    _register(
        f"{SERVER_PREFIX}login",
        f"Webgerest/{serveur}",
        created, skipped,
    )

    # 3. Tables par groupe
    for grp in LOGIN_GROUPS:
        grp_safe = _safe_id(grp)
        grp_path = f"Webgerest/{serveur}/{grp}"

        # descfic du groupe
        descfic_table = f"{SERVER_PREFIX}{grp_safe}_descfic"
        _register(descfic_table, grp_path, created, skipped)

        # Lire descfic pour connaître statut par table
        try:
            df_d = source.select(descfic_table)
        except Exception as e:
            logger.warning(f"  [{grp}] impossible de lire {descfic_table}: {e}")
            continue

        for _, row in df_d.iterrows():
            nomfic = str(row.get("nomfic", "")).upper()
            table_suffix = NOMFIC_TO_TABLE.get(nomfic)
            if not table_suffix:
                continue

            try:
                statut = int(row["statut"])
            except (ValueError, TypeError):
                continue

            if statut == 1:
                phys = f"{SERVER_PREFIX}{grp_safe}_{table_suffix}"
                _register(phys, grp_path, created, skipped)

            elif statut == 2:
                for site in login_map.get(grp, []):
                    phys = f"{SERVER_PREFIX}{_safe_id(site)}_{table_suffix}"
                    site_path = f"{grp_path}/{site}"
                    _register(phys, site_path, created, skipped)

    logger.info(
        f"Terminé — {len(created)} créé(s), {len(skipped)} ignoré(s) "
        f"(déjà existant ou table physique absente)"
    )

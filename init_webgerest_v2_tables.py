"""
Initialisation des tables Webgerest V2 (architecture démultipliée).

Crée et charge les tables login et descfic, puis crée toutes les tables de
données selon le statut descfic (une par login_group ou une par login_site).

Usage :
    python init_webgerest_v2_tables.py

Variables d'environnement requises (dans .env ou l'environnement) :
    WEBGEREST_SECRETS  : JSON mapping url → {client_key, client_secret}
                         ex: {"https://api.rcolcentre...": {"client_key": "...", "client_secret": "..."}}
    OVH_API_KEY        : clé OVH pour la connexion Trino
    OVH_SECRET_KEY     : secret OVH pour la connexion Trino
"""

import json
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

from data_process.jobs import run_create_webgerest_v2_tables_job

# ── Configuration ────────────────────────────────────────────────────────────

# DATASET       = "prodcentre"
# SERVER_PREFIX = "centre_"          # Préfixe commun → centre_login, centre_cd28_article…
# LOGIN_GROUPS  = ["CD28", "CD18", "CD19", "CD41", "REG-CENT"]
# BASE_URL             = "https://api.rcolcentre-internal.webgerest.fr"

DATASET       = "prodcentre"
SERVER_PREFIX = "centre_"          # Préfixe commun → centre_login, centre_cd28_article…
LOGIN_GROUPS  = ["CD28", "CD18", "CD19", "CD41", "REG-CENT"]
BASE_URL             = "https://api.rcolcentre-internal.webgerest.fr"
# ── Credentials ──────────────────────────────────────────────────────────────


_wg_secrets          = json.loads(os.environ["WEBGEREST_SECRETS"])
CLIENT_WEBGEREST     = _wg_secrets[BASE_URL]["client_key"]
SECRET_KEY_WEBGEREST = _wg_secrets[BASE_URL]["client_secret"]
OVH_API_KEY          = os.getenv("OVH_API_KEY")
OVH_SECRET_KEY       = os.getenv("OVH_SECRET_KEY")

# ── Exécution ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_create_webgerest_v2_tables_job(
        dataset=DATASET,
        server_prefix=SERVER_PREFIX,
        login_groups=LOGIN_GROUPS,
        base_url=BASE_URL,
        client_key=CLIENT_WEBGEREST,
        client_secret=SECRET_KEY_WEBGEREST,
        ovh_api_key=OVH_API_KEY,
        ovh_secret_key=OVH_SECRET_KEY,
    )
    print(result.summary())
    if not result.success:
        raise SystemExit(1)

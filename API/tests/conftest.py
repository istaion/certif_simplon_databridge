"""
Fixtures partagées pour les tests de l'API (C9).

Les identifiants de rôle (ROLE_CREDENTIALS) sont construits une seule fois,
au moment de l'import de API.app, à partir des variables d'environnement.
On ne peut donc pas se contenter de fixer des variables d'environnement dans
les tests : on patche directement le dictionnaire ROLE_CREDENTIALS après import,
ce qui fonctionne quel que soit l'environnement d'exécution (CI ou poste local),
sans dépendre de vrais secrets.

Les variables OVH_API_KEY / OVH_SECRET_KEY (et leurs variantes read-only), elles,
sont lues à la volée dans chaque route -- de simples valeurs factices suffisent
pour passer les vérifications de présence ; aucune connexion Trino réelle n'a
besoin d'aboutir pour ces tests (cf. docstring de test_predict.py).
"""
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OVH_API_KEY", "test-ovh-key")
os.environ.setdefault("OVH_SECRET_KEY", "test-ovh-secret")
os.environ.setdefault("OVH_API_KEY_READ_ONLY", "test-ovh-key-ro")
os.environ.setdefault("OVH_SECRET_KEY_READ_ONLY", "test-ovh-secret-ro")

import API.app as app_module  # noqa: E402

ADMIN_KEY, ADMIN_SECRET = "test-admin-key", "test-admin-secret"
WEBGEREST_RO_KEY, WEBGEREST_RO_SECRET = "test-webgerest-ro-key", "test-webgerest-ro-secret"
WEBRESTO_RO_KEY, WEBRESTO_RO_SECRET = "test-webresto-ro-key", "test-webresto-ro-secret"

app_module.ROLE_CREDENTIALS[app_module.Role.admin] = (ADMIN_KEY, ADMIN_SECRET)
app_module.ROLE_CREDENTIALS[app_module.Role.webgerest_readonly] = (WEBGEREST_RO_KEY, WEBGEREST_RO_SECRET)
app_module.ROLE_CREDENTIALS[app_module.Role.webresto_readonly] = (WEBRESTO_RO_KEY, WEBRESTO_RO_SECRET)


@pytest.fixture()
def client():
    with TestClient(app_module.app) as c:
        yield c
    # Isolation entre tests : jobs et tokens sont des dictionnaires en mémoire
    # partagés par toute l'app (pas de reset automatique entre requêtes).
    app_module.prediction_jobs.clear()
    app_module.active_tokens.clear()


def _get_token(client: TestClient, key: str, secret: str) -> str:
    resp = client.post("/auth/token", json={"client_key": key, "client_secret": secret})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


@pytest.fixture()
def admin_token(client):
    return _get_token(client, ADMIN_KEY, ADMIN_SECRET)


@pytest.fixture()
def webgerest_ro_token(client):
    return _get_token(client, WEBGEREST_RO_KEY, WEBGEREST_RO_SECRET)


@pytest.fixture()
def webresto_ro_token(client):
    return _get_token(client, WEBRESTO_RO_KEY, WEBRESTO_RO_SECRET)

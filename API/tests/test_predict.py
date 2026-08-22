"""
Tests des routes /predict/* de l'API DataBridge (C9 -- E3).

Portée couverte pour chaque route : succès nominal, absence/invalidité de
token (401/403), paramètres invalides (422).

Deux familles de routes, deux stratégies de test :

- Routes de LANCEMENT de job (/predict/{model_name}, /predict/all,
  /predict/global_dep_xgb) : la requête ne fait que planifier une tâche de
  fond (asyncio.create_task) et retourne immédiatement -- le job réel
  échouera en tâche de fond faute de vrais identifiants OVH, mais cet échec
  est capturé par l'application elle-même (cf. _run_prediction_job) et
  n'affecte jamais la réponse HTTP synchrone testée ici. Pas de mock requis.

- /predict/results/{uai} : la lecture Trino (get_predictions_for_site) est
  attendue (await) et son résultat renvoyé directement dans la réponse --
  elle est donc mockée, pour tester le contrat de la route indépendamment
  d'un vrai accès Trino.

Note sur les codes d'erreur d'authentification : le header X-Api-Token est
un paramètre requis (Header(...)) au niveau de FastAPI. Son absence totale
est donc rejetée par la validation FastAPI (422), avant même d'atteindre la
logique métier -- alors qu'un token présent mais invalide/inconnu déclenche
un 401 explicite (_verify_token). Les deux cas sont testés séparément avec
le code réellement observé, plutôt que de supposer les deux à 401.
"""
import pytest


# ── /predict/jobs ────────────────────────────────────────────────────────────

def test_predict_jobs_nominal(client, admin_token):
    resp = client.get("/predict/jobs", headers={"X-Api-Token": admin_token})
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_predict_jobs_missing_token_header(client):
    resp = client.get("/predict/jobs")
    assert resp.status_code == 422  # header requis, absent


def test_predict_jobs_invalid_token(client):
    resp = client.get("/predict/jobs", headers={"X-Api-Token": "not-a-real-token"})
    assert resp.status_code == 401


# ── /predict/status/{job_id} ─────────────────────────────────────────────────

def test_predict_status_nominal(client, admin_token):
    launch = client.post("/predict/arima", headers={"X-Api-Token": admin_token})
    assert launch.status_code == 200
    job_id = launch.json()["job_id"]

    resp = client.get(f"/predict/status/{job_id}", headers={"X-Api-Token": admin_token})
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == job_id
    assert body["status"] in ("running", "completed", "failed")


def test_predict_status_unknown_job_id(client, admin_token):
    resp = client.get("/predict/status/does-not-exist", headers={"X-Api-Token": admin_token})
    assert resp.status_code == 404


def test_predict_status_missing_token_header(client):
    resp = client.get("/predict/status/whatever")
    assert resp.status_code == 422


def test_predict_status_invalid_token(client):
    resp = client.get("/predict/status/whatever", headers={"X-Api-Token": "nope"})
    assert resp.status_code == 401


# ── /predict/results/{uai} ───────────────────────────────────────────────────

def test_predict_results_nominal(client, admin_token, monkeypatch):
    fake_result = [{"date": "2026-01-05", "service": "2", "prediction": 123.4}]
    monkeypatch.setattr("API.app.get_predictions_for_site", lambda *a, **k: fake_result)

    resp = client.get(
        "/predict/results/0180000X",
        params={"min_date": "2026-01-01", "max_date": "2026-01-31"},
        headers={"X-Api-Token": admin_token},
    )
    assert resp.status_code == 200
    assert resp.json() == fake_result


def test_predict_results_missing_required_dates(client, admin_token):
    resp = client.get("/predict/results/0180000X", headers={"X-Api-Token": admin_token})
    assert resp.status_code == 422


def test_predict_results_invalid_environnement_client(client, admin_token):
    resp = client.get(
        "/predict/results/0180000X",
        params={
            "min_date": "2026-01-01",
            "max_date": "2026-01-31",
            "environnement_client": "not-a-known-env",
        },
        headers={"X-Api-Token": admin_token},
    )
    assert resp.status_code == 422


def test_predict_results_missing_token_header(client):
    resp = client.get(
        "/predict/results/0180000X",
        params={"min_date": "2026-01-01", "max_date": "2026-01-31"},
    )
    assert resp.status_code == 422


def test_predict_results_invalid_token(client):
    resp = client.get(
        "/predict/results/0180000X",
        params={"min_date": "2026-01-01", "max_date": "2026-01-31"},
        headers={"X-Api-Token": "nope"},
    )
    assert resp.status_code == 401


# ── /predict/{model_name} ────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "model_name", ["arima", "prophet", "xgboost", "moving_average", "ensemble"]
)
def test_predict_model_nominal(client, admin_token, model_name):
    resp = client.post(f"/predict/{model_name}", headers={"X-Api-Token": admin_token})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_name"] == model_name
    assert body["status"] == "running"
    assert "job_id" in body


def test_predict_model_invalid_model_name(client, admin_token):
    resp = client.post("/predict/not-a-real-model", headers={"X-Api-Token": admin_token})
    assert resp.status_code == 422


def test_predict_model_wrong_role(client, webresto_ro_token):
    resp = client.post("/predict/arima", headers={"X-Api-Token": webresto_ro_token})
    assert resp.status_code == 403


def test_predict_model_missing_token_header(client):
    resp = client.post("/predict/arima")
    assert resp.status_code == 422


def test_predict_model_invalid_token(client):
    resp = client.post("/predict/arima", headers={"X-Api-Token": "nope"})
    assert resp.status_code == 401


# ── /predict/all ──────────────────────────────────────────────────────────────

def test_predict_all_nominal(client, admin_token):
    resp = client.post("/predict/all", headers={"X-Api-Token": admin_token})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_name"] == "all"
    assert body["status"] == "running"


def test_predict_all_invalid_body(client, admin_token):
    resp = client.post(
        "/predict/all",
        headers={"X-Api-Token": admin_token},
        json={"horizon_days": "not-a-number"},
    )
    assert resp.status_code == 422


def test_predict_all_wrong_role(client, webgerest_ro_token):
    resp = client.post("/predict/all", headers={"X-Api-Token": webgerest_ro_token})
    assert resp.status_code == 403


def test_predict_all_missing_token_header(client):
    resp = client.post("/predict/all")
    assert resp.status_code == 422


def test_predict_all_invalid_token(client):
    resp = client.post("/predict/all", headers={"X-Api-Token": "nope"})
    assert resp.status_code == 401


# ── /predict/global_dep_xgb ───────────────────────────────────────────────────

def test_predict_global_dep_xgb_nominal(client, admin_token):
    resp = client.post("/predict/global_dep_xgb", headers={"X-Api-Token": admin_token})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_name"] == "global_dep_xgb"
    assert body["status"] == "running"


def test_predict_global_dep_xgb_invalid_body(client, admin_token):
    resp = client.post(
        "/predict/global_dep_xgb",
        headers={"X-Api-Token": admin_token},
        json={"horizon_days": "not-a-number"},
    )
    assert resp.status_code == 422


def test_predict_global_dep_xgb_wrong_role(client, webresto_ro_token):
    resp = client.post("/predict/global_dep_xgb", headers={"X-Api-Token": webresto_ro_token})
    assert resp.status_code == 403


def test_predict_global_dep_xgb_missing_token_header(client):
    resp = client.post("/predict/global_dep_xgb")
    assert resp.status_code == 422


def test_predict_global_dep_xgb_invalid_token(client):
    resp = client.post("/predict/global_dep_xgb", headers={"X-Api-Token": "nope"})
    assert resp.status_code == 401

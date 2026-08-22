"""
Lance le job de prédiction complet (POST /predict/all) sur l'API databridge
et attend sa fin en pollant GET /predict/status/{job_id}.

Deux couches d'authentification sont nécessaires tant que le ticket ops sur
la config Caddy n'est pas résolu :
  - Basic Auth (IANORD_USERNAME/IANORD_PASSWORD) : imposée par le reverse-proxy Caddy
    devant l'API, sur toutes les routes.
  - X-Api-Token : auth applicative databridge, obtenu via POST /auth/token
    avec CLIENT_KEY/CLIENT_SECRET (rôle admin, requis par /predict/all).
"""

import logging
import time

import requests
from forepaas.core.settings import PARAMS
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)

BASE_URL             = PARAMS["API_URL"]
ENVIRONNEMENT_CLIENT = PARAMS["ENVIRONNEMENT_CLIENT"]
CLIENT_KEY           = PARAMS["CLIENT_KEY"]
CLIENT_SECRET        = PARAMS["CLIENT_SECRET"]
IANORD_USERNAME      = PARAMS["IANORD_USERNAME"]
IANORD_PASSWORD      = PARAMS["IANORD_PASSWORD"]

POLL_INTERVAL_SECONDS = 30
TIMEOUT_SECONDS       = 3 * 3600  # un run "all" enchaîne ARIMA/Prophet/XGBoost/MA/Ensemble

_CADDY_AUTH = HTTPBasicAuth(IANORD_USERNAME, IANORD_PASSWORD)


def _get_api_token() -> str:
    resp = requests.post(
        f"{BASE_URL}/auth/token",
        auth=_CADDY_AUTH,
        json={"client_key": CLIENT_KEY, "client_secret": CLIENT_SECRET},
        timeout=30,
    )
    resp.raise_for_status()
    token = resp.json().get("access_token")
    if not token:
        raise RuntimeError("access_token non reçu depuis /auth/token")
    return token


def _launch_predict_all(api_token: str) -> str:
    resp = requests.post(
        f"{BASE_URL}/predict/all",
        auth=_CADDY_AUTH,
        headers={"X-Api-Token": api_token},
        params={"environnement_client": ENVIRONNEMENT_CLIENT},
        timeout=30,
    )
    resp.raise_for_status()
    job_id = resp.json().get("job_id")
    if not job_id:
        raise RuntimeError(f"job_id non reçu depuis /predict/all : {resp.json()}")
    return job_id


def _wait_for_job(api_token: str, job_id: str) -> dict:
    deadline = time.time() + TIMEOUT_SECONDS
    while True:
        resp = requests.get(
            f"{BASE_URL}/predict/status/{job_id}",
            auth=_CADDY_AUTH,
            headers={"X-Api-Token": api_token},
            timeout=30,
        )
        resp.raise_for_status()
        job = resp.json()
        job_status = job["status"]
        logger.info(f"[predict/all] job {job_id} — statut : {job_status}")

        if job_status == "completed":
            return job
        if job_status == "failed":
            logger.error(f"[predict/all] job {job_id} en échec : {job.get('error')}")
            raise RuntimeError(f"Job {job_id} en échec : {job.get('error')}")
        if time.time() > deadline:
            logger.error(
                f"[predict/all] job {job_id} toujours en statut '{job_status}' après {TIMEOUT_SECONDS}s"
            )
            raise TimeoutError(
                f"Job {job_id} toujours en statut '{job_status}' après {TIMEOUT_SECONDS}s"
            )
        time.sleep(POLL_INTERVAL_SECONDS)


def customfunc(event):
    api_token = _get_api_token()
    job_id = _launch_predict_all(api_token)
    logger.info(f"[predict/all] job lancé : {job_id}")
    job = _wait_for_job(api_token, job_id)
    logger.info(
        f"[predict/all] job {job_id} terminé avec succès — durée : {job.get('duration_seconds')}s"
    )

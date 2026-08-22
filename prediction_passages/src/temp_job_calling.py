import logging
import time

from forepaas.core.settings import PARAMS
import requests

logger = logging.getLogger(__name__)

API_URL = PARAMS["API_URL"]
CLIENT_KEY = PARAMS["CLIENT_KEY"]
CLIENT_SECRET = PARAMS["CLIENT_SECRET"]
MODEL_NAME = "arima"
POLL_INTERVAL = 30  # secondes entre chaque vérification de statut


def _get_token() -> str:
    """Obtient un token Bearer depuis l'API."""
    response = requests.post(
        f"{API_URL}/auth/token",
        json={"client_key": CLIENT_KEY, "client_secret": CLIENT_SECRET},
    )
    response.raise_for_status()
    return response.json()["access_token"]


def _check_vars() -> None:
    """Lève une EnvironmentError si des variables obligatoires sont manquantes."""
    required = {"API_URL": API_URL, "CLIENT_KEY": CLIENT_KEY, "CLIENT_SECRET": CLIENT_SECRET}
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise EnvironmentError(f"Variables d'environnement manquantes : {missing}")


def customfunc(event):
    """Lance un job de prediction rolling via POST /predict/{model_name} et attend la fin."""
    logger.info(f"Lancement du job rolling '{MODEL_NAME}' via l'API")
    try:
        _check_vars()

        token = _get_token()
        headers = {"X-Api-Token": token}
        logger.info("Token obtenu")

        # Lancer le job (retour immédiat avec job_id)
        response = requests.post(
            f"{API_URL}/predict/{MODEL_NAME}",
            headers=headers,
            json={},
            timeout=30,
        )
        logger.info(f"Réponse API : HTTP {response.status_code}")
        response.raise_for_status()
        launch = response.json()

        job_id = launch["job_id"]
        logger.info(f"Job lancé en arrière-plan : job_id={job_id}")

        # Polling du statut jusqu'à completion
        while True:
            time.sleep(POLL_INTERVAL)

            status_resp = requests.get(
                f"{API_URL}/predict/status/{job_id}",
                headers=headers,
                timeout=30,
            )
            status_resp.raise_for_status()
            job = status_resp.json()

            job_status = job["status"]
            duration = job.get("duration_seconds", 0)
            logger.info(f"[{MODEL_NAME}] statut={job_status} | durée={duration:.0f}s")

            if job_status == "completed":
                logger.info(f"Job '{MODEL_NAME}' terminé avec succès en {duration:.0f}s")
                return

            if job_status == "failed":
                error = job.get("error", "erreur inconnue")
                raise Exception(f"Job '{MODEL_NAME}' en échec : {error}")

    except requests.HTTPError as e:
        logger.error(f"Erreur HTTP {e.response.status_code} : {e.response.text}")
        raise
    except requests.ConnectionError as e:
        logger.error(f"Impossible de joindre l'API ({API_URL}) : {e}")
        raise
    except Exception as e:
        logger.error(f"Erreur inattendue : {type(e).__name__}: {e}")
        raise

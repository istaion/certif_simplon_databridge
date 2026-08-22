import logging
import time
from typing import Any, Callable, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_RETRY_STATUSES = {502, 503, 504}
_RETRY_DELAYS   = [5, 15, 30]   # secondes entre tentatives


class WebrestoFetcher:
    """
    Classe générique pour récupérer des données depuis l'API Webresto.

    Supporte GET et POST. Pour les POST, le body est passé directement
    (construction laissée à l'appelant car il varie par endpoint).
    Un preprocess optionnel permet de transformer le JSON brut avant
    conversion en DataFrame — utile pour aplatir des objets imbriqués.

    Exemples d'usage :

        fetcher = WebrestoFetcher(base_url=BASE_URL, api_key=API_KEY)

        # GET simple
        df = fetcher.fetch_as_dataframe("/findAll/organizations")

        # POST sans preprocess
        body = {"updatedSince": "2024-01-01", "updatedBefore": "2024-05-01"}
        df = fetcher.fetch_as_dataframe("/findAll/passages", method="POST", body=body)

        # POST avec preprocess (extraction d'un champ imbriqué)
        def preprocess(items):
            return [
                {**item, "sessionId": item["session"]["id"]}
                for item in items
                if item.get("session") is not None
            ]
        df = fetcher.fetch_as_dataframe("/findAll/registrations", method="POST",
                                        body=body, preprocess=preprocess)
    """

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, endpoint: str, body: Optional[dict]) -> Any:
        url = self.base_url + endpoint
        method = method.upper()

        last_error: Exception = RuntimeError("Aucune tentative effectuée")

        for attempt, delay in enumerate([0] + _RETRY_DELAYS, start=1):
            if delay:
                logger.warning(
                    f"[{method}] {url} — nouvelle tentative {attempt}/{len(_RETRY_DELAYS) + 1} "
                    f"dans {delay}s"
                )
                time.sleep(delay)

            try:
                if method == "GET":
                    response = requests.get(
                        url, headers=self._headers, params=body or {}, timeout=120
                    )
                elif method == "POST":
                    response = requests.post(
                        url, headers=self._headers, json=body or {}, timeout=120
                    )
                else:
                    raise ValueError(f"Méthode HTTP non supportée : {method}")
            except requests.exceptions.ConnectionError as conn_err:
                # SSL EOF, connexion reset, timeout réseau…
                last_error = conn_err
                logger.warning(
                    f"[{method}] {url} — erreur de connexion (tentative {attempt}): {conn_err}"
                )
                continue

            logger.info(f"[{method}] {url} -> {response.status_code}")

            if response.status_code in (200, 201):
                try:
                    return response.json()
                except requests.exceptions.JSONDecodeError as e:
                    raise ValueError(
                        f"Réponse non-JSON de {url} (HTTP {response.status_code}, "
                        f"body={response.text!r}) : {e}"
                    ) from e

            if response.status_code in _RETRY_STATUSES:
                last_error = requests.HTTPError(
                    f"Erreur API {response.status_code} sur {url} : {response.text}",
                    response=response,
                )
                logger.warning(f"[{method}] {url} — erreur transitoire {response.status_code}")
                continue

            # Erreur non retriable (400, 401, 403, 404…)
            raise requests.HTTPError(
                f"Erreur API {response.status_code} sur {url} : {response.text}",
                response=response,
            )

        raise last_error

    def fetch(
        self,
        endpoint: str,
        method: str = "GET",
        body: Optional[dict] = None,
        preprocess: Optional[Callable[[Any], Any]] = None,
    ) -> list:
        """
        Récupère les données brutes depuis l'API.

        Args:
            endpoint:   Chemin de l'endpoint (ex: "/findAll/passages").
            method:     "GET" ou "POST".
            body:       Corps JSON de la requête POST (ignoré pour GET).
            preprocess: Callable optionnel appliqué sur le JSON brut avant retour.
                        Signature : (data: Any) -> Any
                        Typiquement utilisé pour aplatir des objets imbriqués ou
                        filtrer des éléments invalides.

        Returns:
            Liste d'éléments (après preprocess si fourni).

        Raises:
            requests.HTTPError: Si le code HTTP n'est pas 200.
            ValueError:         Si la méthode HTTP n'est pas supportée.
        """
        data = self._request(method, endpoint, body)

        if preprocess is not None:
            data = preprocess(data)

        return data

    def fetch_as_dataframe(
        self,
        endpoint: str,
        method: str = "GET",
        body: Optional[dict] = None,
        preprocess: Optional[Callable[[Any], Any]] = None,
    ) -> pd.DataFrame:
        """
        Récupère les données et les retourne sous forme de DataFrame.

        Args:
            endpoint:   Chemin de l'endpoint.
            method:     "GET" ou "POST".
            body:       Corps JSON de la requête POST.
            preprocess: Callable optionnel appliqué sur le JSON brut.

        Returns:
            pd.DataFrame construit depuis les données récupérées.
        """
        return pd.DataFrame(self.fetch(endpoint, method, body, preprocess))

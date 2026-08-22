"""
Client HTTP pour l'API Webgerest.

Authentification : GET /auth?client_id=...&client_secret=... → token JWT
Fetch : GET /{table}?LOGIN={login}&from_date={from_date}

Retry automatique 1x sur erreur 500 (avec pause 60s).

Usage :
    fetcher = WebgestFetcher(
        base_url="https://api.webgerest.example.com",
        client_key="MY_CLIENT_KEY",
        client_secret="my_secret",
    )
    df = fetcher.fetch_table("article", login="REG-CENT", from_date="2016-01-01")
"""

import logging
import threading
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

_MAX_RETRIES = 1
_RETRY_DELAY_S = 60
# Nombre max de connexions par thread — évite l'exhaustion de file descriptors
_POOL_SIZE = 2


class WebgestFetcher:
    def __init__(self, base_url: str, client_key: str, client_secret: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._client_key = client_key
        self._client_secret = client_secret
        self._local = threading.local()

    def _session(self) -> requests.Session:
        """Retourne une Session propre au thread courant avec pool borné."""
        if not hasattr(self._local, "session"):
            session = requests.Session()
            adapter = HTTPAdapter(pool_connections=_POOL_SIZE, pool_maxsize=_POOL_SIZE)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            self._local.session = session
        return self._local.session

    def _get_token(self) -> str:
        """Obtient un token JWT via GET /auth."""
        resp = self._session().get(
            f"{self.base_url}/auth",
            params={"client_id": self._client_key, "client_secret": self._client_secret},
            timeout=30,
        )
        resp.raise_for_status()
        token = resp.json().get("token")
        if not token:
            raise RuntimeError("Token non reçu depuis /auth")
        return token

    def fetch_table(
        self,
        table_name: str,
        login: str,
        from_date: str | None = None,
    ) -> pd.DataFrame | None:
        """Récupère une table Webgerest pour un login donné.

        Args:
            table_name: nom de la route API (ex: "article", "detailarticle")
            login:      login_group ou login_site selon le statut descfic
            from_date:  date de début au format "YYYY-MM-DD" (None = pas de filtre)

        Returns:
            DataFrame ou None si la réponse ne contient aucune donnée.
        """
        token = self._get_token()
        url = f"{self.base_url}/{table_name}"
        headers = {"Authorization": token}
        params: dict[str, str] = {"LOGIN": login}
        if from_date:
            params["from_date"] = from_date

        for attempt in range(_MAX_RETRIES + 1):
            resp = self._session().get(url, headers=headers, params=params, timeout=120)

            if resp.status_code == 500 and attempt < _MAX_RETRIES:
                logger.warning(
                    f"[{login}] Erreur 500 sur {table_name}, retry dans {_RETRY_DELAY_S}s "
                    f"(tentative {attempt + 1}/{_MAX_RETRIES + 1})"
                )
                time.sleep(_RETRY_DELAY_S)
                token = self._get_token()
                headers = {"Authorization": token}
                continue

            resp.raise_for_status()
            break

        json_data = resp.json()
        if not json_data:
            return None
        message = json_data.get("message") or {}
        data_list = message.get("data", []) if isinstance(message, dict) else []
        if not data_list:
            return None

        return pd.DataFrame(data_list)

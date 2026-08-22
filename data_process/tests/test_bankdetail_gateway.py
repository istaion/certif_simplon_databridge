"""
Test de non-régression -- incident E5 : la gateway WebResto a migré la route
GET/POST /findAll/bankDetails de POST (corps JSON imbriqué filterDto/options,
userId niché sous user.userId) vers GET (paramètres de requête à plat,
userId directement au niveau racine), sans prévenir.

Reproduit le comportement de la nouvelle gateway (constaté en direct sur
gateway.pprod.region-centre.ianord.fr le 2026-07-30) via un mock de
`requests.get` -- aucun appel réseau réel ni dépendance à des identifiants.

Sans le correctif (method="POST" + corps imbriqué dans data_process/jobs.py,
et préprocess attendant `item["user"]["userId"]`), ce test échoue de deux
façons possibles selon ce qui est cassé :
  - `test_fetch_bankdetail_uses_get_with_flat_params` échoue si le job est
    remis en POST (assertion sur la méthode/les paramètres HTTP envoyés) ;
  - `test_preprocess_bankdetail_*` échoue si le préprocess revient à lire
    `item["user"]["userId"]` au lieu du champ plat `item["userId"]`.
"""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_process.fetch.fetch_webresto import WebrestoFetcher
from data_process.process.users import preprocess_bankdetail, transform_bankdetail

GATEWAY_URL = "https://gateway.pprod.region-centre.ianord.fr/data-lake"


def _fake_gateway_response(status_code=200, payload=None):
    resp = MagicMock(status_code=status_code)
    resp.json.return_value = payload if payload is not None else []
    return resp


def test_fetch_bankdetail_uses_get_with_flat_params():
    """La route a migré de POST (corps JSON imbriqué) à GET (params à plat)."""
    fetcher = WebrestoFetcher(base_url=GATEWAY_URL, api_key="test-key")

    with patch("requests.get", return_value=_fake_gateway_response(payload=[])) as mock_get, \
         patch("requests.post") as mock_post:
        fetcher.fetch_as_dataframe(
            "/findAll/bankDetails",
            method="GET",
            body={
                "updatedSince": "2026-01-01",
                "updatedBefore": "2026-01-31",
                "selects": "createdAt, updatedAt, deletedAt, bankDetailId, trancheId, choiceBankDetails, userId",
            },
        )

    mock_post.assert_not_called()
    mock_get.assert_called_once()
    _, kwargs = mock_get.call_args
    assert kwargs["params"]["updatedSince"] == "2026-01-01"
    assert kwargs["params"]["updatedBefore"] == "2026-01-31"
    assert "userId" in kwargs["params"]["selects"]
    # L'ancien format imbriqué ne doit plus être envoyé.
    assert "filterDto" not in kwargs["params"]
    assert "options" not in kwargs["params"]


def test_fetch_bankdetail_old_post_contract_now_returns_404():
    """Documente le comportement observé de l'incident : l'ancienne route POST
    répond 404 sur la nouvelle gateway -- un job qui repartirait en POST
    échouerait immédiatement (visible en monitoring via le taux d'erreurs HTTP)."""
    fetcher = WebrestoFetcher(base_url=GATEWAY_URL, api_key="test-key")
    error_body = '{"message":"Cannot POST /data-lake/findAll/bankDetails","error":"Not Found","statusCode":404}'
    resp_404 = MagicMock(status_code=404, text=error_body)

    with patch("requests.post", return_value=resp_404):
        with pytest.raises(Exception):
            fetcher.fetch_as_dataframe(
                "/findAll/bankDetails",
                method="POST",
                body={"filterDto": {"updatedSince": "2026-01-01", "updatedBefore": "2026-01-31"}},
            )


def test_preprocess_bankdetail_flat_userid_new_gateway_shape():
    """Nouvelle gateway (GET) : userId est un champ plat au niveau racine."""
    items = [
        {"bankDetailId": 1, "userId": 42, "createdAt": "2026-01-01T00:00:00Z",
         "updatedAt": "2026-01-02T00:00:00Z", "deletedAt": None,
         "choiceBankDetails": "IBAN", "trancheId": 4},
    ]
    warnings = []
    clean = preprocess_bankdetail(items, warnings)

    assert len(clean) == 1
    assert clean[0]["id_user"] == 42
    assert warnings == []


def test_preprocess_bankdetail_missing_userid_is_filtered_with_warning():
    """Un item sans userId (compte supprimé côté gateway) est ignoré, pas planté."""
    items = [{"bankDetailId": 2, "userId": None}]
    warnings = []
    clean = preprocess_bankdetail(items, warnings)

    assert clean == []
    assert len(warnings) == 1
    assert "2" in warnings[0]


def test_transform_bankdetail_end_to_end_matches_expected_schema():
    """Fetch (GET, mocké) -> preprocess -> transform : vérifie le schéma final,
    identique à celui vérifié manuellement contre la vraie gateway pprod."""
    fake_items = [
        {"bankDetailId": 49658, "userId": 49264, "createdAt": "2024-12-19T10:30:36.422Z",
         "updatedAt": "2026-01-29T09:17:05.849Z", "deletedAt": None,
         "choiceBankDetails": None, "trancheId": 4},
    ]
    fetcher = WebrestoFetcher(base_url=GATEWAY_URL, api_key="test-key")
    warnings = []

    with patch("requests.get", return_value=_fake_gateway_response(payload=fake_items)):
        df = fetcher.fetch_as_dataframe(
            "/findAll/bankDetails",
            method="GET",
            body={
                "updatedSince": "2026-01-01", "updatedBefore": "2026-01-31",
                "selects": "createdAt, updatedAt, deletedAt, bankDetailId, trancheId, choiceBankDetails, userId",
            },
            preprocess=lambda items: preprocess_bankdetail(items, warnings),
        )

    result = transform_bankdetail(df, "prodcentre")

    assert warnings == []
    assert list(result.columns) == [
        "bank_detail_id", "created_at", "updated_at", "id_user",
        "choice_bank_details", "id_tranche",
    ]
    assert result.iloc[0]["bank_detail_id"] == 49658
    assert result.iloc[0]["id_user"] == 49264
    assert result.iloc[0]["id_tranche"] == 4

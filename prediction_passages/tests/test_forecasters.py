"""
Tests des forecasters individuels (C12) :
  - non-régression des métriques (MAE/MAPE) sur un jeu de données synthétique fixe
  - comportement sur des cas limites : établissement avec peu de données,
    valeurs manquantes, établissement absent du jeu d'entraînement (repli attendu)

Le jeu de données synthétique est déterministe (graine fixée dans conftest.py) :
les seuils de non-régression ci-dessous sont calés sur son niveau de bruit
connu, avec une marge -- ils ne visent pas une performance absolue, mais à
détecter une régression du pipeline (ex. feature cassée, mauvais encodage).
"""
import numpy as np
import pandas as pd
import pytest

from prediction_passages.src.arima_forecast import ARIMAForecaster
from prediction_passages.src.mov_avg_forecast import MovingAverageForecaster
from prediction_passages.src.prophet_forecast import ProphetForecaster
from prediction_passages.src.xgb_forecast import XGBoostForecaster


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def _mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


@pytest.fixture(scope="module")
def train_test_split(request):
    """
    Fixture "module" : la préparation des données (mock Trino + pipeline complet)
    ne dépend d'aucun état modifié par les tests, on la calcule une seule fois
    pour toute la session de tests de ce module (les fits Prophet/XGBoost sont
    les étapes les plus coûteuses de cette suite).
    """
    monkeypatch = pytest.MonkeyPatch()
    request.addfinalizer(monkeypatch.undo)

    from conftest import build_data_preparation, make_synthetic_effect_df, make_synthetic_tables

    effect_df = make_synthetic_effect_df()
    tables = make_synthetic_tables(effect_df)
    dp = build_data_preparation(monkeypatch, tables, use_manual_entry=True)
    df = dp.load_and_prepare()
    return dp.train_test_split_by_date(df, test_days=30)


# ── Non-régression des métriques ────────────────────────────────────────────

def test_arima_non_regression(train_test_split):
    train, test = train_test_split
    model = ARIMAForecaster().fit(train)
    preds = model.predict(test)

    assert np.isfinite(preds).all()
    mae = _mae(test["efreel"], preds)
    mape = _mape(test["efreel"], preds)
    assert mae < 25, f"MAE ARIMA = {mae:.1f} (seuil de non-régression : 25)"
    assert mape < 30, f"MAPE ARIMA = {mape:.1f}% (seuil de non-régression : 30%)"


def test_prophet_non_regression(train_test_split):
    train, test = train_test_split
    model = ProphetForecaster(horizon_variant="21").fit(train)
    preds = model.predict(test)

    assert np.isfinite(preds).all()
    mae = _mae(test["efreel"], preds)
    mape = _mape(test["efreel"], preds)
    assert mae < 25, f"MAE Prophet = {mae:.1f} (seuil de non-régression : 25)"
    assert mape < 30, f"MAPE Prophet = {mape:.1f}% (seuil de non-régression : 30%)"


def test_xgboost_non_regression(train_test_split):
    train, test = train_test_split
    model = XGBoostForecaster(horizon_variant="21").fit(train)
    preds = model.predict(test)

    assert np.isfinite(preds).all()
    mae = _mae(test["efreel"], preds)
    mape = _mape(test["efreel"], preds)
    assert mae < 25, f"MAE XGBoost = {mae:.1f} (seuil de non-régression : 25)"
    assert mape < 30, f"MAPE XGBoost = {mape:.1f}% (seuil de non-régression : 30%)"


def test_moving_average_non_regression(train_test_split):
    train, test = train_test_split
    model = MovingAverageForecaster().fit(train)
    preds = model.predict(test)

    assert np.isfinite(preds).all()
    mae = _mae(test["efreel"], preds)
    mape = _mape(test["efreel"], preds)
    assert mae < 25, f"MAE MovingAverage = {mae:.1f} (seuil de non-régression : 25)"
    assert mape < 30, f"MAPE MovingAverage = {mape:.1f}% (seuil de non-régression : 30%)"


# ── Cas limite : établissement avec peu de données ──────────────────────────

def test_arima_handles_sparse_site(train_test_split):
    train, test = train_test_split
    sparse_site = "SITE_SPARSE"
    sparse_rows = train[train["login_site"] == train["login_site"].iloc[0]].head(3).copy()
    sparse_rows["login_site"] = sparse_site
    train_sparse = pd.concat([train, sparse_rows], ignore_index=True)

    model = ARIMAForecaster().fit(train_sparse)
    test_sparse = test[test["login_site"] == test["login_site"].iloc[0]].head(3).copy()
    test_sparse["login_site"] = sparse_site

    preds = model.predict(test_sparse)
    assert np.isfinite(preds).all()
    assert (preds >= 0).all()


def test_moving_average_handles_sparse_site(train_test_split):
    train, test = train_test_split
    sparse_site = "SITE_SPARSE_MA"
    sparse_rows = train[train["login_site"] == train["login_site"].iloc[0]].head(2).copy()
    sparse_rows["login_site"] = sparse_site
    train_sparse = pd.concat([train, sparse_rows], ignore_index=True)

    model = MovingAverageForecaster().fit(train_sparse)
    test_sparse = test[test["login_site"] == test["login_site"].iloc[0]].head(3).copy()
    test_sparse["login_site"] = sparse_site

    preds = model.predict(test_sparse)
    # Moins de 8 observations positives -> repli sur weekday_mean / moyenne globale (pas de NaN)
    assert np.isfinite(preds).all()


# ── Cas limite : valeurs manquantes dans les régresseurs ────────────────────

def test_prophet_handles_missing_regressor_values(train_test_split):
    train, test = train_test_split
    model = ProphetForecaster(horizon_variant="21").fit(train)

    test_with_nan = test.copy()
    test_with_nan.loc[test_with_nan.index[:5], "jours_avant_vacance"] = np.nan

    preds = model.predict(test_with_nan)
    assert np.isfinite(preds).all()


def test_xgboost_handles_missing_regressor_values(train_test_split):
    train, test = train_test_split
    model = XGBoostForecaster(horizon_variant="21").fit(train)

    test_with_nan = test.copy()
    test_with_nan.loc[test_with_nan.index[:5], "ips"] = np.nan

    preds = model.predict(test_with_nan)
    assert np.isfinite(preds).all()


# ── Cas limite : établissement absent du jeu d'entraînement (repli attendu) ─

def test_arima_unknown_site_falls_back(train_test_split):
    train, test = train_test_split
    model = ARIMAForecaster().fit(train)

    unknown = test[test["login_site"] == test["login_site"].iloc[0]].head(3).copy()
    unknown["login_site"] = "SITE_JAMAIS_VU"

    preds = model.predict(unknown)
    assert np.isfinite(preds).all()
    # Aucune moyenne connue pour ce site -> repli sur 0.0 (cf. ARIMAForecaster.predict)
    assert (preds == 0.0).all()


def test_prophet_unknown_site_falls_back(train_test_split):
    train, test = train_test_split
    model = ProphetForecaster(horizon_variant="21").fit(train)

    unknown = test[test["login_site"] == test["login_site"].iloc[0]].head(3).copy()
    unknown["login_site"] = "SITE_JAMAIS_VU"

    preds = model.predict(unknown)
    assert np.isfinite(preds).all()
    assert (preds == 0.0).all()


def test_xgboost_unknown_site_does_not_crash(train_test_split):
    train, test = train_test_split
    model = XGBoostForecaster(horizon_variant="21").fit(train)

    unknown = test[test["login_site"] == test["login_site"].iloc[0]].head(3).copy()
    unknown["login_site"] = "SITE_JAMAIS_VU"

    # Le target encoder impute la moyenne globale pour une catégorie inconnue
    # (pas de KeyError/ValueError attendu).
    preds = model.predict(unknown)
    assert np.isfinite(preds).all()


def test_moving_average_unknown_site_falls_back(train_test_split):
    train, test = train_test_split
    model = MovingAverageForecaster().fit(train)

    unknown = test[test["login_site"] == test["login_site"].iloc[0]].head(3).copy()
    unknown["login_site"] = "SITE_JAMAIS_VU"

    preds = model.predict(unknown)
    # Repli sur weekday_mean (colonne présente sur les lignes de test) puis moyenne globale
    assert np.isfinite(preds).all()

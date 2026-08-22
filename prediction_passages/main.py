import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Tuple, Dict, List, Optional

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

from scipy.optimize import minimize
from src.data_prep import DataPreparation
from src.ensemble_forecast import EnsembleForecaster
from src.ensemble_from_store import EnsembleFromStore, MODELS as ENSEMBLE_MODELS
from src.grouped_ensemble_forecast import GroupedEnsembleForecaster
from src.statistics import (generate_site_report,
                            get_metrics_by_site,
                            print_summary_report,
                            plot_site_predictions,
                            plot_metrics_heatmap,
                            plot_mape_vs_volume,
                            plot_best_model_by_site,
                            plot_error_by_weekday_and_site,
                            plot_mase_boxplot_by_model)
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import os 
from dotenv import load_dotenv

load_dotenv()

def load_predictions_from_trino(ovh_api_key: str, ovh_secret_key: str) -> pd.DataFrame:
    """
    Charge les prédictions depuis passage_predict (Trino) et les pivote
    au format wide attendu par les fonctions d'analyse.

    Pour chaque (target_date, uai, service, model), garde la prédiction
    avec le plus petit horizon (prédiction la plus récente).

    Returns:
        DataFrame avec colonnes : login_site, efdate, efreel, day,
        pred_ARIMA, pred_Prophet21, pred_Prophet35,
        pred_XGBoost21, pred_XGBoost35, pred_MovingAverage, pred_Ensemble
    """
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data_process.db.trino_client import TrinoClient

    db = TrinoClient("default_dataset", ovh_api_key, ovh_secret_key)

    sql = """
        SELECT target_date, uai, service, model, prediction, effectif_reel
        FROM (
            SELECT
                target_date, uai, service, model, prediction, effectif_reel,
                ROW_NUMBER() OVER (
                    PARTITION BY target_date, uai, service, model
                    ORDER BY horizon ASC
                ) AS rn
            FROM passage_predict
            WHERE effectif_reel IS NOT NULL
              AND effectif_reel > 0
              AND prediction IS NOT NULL
        )
        WHERE rn = 1
        ORDER BY target_date, uai, service, model
    """
    print("Requête Trino en cours…")
    df = db.query_as_dataframe(sql)

    if df.empty:
        raise ValueError("Aucune donnée dans passage_predict avec effectif_reel renseigné.")

    df["target_date"] = pd.to_datetime(df["target_date"])
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df["effectif_reel"] = pd.to_numeric(df["effectif_reel"], errors="coerce")

    # Pivot : une colonne pred_{model} par modèle
    pivot = df.pivot_table(
        index=["uai", "target_date", "service"],
        columns="model",
        values="prediction",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    for col in pivot.columns:
        if col not in ("uai", "target_date", "service"):
            pivot.rename(columns={col: f"pred_{col}"}, inplace=True)

    # Joindre effectif_reel (même valeur pour tous les modèles)
    efreel = (
        df.groupby(["uai", "target_date", "service"])["effectif_reel"]
        .first()
        .reset_index()
    )
    pivot = pivot.merge(efreel, on=["uai", "target_date", "service"], how="left")

    # Renommage pour compatibilité avec les fonctions d'analyse existantes
    pivot.rename(columns={
        "uai": "login_site",
        "target_date": "efdate",
        "effectif_reel": "efreel",
    }, inplace=True)
    pivot["day"] = pivot["efdate"].dt.weekday

    print(f"  {len(pivot)} créneaux, {pivot['login_site'].nunique()} sites, "
          f"{pivot['efdate'].min().date()} → {pivot['efdate'].max().date()}")
    return pivot


def analyses(ovh_api_key: str | None = None, ovh_secret_key: str | None = None):
    """
    Lance toutes les analyses et génère les graphiques.

    Si ovh_api_key et ovh_secret_key sont fournis, charge les données
    depuis Trino (passage_predict). Sinon, lit pred_stats/predictions_ensemble.csv.
    """
    output_dir = "pred_stats/analyse"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Chargement des données ────────────────────────────────────────────────
    if ovh_api_key and ovh_secret_key:
        results_df = load_predictions_from_trino(ovh_api_key, ovh_secret_key)
    else:
        data_path = "pred_stats/predictions_ensemble.csv"
        print("Chargement des données depuis CSV...")
        results_df = pd.read_csv(data_path)
        results_df['efdate'] = pd.to_datetime(results_df['efdate'])

    print(f"Données chargées: {len(results_df)} lignes, {results_df['login_site'].nunique()} sites")
    
    # Calculer les métriques par site
    print("\nCalcul des métriques par site...")
    metrics_df = get_metrics_by_site(results_df)
    
    # Afficher le rapport console
    print_summary_report(metrics_df, results_df)
    
    # Générer tous les graphiques
    print("\n" + "=" * 80)
    print("GÉNÉRATION DES GRAPHIQUES")
    print("=" * 80)
    
    print("\n1. Graphique des prédictions par site...")
    plot_site_predictions(results_df, output_dir)
    
    print("\n2. Heatmap des MAE...")
    plot_metrics_heatmap(metrics_df, output_dir)
    
    print("\n3. MAPE vs Volume...")
    plot_mape_vs_volume(metrics_df, output_dir)
    
    print("\n4. Meilleur modèle par site...")
    plot_best_model_by_site(metrics_df, output_dir)
    
    print("\n5. Erreurs par jour de semaine...")
    plot_error_by_weekday_and_site(results_df, output_dir)

    print("\n6. Boîtes à moustaches du MASE par modèle...")
    plot_mase_boxplot_by_model(results_df, output_dir, naive_column='pred_MovingAverage')

    print("\n7. Rapport CSV par site...")
    generate_site_report(metrics_df, output_dir)
    
    print("\n" + "=" * 80)
    print("TERMINÉ!")
    print("=" * 80)
    
    return metrics_df, results_df

def predictions(use_grouped: bool = True):
    """
    Entraîne et évalue les modèles de prédiction.

    Args:
        use_grouped: Si True, utilise GroupedEnsembleForecaster (un ensemble par groupe de taille).
                     Si False, utilise EnsembleForecaster global.
    """
    data_prep = DataPreparation(
        use_manual_entry=False
    ) # , [1, 2, 3, 7, 9, 14, 15, 16, 27, 33, 34, 36, 37, 38, 42, 49, 59, 60, 62]
    df = data_prep.load_and_prepare()
    print(f"Données: {len(df)} lignes, {df['login_site'].nunique()} sites")

    train_df, test_df = data_prep.train_test_split_by_date(df, test_days=30)

    ensemble_params = {
        'horizon_bins': [7, 14, 21],
        'min_samples_per_cell': 3,
        'regularization_strength': 0.3,
        'mean_floor': 50.0,
        'exclude_mape_threshold': 500.0,
        'val_days': 30,
        'n_splits': 4,
        'naive_column': 'daily_mobile_mean_8_shifted_4',
        'optimization_metric': 'mase'
    }

    if use_grouped:
        print("\n>>> Utilisation du GroupedEnsembleForecaster (un ensemble par groupe de taille)")
        ensemble = GroupedEnsembleForecaster(
            group_column='group_size',
            **ensemble_params
        )
    else:
        print("\n>>> Utilisation du EnsembleForecaster global")
        ensemble = EnsembleForecaster(**ensemble_params)

    ensemble.fit(train_df, optimize_weights=True)

    preds = ensemble.predict(test_df)
    results = ensemble.evaluate(test_df, preds)

    print("\n" + "=" * 60)
    print("RÉSULTATS GLOBAUX")
    print("=" * 60)
    print(results.to_string(index=False))

    # Résultats par groupe si utilisation du GroupedEnsembleForecaster
    if use_grouped and hasattr(ensemble, 'evaluate_by_group'):
        print("\n" + "=" * 60)
        print("RÉSULTATS PAR GROUPE")
        print("=" * 60)
        results_by_group = ensemble.evaluate_by_group(test_df, preds)
        # Afficher uniquement l'Ensemble pour chaque groupe
        ensemble_by_group = results_by_group[results_by_group['Modèle'] == 'Ensemble']
        print(ensemble_by_group.to_string(index=False))

    results_df = test_df.copy()
    for name, pred_values in preds.items():
        results_df[f'pred_{name}'] = pred_values

    results_df.to_csv("pred_stats/predictions_ensemble.csv", index=False)

    return ensemble, preds, results

def run_arima_rolling(
    horizon_days: int = 30,
    step_days: int = 7,
    min_train_weeks: int = 10,
    env: str | None = None,
    prefix: str | None = None,
    df: pd.DataFrame | None = None,
    data_prep: DataPreparation | None = None,
    force_start_date: str | None = None,
):
    """
    Job de prédiction ARIMA rolling. Reprend automatiquement depuis le dernier
    prediction_date enregistré dans passage_predict.

    À chaque pas (step_days) :
      - prédit les horizon_days jours suivants le cutoff
      - remplit effectif_reel pour les dates déjà connues dans l'historique
      - laisse effectif_reel = NULL pour les dates futures
      - met à jour ARIMA incrémentalement via append() (sans ré-estimation complète)

    Premier lancement : démarre min_train_weeks semaines après la première observation.

    Args:
        horizon_days: fenêtre de prédiction après chaque cutoff (défaut 30 j)
        step_days: pas d'avancement entre deux cutoffs (défaut 7 j = 1 semaine)
        min_train_weeks: nombre de semaines minimum avant le premier cutoff (défaut 10)
    """
    import os
    from src.arima_forecast import ARIMAForecaster
    from src.trino_store import (
        build_future_df, get_last_prediction_date, store_predictions, delete_predictions_since,
        passage_predict_table,
    )

    ovh_api_key = os.getenv("OVH_API_KEY")
    ovh_secret_key = os.getenv("OVH_SECRET_KEY")

    # 1. Chargement des données (réutilise le df partagé si fourni)
    if df is None or data_prep is None:
        data_prep = DataPreparation(
            use_manual_entry=True,
            env=env,
            prefix=prefix,
        )
        df = data_prep.load_and_prepare()
    print(f"Données : {len(df)} lignes, {df['login_site'].nunique()} sites")

    table_name = passage_predict_table(data_prep._prefix)
    today = pd.Timestamp.today().normalize()

    # 2. Déterminer le cutoff de départ (reprise ou premier run)
    if force_start_date:
        delete_predictions_since(ovh_api_key, ovh_secret_key, 'ARIMA', force_start_date,
                                 table_name=table_name)
        cutoff = pd.Timestamp(force_start_date)
        print(f"Force restart depuis {cutoff.date()}")
    else:
        last_pred_date = get_last_prediction_date(ovh_api_key, ovh_secret_key, model_name='ARIMA',
                                                  table_name=table_name)
        if last_pred_date is None:
            cutoff = df['efdate'].min() + timedelta(weeks=min_train_weeks)
            print(f"Premier run — cutoff initial : {cutoff.date()}")
        else:
            cutoff = pd.Timestamp(last_pred_date) + timedelta(days=step_days)
            print(f"Reprise depuis {last_pred_date} — prochain cutoff : {cutoff.date()}")

    if cutoff > today + timedelta(days=horizon_days):
        print("Prédictions déjà à jour, rien à faire.")
        return

    min_obs = min_train_weeks * 5  # ~5 jours ouverts par semaine

    # 3. État du modèle : suivi des sites déjà entraînés
    arima = ARIMAForecaster()
    known_sites: set = set()

    # 4. Boucle rolling
    steps_done = 0
    while cutoff <= today + timedelta(days=horizon_days):

        # Sites ayant assez de données au cutoff courant
        train_step = df[df['efdate'] <= cutoff]
        site_counts = train_step.groupby('login_site').size()
        valid_sites_now = set(site_counts[site_counts >= min_obs].index)

        # Nouveaux sites franchissant le seuil → refit complet du modèle
        new_sites = valid_sites_now - known_sites
        if new_sites:
            print(f"  +{len(new_sites)} site(s) → refit ARIMA ({len(valid_sites_now)} sites total)")
            arima = ARIMAForecaster()
            arima.fit(train_step[train_step['login_site'].isin(valid_sites_now)])
            known_sites = valid_sites_now

        if not known_sites:
            cutoff += timedelta(days=step_days)
            continue

        print(f"\n--- cutoff : {cutoff.date()} | {len(known_sites)} sites | pas {steps_done + 1} ---")

        # Construire le DataFrame de prédiction (horizon_days jours après cutoff)
        future_df = build_future_df(
            df_history=train_step[train_step['login_site'].isin(known_sites)],
            cutoff_date=cutoff,
            horizon_days=horizon_days,
            vacances_df=data_prep._vacances_df,
            jours_feries_df=data_prep._jours_feries_df,
            fallback_zone=data_prep._fallback_zone,
        )

        if future_df.empty:
            cutoff += timedelta(days=step_days)
            continue

        # Remplir effectif_reel pour les dates déjà connues dans l'historique
        known = df.groupby(['login_site', 'efdate', 'codss2'], as_index=False)['efreel'].sum()
        future_df = future_df.merge(
            known.rename(columns={'efreel': 'efreel_known'}),
            on=['login_site', 'efdate', 'codss2'],
            how='left',
        )
        future_df['efreel'] = future_df['efreel_known'].combine_first(future_df['efreel'])
        future_df = future_df.drop(columns=['efreel_known'])

        # Prédictions ARIMA
        preds = arima.predict(future_df)

        # Stockage dans Trino
        store_predictions(
            test_df=future_df,
            predictions={'ARIMA': preds},
            prediction_date=cutoff.date(),
            ovh_api_key=ovh_api_key,
            ovh_secret_key=ovh_secret_key,
            table_name=table_name,
        )
        steps_done += 1

        # Avancer le cutoff et mettre à jour ARIMA incrémentalement
        next_cutoff = cutoff + timedelta(days=step_days)
        new_week = df[
            (df['efdate'] > cutoff) &
            (df['efdate'] <= next_cutoff) &
            (df['login_site'].isin(known_sites))
        ]
        if not new_week.empty:
            arima.update(new_week)

        cutoff = next_cutoff

    print(f"\nTerminé : {steps_done} pas de {step_days} jours générés.")


def run_prophet_rolling(
    variant: str = '21',
    horizon_days: int = 30,
    step_days: int = 7,
    min_train_weeks: int = 10,
    env: str | None = None,
    prefix: str | None = None,
    df: pd.DataFrame | None = None,
    data_prep: DataPreparation | None = None,
    force_start_date: str | None = None,
):
    """
    Job de prédiction Prophet rolling.

    variant='21' → Prophet21 (mobile_mean_42_shifted_21 + daily_mobile_mean_8_shifted_4)
    variant='35' → Prophet35 (mobile_mean_42_shifted_35 + daily_mobile_mean_8_shifted_6)

    Reprend automatiquement depuis le dernier prediction_date de 'Prophet{variant}'
    dans passage_predict. Refait un fit complet à chaque pas (Prophet n'a pas de mise
    à jour incrémentale).

    Args:
        variant: '21' ou '35'
        horizon_days: fenêtre de prédiction après chaque cutoff (défaut 30 j)
        step_days: pas d'avancement entre deux cutoffs (défaut 7 j = 1 semaine)
        min_train_weeks: nombre de semaines minimum avant le premier cutoff (défaut 10)
    """
    import os
    from src.prophet_forecast import ProphetForecaster
    from src.trino_store import (
        build_future_df, get_last_prediction_date, store_predictions, delete_predictions_since,
        passage_predict_table,
    )

    model_name = f'Prophet{variant}'
    ovh_api_key = os.getenv("OVH_API_KEY")
    ovh_secret_key = os.getenv("OVH_SECRET_KEY")
    # Plafonner l'horizon au nombre de jours correspondant au variant (21 ou 35)
    horizon_days = min(horizon_days, int(variant))

    # 1. Chargement des données (réutilise le df partagé si fourni)
    if df is None or data_prep is None:
        data_prep = DataPreparation(
            use_manual_entry=True,
            env=env,
            prefix=prefix,
        )
        df = data_prep.load_and_prepare()
    print(f"Données : {len(df)} lignes, {df['login_site'].nunique()} sites")

    table_name = passage_predict_table(data_prep._prefix)
    today = pd.Timestamp.today().normalize()

    # 2. Déterminer le cutoff de départ (reprise ou premier run)
    if force_start_date:
        delete_predictions_since(ovh_api_key, ovh_secret_key, model_name, force_start_date,
                                 table_name=table_name)
        cutoff = pd.Timestamp(force_start_date)
        print(f"Force restart depuis {cutoff.date()}")
    else:
        last_pred_date = get_last_prediction_date(ovh_api_key, ovh_secret_key, model_name=model_name,
                                                  table_name=table_name)
        if last_pred_date is None:
            cutoff = df['efdate'].min() + timedelta(weeks=min_train_weeks)
            print(f"Premier run — cutoff initial : {cutoff.date()}")
        else:
            cutoff = pd.Timestamp(last_pred_date) + timedelta(days=step_days)
            print(f"Reprise depuis {last_pred_date} — prochain cutoff : {cutoff.date()}")

    if cutoff > today + timedelta(days=horizon_days):
        print("Prédictions déjà à jour, rien à faire.")
        return

    min_obs = min_train_weeks * 5  # ~5 jours ouverts par semaine

    # 3. Boucle rolling (refit complet à chaque pas)
    steps_done = 0
    while cutoff <= today + timedelta(days=horizon_days):

        train_step = df[df['efdate'] <= cutoff]
        site_counts = train_step.groupby('login_site').size()
        valid_sites = set(site_counts[site_counts >= min_obs].index)

        if not valid_sites:
            cutoff += timedelta(days=step_days)
            continue

        print(f"\n--- cutoff : {cutoff.date()} | {len(valid_sites)} sites | pas {steps_done + 1} ---")

        # Refit complet Prophet à chaque pas
        prophet = ProphetForecaster(horizon_variant=variant)
        prophet.fit(train_step[train_step['login_site'].isin(valid_sites)])

        # Construire le DataFrame de prédiction
        future_df = build_future_df(
            df_history=train_step[train_step['login_site'].isin(valid_sites)],
            cutoff_date=cutoff,
            horizon_days=horizon_days,
            vacances_df=data_prep._vacances_df,
            jours_feries_df=data_prep._jours_feries_df,
            fallback_zone=data_prep._fallback_zone,
        )

        if future_df.empty:
            cutoff += timedelta(days=step_days)
            continue

        # Remplir effectif_reel pour les dates déjà connues
        known = df.groupby(['login_site', 'efdate', 'codss2'], as_index=False)['efreel'].sum()
        future_df = future_df.merge(
            known.rename(columns={'efreel': 'efreel_known'}),
            on=['login_site', 'efdate', 'codss2'],
            how='left',
        )
        future_df['efreel'] = future_df['efreel_known'].combine_first(future_df['efreel'])
        future_df = future_df.drop(columns=['efreel_known'])

        # Prédictions Prophet
        preds = prophet.predict(future_df)

        # Stockage dans Trino
        store_predictions(
            test_df=future_df,
            predictions={model_name: preds},
            prediction_date=cutoff.date(),
            ovh_api_key=ovh_api_key,
            ovh_secret_key=ovh_secret_key,
            table_name=table_name,
        )
        steps_done += 1

        cutoff += timedelta(days=step_days)

    print(f"\nTerminé : {steps_done} pas de {step_days} jours générés.")


def run_xgb_rolling(
    variant: str = '21',
    horizon_days: int = 30,
    step_days: int = 7,
    min_train_weeks: int = 10,
    env: str | None = None,
    prefix: str | None = None,
    df: pd.DataFrame | None = None,
    data_prep: DataPreparation | None = None,
    force_start_date: str | None = None,
):
    """
    Job de prédiction XGBoost rolling.

    variant='21' → XGBoost21 (mobile_mean_42_shifted_21 + daily_mobile_mean_8_shifted_4)
    variant='35' → XGBoost35 (mobile_mean_42_shifted_35 + daily_mobile_mean_8_shifted_6)

    Reprend automatiquement depuis le dernier prediction_date de 'XGBoost{variant}'
    dans passage_predict. Refait un fit complet à chaque pas.

    Args:
        variant: '21' ou '35'
        horizon_days: fenêtre de prédiction après chaque cutoff (défaut 30 j)
        step_days: pas d'avancement entre deux cutoffs (défaut 7 j)
        min_train_weeks: nombre de semaines minimum avant le premier cutoff (défaut 10)
    """
    import os
    from src.xgb_forecast import XGBoostForecaster
    from src.trino_store import (
        build_future_df, get_last_prediction_date, store_predictions, delete_predictions_since,
        passage_predict_table,
    )

    model_name = f'XGBoost{variant}'
    ovh_api_key = os.getenv("OVH_API_KEY")
    ovh_secret_key = os.getenv("OVH_SECRET_KEY")
    # Plafonner l'horizon au nombre de jours correspondant au variant (21 ou 35)
    horizon_days = min(horizon_days, int(variant))

    # 1. Chargement des données (réutilise le df partagé si fourni)
    if df is None or data_prep is None:
        data_prep = DataPreparation(
            use_manual_entry=True,
            env=env,
            prefix=prefix,
        )
        df = data_prep.load_and_prepare()
    print(f"Données : {len(df)} lignes, {df['login_site'].nunique()} sites")

    table_name = passage_predict_table(data_prep._prefix)
    today = pd.Timestamp.today().normalize()

    # 2. Déterminer le cutoff de départ
    if force_start_date:
        delete_predictions_since(ovh_api_key, ovh_secret_key, model_name, force_start_date,
                                 table_name=table_name)
        cutoff = pd.Timestamp(force_start_date)
        print(f"Force restart depuis {cutoff.date()}")
    else:
        last_pred_date = get_last_prediction_date(ovh_api_key, ovh_secret_key, model_name=model_name,
                                                  table_name=table_name)
        if last_pred_date is None:
            cutoff = df['efdate'].min() + timedelta(weeks=min_train_weeks)
            print(f"Premier run — cutoff initial : {cutoff.date()}")
        else:
            cutoff = pd.Timestamp(last_pred_date) + timedelta(days=step_days)
            print(f"Reprise depuis {last_pred_date} — prochain cutoff : {cutoff.date()}")

    if cutoff > today + timedelta(days=horizon_days):
        print("Prédictions déjà à jour, rien à faire.")
        return

    min_obs = min_train_weeks * 5

    # 3. Boucle rolling (refit complet à chaque pas)
    steps_done = 0
    while cutoff <= today + timedelta(days=horizon_days):

        train_step = df[df['efdate'] <= cutoff]
        site_counts = train_step.groupby('login_site').size()
        valid_sites = set(site_counts[site_counts >= min_obs].index)

        if not valid_sites:
            cutoff += timedelta(days=step_days)
            continue

        print(f"\n--- cutoff : {cutoff.date()} | {len(valid_sites)} sites | pas {steps_done + 1} ---")

        # Refit complet XGBoost à chaque pas
        xgb_model = XGBoostForecaster(horizon_variant=variant)
        xgb_model.fit(train_step[train_step['login_site'].isin(valid_sites)])

        # Construire le DataFrame de prédiction
        future_df = build_future_df(
            df_history=train_step[train_step['login_site'].isin(valid_sites)],
            cutoff_date=cutoff,
            horizon_days=horizon_days,
            vacances_df=data_prep._vacances_df,
            jours_feries_df=data_prep._jours_feries_df,
            fallback_zone=data_prep._fallback_zone,
        )

        if future_df.empty:
            cutoff += timedelta(days=step_days)
            continue

        # Remplir effectif_reel pour les dates déjà connues
        known = df.groupby(['login_site', 'efdate', 'codss2'], as_index=False)['efreel'].sum()
        future_df = future_df.merge(
            known.rename(columns={'efreel': 'efreel_known'}),
            on=['login_site', 'efdate', 'codss2'],
            how='left',
        )
        future_df['efreel'] = future_df['efreel_known'].combine_first(future_df['efreel'])
        future_df = future_df.drop(columns=['efreel_known'])

        # Prédictions XGBoost
        preds = xgb_model.predict(future_df)

        # Stockage dans Trino
        store_predictions(
            test_df=future_df,
            predictions={model_name: preds},
            prediction_date=cutoff.date(),
            ovh_api_key=ovh_api_key,
            ovh_secret_key=ovh_secret_key,
            table_name=table_name,
        )
        steps_done += 1

        cutoff += timedelta(days=step_days)

    print(f"\nTerminé : {steps_done} pas de {step_days} jours générés.")


def run_ma_rolling(
    horizon_weeks: int = 6,
    step_days: int = 7,
    min_train_weeks: int = 8,
    env: str | None = None,
    prefix: str | None = None,
    df: pd.DataFrame | None = None,
    data_prep: DataPreparation | None = None,
    force_start_date: str | None = None,
):
    """
    Job de prédiction MovingAverage rolling.

    Le modèle est entraîné une seule fois sur le dataset complet (la MA étant
    indépendante du cutoff pour le backfill). La boucle sert uniquement à générer
    les paires (prediction_date, target_date) avec un horizon de 6 semaines (42 j).

    Chaque cutoff produit au plus 6 prédictions par (site, weekday) ouvert.

    Args:
        horizon_weeks: nombre de semaines de prédiction après chaque cutoff (défaut 6)
        step_days: pas d'avancement entre deux cutoffs (défaut 7 j)
        min_train_weeks: nombre minimum de semaines d'historique avant le premier cutoff (défaut 8)
    """
    import os
    from src.mov_avg_forecast import MovingAverageForecaster
    from src.trino_store import (
        build_future_df, get_last_prediction_date, store_predictions, delete_predictions_since,
        passage_predict_table,
    )

    model_name = 'MovingAverage'
    ovh_api_key = os.getenv("OVH_API_KEY")
    ovh_secret_key = os.getenv("OVH_SECRET_KEY")
    horizon_days = horizon_weeks * 7  # 42 jours = 6 semaines

    # 1. Chargement des données (réutilise le df partagé si fourni)
    if df is None or data_prep is None:
        data_prep = DataPreparation(
            use_manual_entry=True,
            env=env,
            prefix=prefix,
        )
        df = data_prep.load_and_prepare()
    print(f"Données : {len(df)} lignes, {df['login_site'].nunique()} sites")

    table_name = passage_predict_table(data_prep._prefix)
    today = pd.Timestamp.today().normalize()

    # 2. Déterminer le cutoff de départ
    if force_start_date:
        delete_predictions_since(ovh_api_key, ovh_secret_key, model_name, force_start_date,
                                 table_name=table_name)
        cutoff = pd.Timestamp(force_start_date)
        print(f"Force restart depuis {cutoff.date()}")
    else:
        last_pred_date = get_last_prediction_date(ovh_api_key, ovh_secret_key, model_name=model_name,
                                                  table_name=table_name)
        if last_pred_date is None:
            cutoff = df['efdate'].min() + timedelta(weeks=min_train_weeks)
            print(f"Premier run — cutoff initial : {cutoff.date()}")
        else:
            cutoff = pd.Timestamp(last_pred_date) + timedelta(days=step_days)
            print(f"Reprise depuis {last_pred_date} — prochain cutoff : {cutoff.date()}")

    if cutoff > today + timedelta(days=horizon_days):
        print("Prédictions déjà à jour, rien à faire.")
        return

    min_obs = min_train_weeks * 5  # ~5 jours ouverts par semaine

    # 4. Boucle rolling (pas d'entraînement, juste génération des paires dates)
    steps_done = 0
    while cutoff <= today + timedelta(days=horizon_days):

        train_step = df[df['efdate'] <= cutoff]
        site_counts = train_step.groupby('login_site').size()
        valid_sites = set(site_counts[site_counts >= min_obs].index)

        if not valid_sites:
            cutoff += timedelta(days=step_days)
            continue

        print(f"\n--- cutoff : {cutoff.date()} | {len(valid_sites)} sites | pas {steps_done + 1} ---")

        # Refit sur les données disponibles au cutoff courant
        ma = MovingAverageForecaster()
        ma.fit(train_step[train_step['login_site'].isin(valid_sites)])
        print(f"  {len(ma.ma_values)} paires (site, weekday) calculées")

        # Construire le DataFrame futur (6 semaines = 42 jours)
        future_df = build_future_df(
            df_history=train_step[train_step['login_site'].isin(valid_sites)],
            cutoff_date=cutoff,
            horizon_days=horizon_days,
            vacances_df=data_prep._vacances_df,
            jours_feries_df=data_prep._jours_feries_df,
            fallback_zone=data_prep._fallback_zone,
        )

        if future_df.empty:
            cutoff += timedelta(days=step_days)
            continue

        # Remplir effectif_reel pour les dates déjà connues dans l'historique
        known = df.groupby(['login_site', 'efdate', 'codss2'], as_index=False)['efreel'].sum()
        future_df = future_df.merge(
            known.rename(columns={'efreel': 'efreel_known'}),
            on=['login_site', 'efdate', 'codss2'],
            how='left',
        )
        future_df['efreel'] = future_df['efreel_known'].combine_first(future_df['efreel'])
        future_df = future_df.drop(columns=['efreel_known'])

        # Prédictions MA
        preds = ma.predict(future_df)

        # Stockage dans Trino
        store_predictions(
            test_df=future_df,
            predictions={model_name: preds},
            prediction_date=cutoff.date(),
            ovh_api_key=ovh_api_key,
            ovh_secret_key=ovh_secret_key,
            table_name=table_name,
        )
        steps_done += 1

        cutoff += timedelta(days=step_days)

    print(f"\nTerminé : {steps_done} pas de {step_days} jours générés.")


def run_ensemble_rolling(
    prefix: str | None = None,
    env: str | None = None,
    force_start_date: str | None = None,
):
    """
    Lit {prefix}passage_predict, optimise les poids d'ensemble par horizon bin,
    génère les prédictions Ensemble pour tous les créneaux futurs,
    et bulk-insère le résultat dans {prefix}passage_predict.

    À lancer après que tous les jobs rolling individuels ont tourné.

    Args:
        prefix: préfixe de la table (ex: 'wg_13_', 'wg_test_'). Prioritaire sur env.
        env: environnement client (ex: 'prodcentre', 'prod13') — déduit le prefix si prefix absent.
        force_start_date: date ISO (YYYY-MM-DD) pour forcer un recalcul depuis cette date.
    """
    import os
    import gc
    from src.data_prep import _ENV_TO_PREFIX
    from src.trino_store import (
        PRIMARY_KEYS, FEATURE_COLUMNS,
        passage_predict_table, ensemble_weights_table,
        get_last_prediction_date, bulk_insert_predictions,
        read_calibration_predictions, read_future_predictions,
        delete_predictions_since, store_ensemble_weights,
    )
    effective_prefix = prefix or _ENV_TO_PREFIX.get(env or 'prodcentre', 'wg_test_')
    table_name = passage_predict_table(effective_prefix)

    ovh_api_key = os.getenv("OVH_API_KEY")
    ovh_secret_key = os.getenv("OVH_SECRET_KEY")

    # 1. Déterminer la date de reprise Ensemble
    if force_start_date:
        delete_predictions_since(ovh_api_key, ovh_secret_key, 'Ensemble', force_start_date,
                                 table_name=table_name)
        effective_after = str(pd.Timestamp(force_start_date) - pd.Timedelta(days=1))[:10]
        print(f"Force restart Ensemble depuis {force_start_date}")
    else:
        last_date = get_last_prediction_date(ovh_api_key, ovh_secret_key, model_name='Ensemble',
                                             table_name=table_name)
        effective_after = str(last_date) if last_date is not None else None

    # 2. Calibration : 180 derniers jours, colonnes minimales uniquement
    print("Lecture calibration (180 derniers jours)…")
    calibration = read_calibration_predictions(ovh_api_key, ovh_secret_key, ENSEMBLE_MODELS,
                                               table_name=table_name, days=180)
    print(f"  {len(calibration)} lignes, {calibration['uai'].nunique()} sites")

    # 3. Optimiser les poids puis libérer la calibration
    ens = EnsembleFromStore()
    ens.fit(calibration)
    weights_df = ens.get_weights_df()
    ew_table = ensemble_weights_table(effective_prefix)
    store_ensemble_weights(weights_df, ovh_api_key, ovh_secret_key, table_name=ew_table)
    del calibration
    gc.collect()

    # 4. Prédictions futures uniquement après la date de reprise
    print(f"Lecture prédictions futures (après {effective_after})…")
    new_future = read_future_predictions(ovh_api_key, ovh_secret_key, ENSEMBLE_MODELS,
                                         table_name=table_name, after_date=effective_after)
    if effective_after is not None:
        print(f"  Reprise depuis {effective_after} : {len(new_future)} nouvelles lignes futures")
    else:
        print(f"  Premier run : {len(new_future)} lignes futures à insérer")

    if new_future.empty:
        print("  Aucune nouvelle prédiction à générer.")
        return

    ensemble_df = ens.predict(new_future)
    del new_future
    gc.collect()

    # Compléter les colonnes manquantes pour le schéma passage_predict
    for col in FEATURE_COLUMNS:
        if col not in ensemble_df.columns:
            ensemble_df[col] = float('nan')
    output_cols = PRIMARY_KEYS + ['horizon', 'prediction', 'effectif_reel'] + FEATURE_COLUMNS
    ensemble_df = ensemble_df[[c for c in output_cols if c in ensemble_df.columns]]

    bulk_insert_predictions(ensemble_df, ovh_api_key, ovh_secret_key, table_name=table_name, step=500)
    print(f"Terminé : {len(ensemble_df)} prédictions Ensemble insérées pour {ensemble_df['uai'].nunique()} sites.")


_DEP_LOGIN_GROUPS: dict[str, list[str]] = {
    "prod13":     ["CD13"],
    "prodcentre": ["CD18", "CD41", "CD89", "CD28", "CD19", "CD21"],
}


def _get_dep_site_mapping(ovh_api_key: str, ovh_secret_key: str) -> dict[str, str]:
    """Retourne {login_site: prefix} pour tous les sites des login_groups départementaux."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from data_process.db.trino_client import TrinoClient
    from src.data_prep import _ENV_TO_PREFIX

    mapping: dict[str, str] = {}
    for env, groups in _DEP_LOGIN_GROUPS.items():
        prefix = _ENV_TO_PREFIX[env]
        db = TrinoClient(env, ovh_api_key, ovh_secret_key)
        groups_sql = ", ".join(f"'{g}'" for g in groups)
        df = db.query_as_dataframe(
            f"SELECT DISTINCT login FROM {prefix}login "
            f"WHERE logingroupe IN ({groups_sql})"
        )
        for site in df["login"].tolist():
            mapping[site] = prefix
    print(f"  Sites départementaux : {len(mapping)} ({', '.join(f'{env}={sum(1 for p in mapping.values() if p == _ENV_TO_PREFIX[env])}' for env in _DEP_LOGIN_GROUPS)})")
    return mapping


def run_global_dep_xgb_rolling(
    horizon_days: int = 35,
    step_days: int = 7,
    min_train_weeks: int = 10,
    force_start_date: str | None = None,
):
    """
    Job de prédiction XGBoost rolling multi-env pour les établissements départementaux.

    Charge les données de tous les login_groups définis dans _DEP_LOGIN_GROUPS,
    entraîne un seul modèle XGBoost global (variant='35') sur l'ensemble,
    et stocke les prédictions dans la table passage_predict de chaque env.

    Nom du modèle : 'GlobalDepXGB'
    """
    import os
    from src.data_prep import DataPreparation, _ENV_TO_PREFIX
    from src.xgb_forecast import XGBoostForecaster
    from src.trino_store import (
        build_future_df, build_predict_df,
        get_last_prediction_date, bulk_insert_predictions,
        delete_predictions_since, passage_predict_table,
    )

    model_name = 'GlobalDepXGB'
    variant = '35'
    ovh_api_key = os.getenv("OVH_API_KEY")
    ovh_secret_key = os.getenv("OVH_SECRET_KEY")
    horizon_days = min(horizon_days, 35)

    # 1. Mapping login_site → prefix (pour le routage au stockage)
    site_to_prefix = _get_dep_site_mapping(ovh_api_key, ovh_secret_key)
    if not site_to_prefix:
        print("Aucun site départemental trouvé, rien à faire.")
        return

    # 2. Chargement des données par env
    dfs = []
    ref_dp = None
    for env in _DEP_LOGIN_GROUPS:
        prefix = _ENV_TO_PREFIX[env]
        sites = [s for s, p in site_to_prefix.items() if p == prefix]
        if not sites:
            continue
        dp = DataPreparation(env=env, list_site=sites, use_manual_entry=True)
        df_env = dp.load_and_prepare()
        dfs.append(df_env)
        if ref_dp is None:
            ref_dp = dp
    if not dfs:
        print("Aucune donnée chargée, rien à faire.")
        return
    df = pd.concat(dfs, ignore_index=True)
    print(f"Données combinées : {len(df)} lignes, {df['login_site'].nunique()} sites")

    today = pd.Timestamp.today().normalize()

    # 3. Déterminer le cutoff de départ (MIN des dernières dates à jour dans tous les envs)
    if force_start_date:
        for env in _DEP_LOGIN_GROUPS:
            prefix = _ENV_TO_PREFIX[env]
            table = passage_predict_table(prefix)
            delete_predictions_since(ovh_api_key, ovh_secret_key, model_name,
                                     force_start_date, table_name=table)
        cutoff = pd.Timestamp(force_start_date)
        print(f"Force restart depuis {cutoff.date()}")
    else:
        last_dates = []
        for env in _DEP_LOGIN_GROUPS:
            prefix = _ENV_TO_PREFIX[env]
            table = passage_predict_table(prefix)
            d = get_last_prediction_date(ovh_api_key, ovh_secret_key,
                                         model_name=model_name, table_name=table)
            last_dates.append(d)
        if all(d is None for d in last_dates):
            cutoff = df['efdate'].min() + timedelta(weeks=min_train_weeks)
            print(f"Premier run — cutoff initial : {cutoff.date()}")
        else:
            valid = [d for d in last_dates if d is not None]
            oldest = min(valid)
            cutoff = pd.Timestamp(oldest) + timedelta(days=step_days)
            print(f"Reprise depuis {oldest} — prochain cutoff : {cutoff.date()}")

    if cutoff > today + timedelta(days=horizon_days):
        print("Prédictions déjà à jour, rien à faire.")
        return

    min_obs = min_train_weeks * 5

    # 4. Boucle rolling
    steps_done = 0
    while cutoff <= today + timedelta(days=horizon_days):

        train_step = df[df['efdate'] <= cutoff]
        site_counts = train_step.groupby('login_site').size()
        valid_sites = set(site_counts[site_counts >= min_obs].index)

        if not valid_sites:
            cutoff += timedelta(days=step_days)
            continue

        print(f"\n--- cutoff : {cutoff.date()} | {len(valid_sites)} sites | pas {steps_done + 1} ---")

        xgb_model = XGBoostForecaster(horizon_variant=variant)
        xgb_model.fit(train_step[train_step['login_site'].isin(valid_sites)])

        future_df = build_future_df(
            df_history=train_step[train_step['login_site'].isin(valid_sites)],
            cutoff_date=cutoff,
            horizon_days=horizon_days,
            vacances_df=ref_dp._vacances_df,
            jours_feries_df=ref_dp._jours_feries_df,
            fallback_zone=ref_dp._fallback_zone,
        )

        if future_df.empty:
            cutoff += timedelta(days=step_days)
            continue

        known = df.groupby(['login_site', 'efdate', 'codss2'], as_index=False)['efreel'].sum()
        future_df = future_df.merge(
            known.rename(columns={'efreel': 'efreel_known'}),
            on=['login_site', 'efdate', 'codss2'],
            how='left',
        )
        future_df['efreel'] = future_df['efreel_known'].combine_first(future_df['efreel'])
        future_df = future_df.drop(columns=['efreel_known'])

        preds = xgb_model.predict(future_df)

        predict_df = build_predict_df(future_df, {model_name: preds}, cutoff.date())
        if predict_df.empty:
            print("  Aucune prédiction à stocker (aucun jour ouvert dans le test set ?)")
            cutoff += timedelta(days=step_days)
            continue

        # Stocker dans chaque table env selon l'origine du login_site
        for env in _DEP_LOGIN_GROUPS:
            prefix = _ENV_TO_PREFIX[env]
            table = passage_predict_table(prefix)
            sites_env = [s for s, p in site_to_prefix.items() if p == prefix]
            subset = predict_df[predict_df['uai'].isin(sites_env)]
            if not subset.empty:
                bulk_insert_predictions(subset, ovh_api_key, ovh_secret_key, table_name=table)

        steps_done += 1
        cutoff += timedelta(days=step_days)

    print(f"\nTerminé : {steps_done} pas de {step_days} jours générés.")


def run_ensemble_rolling_retry(wait: int = 30, max_retries: int = 50):
    """Relance run_ensemble_rolling toutes les `wait` secondes en cas d'erreur Trino."""
    import time
    for attempt in range(1, max_retries + 1):
        try:
            run_ensemble_rolling()
            print(f"run_ensemble_rolling terminé avec succès (tentative {attempt}).")
            return
        except Exception as e:
            print(f"[Tentative {attempt}/{max_retries}] Erreur : {e}")
            if attempt < max_retries:
                print(f"  Nouvelle tentative dans {wait}s…")
                time.sleep(wait)
    print(f"Échec après {max_retries} tentatives.")


if __name__ == "__main__":
    # ensemble, predictions, results = predictions(use_grouped=False)
    analyses(
    ovh_api_key=os.environ['OVH_API_KEY'],
    ovh_secret_key=os.environ['OVH_SECRET_KEY'],
)
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

class ARIMAForecaster:
    """
    Modèle ARIMA par jour de la semaine (weekday-based ARIMA).
    Entraîne un modèle séparé pour chaque combinaison (site, jour_semaine).
    """

    def __init__(self, order: Tuple[int, int, int] = (3, 0, 1), order_global: Tuple[int, int, int] = (5, 0, 2)):
        """
        Args:
            order: Ordre ARIMA (p, d, q) pour les modèles par jour de semaine.
            order_global: Ordre ARIMA pour les modèles globaux par site (fallback).
        """
        self.order = order
        self.order_global = order_global
        # Dictionnaire: (site, day) -> modèle ARIMA par jour
        self.models: Dict[Tuple[str, int], object] = {}
        # Dictionnaire: site -> modèle ARIMA global (fallback)
        self.models_global: Dict[str, object] = {}
        # Dictionnaire: site -> moyenne naïve (fallback si ARIMA a échoué)
        self.site_means: Dict[str, float] = {}
        # Dictionnaire: (site, day) -> dernières valeurs pour prédiction
        self.last_values: Dict[Tuple[str, int], pd.Series] = {}

    def fit(self, train_df: pd.DataFrame) -> 'ARIMAForecaster':
        """
        Entraîne un modèle ARIMA par site ET par jour de la semaine.
        Divise chaque série temporelle en 5-6 sous-séries (une par jour).
        Entraîne aussi un modèle global par site comme fallback.
        """
        sites = train_df['login_site'].unique()

        for site in sites:
            site_df = train_df[train_df['login_site'] == site].copy()
            site_df = site_df.sort_values('efdate')

            # Sauvegarder la moyenne comme fallback ultime
            self.site_means[site] = site_df['efreel'].mean()

            # 1. Entraîner un modèle global pour le site (fallback)
            # Index entier pour éviter les problèmes de fréquence DatetimeIndex avec append()
            ts_global = pd.Series(site_df['efreel'].values)
            try:
                model_global = ARIMA(ts_global, order=self.order_global)
                self.models_global[site] = model_global.fit()
            except Exception as e:
                print(f"Erreur ARIMA global pour site {site}: {e}")

            # 2. Entraîner un modèle pour chaque jour de la semaine
            for day in range(7):  # 0=Lundi, ..., 6=Dimanche
                day_df = site_df[site_df['day'] == day].copy()

                if len(day_df) < 10:  # Pas assez de données pour ce jour
                    continue

                day_df = day_df.sort_values('efdate')

                # Index entier pour compatibilité avec append()
                ts = pd.Series(day_df['efreel'].values)

                # Garder les dernières valeurs
                self.last_values[(site, day)] = ts

                try:
                    model = ARIMA(ts, order=self.order)
                    fitted_model = model.fit()
                    self.models[(site, day)] = fitted_model
                except Exception as e:
                    print(f"Erreur ARIMA pour site {site}, jour {day}: {e}")

        print(f"ARIMA entraîné: {len(self.models)} modèles (site, jour) + {len(self.models_global)} modèles globaux")
        return self

    def update(self, new_df: pd.DataFrame) -> 'ARIMAForecaster':
        """
        Met à jour les modèles avec une nouvelle tranche de données (typiquement 1 semaine)
        via statsmodels ARIMA.append(), sans ré-estimation complète des paramètres.

        Pour les combinaisons (site, jour) déjà connues, on appende les nouvelles observations.
        Pour les nouvelles combinaisons, on accumule dans last_values jusqu'au seuil de 10 obs.

        Args:
            new_df: DataFrame contenant les nouvelles observations

        Returns:
            self
        """
        for site in new_df['login_site'].unique():
            site_df = new_df[new_df['login_site'] == site].sort_values('efdate')

            # Mise à jour de la moyenne de fallback (moyenne glissante simple)
            if site in self.site_means:
                self.site_means[site] = (self.site_means[site] + site_df['efreel'].mean()) / 2
            else:
                self.site_means[site] = site_df['efreel'].mean()

            # Mise à jour du modèle global (index entier continu)
            ts_global_vals = site_df['efreel'].values
            if site in self.models_global:
                try:
                    n = len(self.models_global[site].model.endog)
                    ts_global_new = pd.Series(ts_global_vals, index=range(n, n + len(ts_global_vals)))
                    self.models_global[site] = self.models_global[site].append(
                        ts_global_new, refit=False
                    )
                except Exception as e:
                    print(f"  Erreur append ARIMA global (site={site}): {e} — ignoré")
            elif len(site_df) >= 10:
                try:
                    self.models_global[site] = ARIMA(
                        pd.Series(ts_global_vals), order=self.order_global
                    ).fit()
                except Exception as e:
                    print(f"  Erreur fit ARIMA global nouveau site {site}: {e}")

            # Mise à jour des modèles par jour de semaine (index entier continu)
            for day in range(7):
                day_df = site_df[site_df['day'] == day].sort_values('efdate')
                if len(day_df) == 0:
                    continue

                ts_vals = day_df['efreel'].values

                if (site, day) in self.models:
                    try:
                        n = len(self.models[(site, day)].model.endog)
                        ts_new = pd.Series(ts_vals, index=range(n, n + len(ts_vals)))
                        self.models[(site, day)] = self.models[(site, day)].append(
                            ts_new, refit=False
                        )
                        existing = self.last_values.get((site, day), pd.Series(dtype=float))
                        self.last_values[(site, day)] = pd.concat(
                            [existing, pd.Series(ts_vals)], ignore_index=True
                        )
                    except Exception as e:
                        print(f"  Erreur append ARIMA (site={site}, jour={day}): {e} — ignoré")
                else:
                    # Accumule jusqu'au seuil de 10 obs pour entraîner un nouveau modèle
                    existing = self.last_values.get((site, day), pd.Series(dtype=float))
                    combined = pd.concat([existing, pd.Series(ts_vals)], ignore_index=True)
                    self.last_values[(site, day)] = combined
                    if len(combined) >= 10:
                        try:
                            self.models[(site, day)] = ARIMA(combined, order=self.order).fit()
                        except Exception as e:
                            print(f"  Erreur fit ARIMA nouveau (site={site}, jour={day}): {e}")

        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Prédit avec ARIMA en utilisant le modèle correspondant au jour de la semaine.
        Utilise le modèle global du site comme fallback si le modèle par jour n'existe pas.
        """
        predictions = []

        # Grouper par site et par jour de la semaine
        for site in test_df['login_site'].unique():
            site_test = test_df[test_df['login_site'] == site].copy()

            # Si aucun modèle n'a pu être entraîné, fallback sur la moyenne
            has_global = site in self.models_global
            has_any_day = any((site, d) in self.models for d in range(7))
            if not has_global and not has_any_day:
                site_mean = self.site_means.get(site, 0.0)
                print(f"  Fallback moyenne ({site_mean:.0f}) pour site {site} (aucun modèle ARIMA)")
                for idx in site_test.index:
                    predictions.append((idx, site_mean))
                continue

            for day in range(7):
                day_test = site_test[site_test['day'] == day].copy()

                if len(day_test) == 0:
                    continue

                day_test = day_test.sort_values('efdate')
                n_steps = len(day_test)

                # Utiliser le modèle par jour si disponible, sinon le modèle global,
                # sinon la moyenne du site
                if (site, day) in self.models:
                    model = self.models[(site, day)]
                    forecast = model.forecast(steps=n_steps)
                    for i, (idx, row) in enumerate(day_test.iterrows()):
                        predictions.append((idx, max(0, forecast.iloc[i])))
                elif has_global:
                    model = self.models_global[site]
                    forecast = model.forecast(steps=n_steps)
                    for i, (idx, row) in enumerate(day_test.iterrows()):
                        predictions.append((idx, max(0, forecast.iloc[i])))
                else:
                    site_mean = self.site_means.get(site, 0.0)
                    for idx in day_test.index:
                        predictions.append((idx, site_mean))

        # Reconstruire l'ordre original
        predictions_dict = dict(predictions)
        return np.array([predictions_dict[idx] for idx in test_df.index])
from prophet import Prophet
import pandas as pd
import numpy as np
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

class ProphetForecaster:
    """Modèle Prophet amélioré pour la prédiction avec régresseurs additionnels."""

    def __init__(
        self,
        seasonality_mode: str = 'multiplicative',
        changepoint_prior_scale: float = 0.05,
        horizon_variant: str = '21',
    ):
        """
        Args:
            seasonality_mode: 'additive' ou 'multiplicative' (par défaut)
            changepoint_prior_scale: Flexibilité de la tendance (0.001-0.5, défaut 0.05)
            horizon_variant: '21' pour mobile_mean_42_shifted_21 + daily_mobile_mean_8_shifted_4,
                             '35' pour mobile_mean_42_shifted_35 + daily_mobile_mean_8_shifted_6
        """
        self.models: Dict[str, Prophet] = {}
        self.site_means: Dict[str, float] = {}
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.horizon_variant = horizon_variant

        mean_col = 'mobile_mean_42_shifted_35' if horizon_variant == '35' else 'mobile_mean_42_shifted_21'
        daily_col = 'daily_mobile_mean_8_shifted_6' if horizon_variant == '35' else 'daily_mobile_mean_8_shifted_4'

        # Liste des régresseurs à ajouter
        self.regressors = [
            'is_holyday',
            'is_ferie',
            'is_bridge',
            'jours_avant_vacance',
            'jours_apres_vacance',
            mean_col,
            daily_col,
            'is_day_usually_open',
        ]

    def fit(self, train_df: pd.DataFrame) -> 'ProphetForecaster':
        """Entraîne un modèle Prophet par site avec régresseurs améliorés."""
        sites = train_df['login_site'].unique()

        for site in sites:
            site_df = train_df[train_df['login_site'] == site].copy()
            self.site_means[site] = site_df['efreel'].mean()

            # Vérifier qu'il y a assez de données
            if len(site_df) < 10:
                print(f"Site {site}: pas assez de données ({len(site_df)} lignes)")
                continue

            # Format Prophet
            prophet_df = site_df[['efdate', 'efreel']].rename(
                columns={'efdate': 'ds', 'efreel': 'y'}
            )

            # Ajouter les régresseurs disponibles
            available_regressors = []
            for reg in self.regressors:
                if reg in site_df.columns:
                    prophet_df[reg] = site_df[reg].values
                    available_regressors.append(reg)

            # Créer le modèle avec paramètres améliorés
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode=self.seasonality_mode,
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=10.0,  # Plus de flexibilité pour la saisonnalité
                holidays_prior_scale=10.0,      # Plus de poids pour les vacances/fériés
                interval_width=0.95
            )

            # Ajouter les régresseurs au modèle
            for reg in available_regressors:
                # Donner plus de poids aux moyennes mobiles
                prior_scale = 15.0 if 'mean' in reg else 10.0
                model.add_regressor(reg, prior_scale=prior_scale, mode='additive')

            try:
                model.fit(prophet_df)
                self.models[site] = model
            except Exception as e:
                print(f"Erreur Prophet pour site {site}: {e}")
                raise

        print(f"Prophet entraîné pour {len(self.models)} sites")
        return self
    
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Prédit avec Prophet en utilisant tous les régresseurs.

        Appelle model.predict() une fois par site (pas une fois par ligne)
        pour éviter l'overhead MCMC × N_lignes.
        """
        predictions = pd.Series(np.nan, index=test_df.index, dtype=float)

        for site, site_df in test_df.groupby('login_site'):
            if site not in self.models:
                predictions.loc[site_df.index] = self.site_means.get(site, 0.0)
                continue

            future_data: dict = {'ds': site_df['efdate'].values}
            for reg in self.regressors:
                if reg in site_df.columns:
                    vals = site_df[reg].values
                    future_data[reg] = np.where(pd.isna(vals), 0.0, vals.astype(float))
                else:
                    future_data[reg] = np.zeros(len(site_df))

            forecast = self.models[site].predict(pd.DataFrame(future_data))
            predictions.loc[site_df.index] = np.maximum(0.0, forecast['yhat'].values)

        return predictions.values
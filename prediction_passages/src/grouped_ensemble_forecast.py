"""
GroupedEnsembleForecaster: Entraîne un EnsembleForecaster séparé par groupe de sites.
Permet d'avoir des modèles spécialisés pour chaque catégorie de taille (XS, S, M, L).
"""

from src.ensemble_forecast import EnsembleForecaster
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error


class GroupedEnsembleForecaster:
    """
    Entraîne un EnsembleForecaster par groupe de sites.
    Les groupes sont définis par la colonne 'group_size' (XS, S, M, L).
    """

    def __init__(
        self,
        group_column: str = 'group_size',
        horizon_bins: List[int] = [7, 14, 21],
        min_samples_per_cell: int = 3,
        regularization_strength: float = 0.3,
        mean_floor: float = 50.0,
        exclude_mape_threshold: float = 500.0,
        val_days: int = 30,
        n_splits: int = 4,
        naive_column: str = 'daily_mobile_mean_8_shifted_4',
        optimization_metric: str = 'mase'
    ):
        """
        Args:
            group_column: Nom de la colonne contenant les groupes (défaut: 'group_size')
            Autres args: passés à chaque EnsembleForecaster
        """
        self.group_column = group_column
        self.ensemble_params = {
            'horizon_bins': horizon_bins,
            'min_samples_per_cell': min_samples_per_cell,
            'regularization_strength': regularization_strength,
            'mean_floor': mean_floor,
            'exclude_mape_threshold': exclude_mape_threshold,
            'val_days': val_days,
            'n_splits': n_splits,
            'naive_column': naive_column,
            'optimization_metric': optimization_metric
        }

        self.ensembles: Dict[str, EnsembleForecaster] = {}
        self.groups: List[str] = []
        self.model_names = ['XGBoost', 'Prophet', 'ARIMA', 'MovingAverage']
        self.site_to_group: Dict = {}
        self.site_means: Dict[str, float] = {}
        self.mean_floor = mean_floor

    def fit(self, train_df: pd.DataFrame, optimize_weights: bool = True) -> 'GroupedEnsembleForecaster':
        """
        Entraîne un EnsembleForecaster par groupe.
        """
        if self.group_column not in train_df.columns:
            raise ValueError(f"Colonne '{self.group_column}' non trouvée. "
                           f"Colonnes disponibles: {train_df.columns.tolist()}")

        self.groups = sorted(train_df[self.group_column].unique())

        # Créer le mapping site -> groupe
        for site in train_df['login_site'].unique():
            site_df = train_df[train_df['login_site'] == site]
            self.site_to_group[site] = site_df[self.group_column].iloc[0]
            self.site_means[site] = site_df['efreel'].mean()

        print("=" * 70)
        print("ENTRAÎNEMENT PAR GROUPE")
        print("=" * 70)
        print(f"Groupes détectés: {self.groups}")

        for group in self.groups:
            group_df = train_df[train_df[self.group_column] == group].copy()
            n_sites = group_df['login_site'].nunique()
            n_rows = len(group_df)

            print(f"\n{'='*70}")
            print(f"GROUPE {group}: {n_sites} sites, {n_rows} observations")
            print(f"{'='*70}")

            # Créer et entraîner l'ensemble pour ce groupe
            ensemble = EnsembleForecaster(**self.ensemble_params)
            ensemble.fit(group_df, optimize_weights=optimize_weights)
            self.ensembles[group] = ensemble

        self._print_summary()
        return self

    def _print_summary(self):
        """Affiche un résumé des poids par groupe."""
        print("\n" + "=" * 70)
        print("RÉSUMÉ DES POIDS PAR GROUPE")
        print("=" * 70)

        print(f"\n{'Groupe':<8} {'XGBoost':>10} {'Prophet':>10} {'ARIMA':>10} {'MovAvg':>10} | Sites")
        print("-" * 70)

        for group in self.groups:
            ensemble = self.ensembles[group]
            weights = ensemble.global_weights
            n_sites = len([s for s, g in self.site_to_group.items() if g == group])
            print(f"{group:<8} {weights[0]:>10.3f} {weights[1]:>10.3f} "
                  f"{weights[2]:>10.3f} {weights[3]:>10.3f} | {n_sites}")

        # Modèle dominant par groupe
        print(f"\nModèle dominant par groupe:")
        for group in self.groups:
            ensemble = self.ensembles[group]
            dominant_idx = np.argmax(ensemble.global_weights)
            dominant = self.model_names[dominant_idx]
            weight = ensemble.global_weights[dominant_idx]
            print(f"  {group}: {dominant} ({weight:.2f})")

    def predict(self, test_df: pd.DataFrame, reference_date: pd.Timestamp = None) -> Dict[str, np.ndarray]:
        """
        Prédit en utilisant l'ensemble approprié pour chaque groupe.
        """
        if self.group_column not in test_df.columns:
            raise ValueError(f"Colonne '{self.group_column}' non trouvée dans test_df")

        # Initialiser les arrays de prédictions
        n_samples = len(test_df)
        predictions = {name: np.zeros(n_samples) for name in self.model_names}
        predictions['Ensemble'] = np.zeros(n_samples)
        predictions['_horizon_idx'] = np.zeros(n_samples)
        predictions['_days_ahead'] = np.zeros(n_samples)
        predictions['_group'] = np.empty(n_samples, dtype=object)

        # Prédire par groupe
        for group in self.groups:
            group_mask = (test_df[self.group_column] == group).values

            if group_mask.sum() == 0:
                continue

            if group not in self.ensembles:
                print(f"Attention: pas d'ensemble pour le groupe {group}, utilisation du groupe le plus proche")
                # Fallback: utiliser le premier groupe disponible
                group = self.groups[0]

            group_df = test_df[group_mask].copy()
            group_preds = self.ensembles[group].predict(group_df, reference_date)

            # Remplir les prédictions
            for name in self.model_names:
                predictions[name][group_mask] = group_preds[name]

            predictions['Ensemble'][group_mask] = group_preds['Ensemble']
            predictions['_horizon_idx'][group_mask] = group_preds['_horizon_idx']
            predictions['_days_ahead'][group_mask] = group_preds['_days_ahead']
            predictions['_group'][group_mask] = group

        return predictions

    def evaluate(self, test_df: pd.DataFrame, predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Évalue les performances globales."""
        y_true = test_df['efreel'].values
        site_means = test_df['login_site'].map(self.site_means).fillna(self.mean_floor).values

        results = []
        for name, y_pred in predictions.items():
            if name.startswith('_'):
                continue

            y_pred = np.maximum(y_pred, 0)

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            smae = self._compute_smae(y_true, y_pred, site_means)

            mask = y_true != 0
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

            results.append({
                'Modèle': name,
                'MAE': mae,
                'RMSE': rmse,
                'sMAE': smae,
                'MAPE (%)': mape
            })

        return pd.DataFrame(results).sort_values('sMAE')

    def evaluate_by_group(self, test_df: pd.DataFrame, predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Évalue les performances par groupe."""
        results = []

        for group in self.groups:
            group_mask = (test_df[self.group_column] == group).values
            if group_mask.sum() == 0:
                continue

            y_true = test_df.loc[group_mask, 'efreel'].values
            site_means = test_df.loc[group_mask, 'login_site'].map(self.site_means).fillna(self.mean_floor).values

            for name, y_pred in predictions.items():
                if name.startswith('_'):
                    continue

                group_pred = np.maximum(y_pred[group_mask], 0)

                mae = mean_absolute_error(y_true, group_pred)
                rmse = np.sqrt(mean_squared_error(y_true, group_pred))
                smae = self._compute_smae(y_true, group_pred, site_means)

                mask = y_true != 0
                mape = np.mean(np.abs((y_true[mask] - group_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan

                # Calculer le MASE
                naive_col = self.ensemble_params.get('naive_column', 'daily_mobile_mean_8_shifted_4')
                if naive_col in test_df.columns:
                    y_naive = test_df.loc[group_mask, naive_col].fillna(y_true.mean()).values
                    mase = self._compute_mase(y_true, group_pred, y_naive)
                else:
                    mase = np.nan

                results.append({
                    'Groupe': group,
                    'N_sites': test_df.loc[group_mask, 'login_site'].nunique(),
                    'N_obs': group_mask.sum(),
                    'Modèle': name,
                    'MAE': mae,
                    'RMSE': rmse,
                    'sMAE': smae,
                    'MAPE (%)': mape,
                    'MASE': mase
                })

        return pd.DataFrame(results)

    def _compute_smae(self, y_true: np.ndarray, y_pred: np.ndarray, site_means: np.ndarray) -> float:
        """sMAE = mean(|y - y_pred| / max(mean_site, floor))"""
        denominators = np.maximum(site_means, self.mean_floor)
        scaled_errors = np.abs(y_true - y_pred) / denominators
        return np.mean(scaled_errors)

    def _compute_mase(self, y_true: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray, epsilon: float = 1e-6) -> float:
        """MASE = mean(|y - y_pred|) / mean(|y - y_naive|)"""
        mae_model = np.mean(np.abs(y_true - y_pred))
        mae_naive = np.mean(np.abs(y_true - y_naive))
        if mae_naive < epsilon:
            return mae_model / epsilon
        return mae_model / mae_naive

    def get_group_weights(self) -> pd.DataFrame:
        """Retourne un DataFrame des poids globaux par groupe."""
        rows = []
        for group in self.groups:
            ensemble = self.ensembles[group]
            row = {
                'group': group,
                'n_sites': len([s for s, g in self.site_to_group.items() if g == group]),
            }
            for name, w in zip(self.model_names, ensemble.global_weights):
                row[f'poids_{name}'] = w
            row['modele_dominant'] = self.model_names[np.argmax(ensemble.global_weights)]
            rows.append(row)
        return pd.DataFrame(rows)

    def get_ensemble(self, group: str) -> EnsembleForecaster:
        """Retourne l'ensemble pour un groupe donné."""
        if group not in self.ensembles:
            raise ValueError(f"Groupe '{group}' non trouvé. Groupes disponibles: {list(self.ensembles.keys())}")
        return self.ensembles[group]

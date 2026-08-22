from src.data_prep import DataPreparation
from src.mov_avg_forecast import MovingAverageForecaster
from src.arima_forecast import ARIMAForecaster
from src.prophet_forecast import ProphetForecaster
from src.xgb_forecast import XGBoostForecaster
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error


class EnsembleForecaster:
    def __init__(
        self, 
        horizon_bins: List[int] = [7, 14, 21],
        min_samples_per_cell: int = 3,
        regularization_strength: float = 0.3,
        mean_floor: float = 50.0,
        exclude_mape_threshold: float = 500.0,
        val_days: int = 30,
        n_splits: int = 4,
        naive_column: str = 'daily_mobile_mean_8_shifted_4',
        optimization_metric: str = 'mase'  # 'mase' ou 'smae'
    ):
        """
        Args:
            horizon_bins: Seuils en jours pour définir les horizons.
            min_samples_per_cell: Nombre minimum d'observations pour optimiser un (site, horizon)
            regularization_strength: Force de régularisation vers les poids globaux (0-1)
            mean_floor: Plancher pour la moyenne dans le calcul du sMAE. (plus très utile j'ai enlevé le petit site)
            exclude_mape_threshold: Seuil MAPE (%) au-dessus duquel un site est exclu de l'opti (on a des observations à 1 ca casse l'opti) Je les ai enlevé normalement plus utile.
            val_days: Nombre de jours pour le set de validation interne (défaut: 30)
            n_splits: Nombre de splits pour la validation croisée temporelle
            naive_column: Colonne de la moyenne mobile naïve pour le MASE
            optimization_metric: Métrique d'optimisation ('mase' ou 'smae')
        """
        self.xgb21     = XGBoostForecaster(horizon_variant='21')
        self.xgb35     = XGBoostForecaster(horizon_variant='35')
        self.prophet21 = ProphetForecaster(horizon_variant='21')
        self.prophet35 = ProphetForecaster(horizon_variant='35')
        self.arima     = ARIMAForecaster()
        self.ma21      = MovingAverageForecaster(
            column='daily_mobile_mean_8_shifted_4',
            fallback_column='mobile_mean_42_shifted_21',
        )
        self.ma35      = MovingAverageForecaster(
            column='daily_mobile_mean_8_shifted_6',
            fallback_column='mobile_mean_42_shifted_35',
        )

        self.horizon_bins = horizon_bins
        self.n_horizons = len(horizon_bins) + 1
        self.min_samples_per_cell = min_samples_per_cell
        self.regularization_strength = regularization_strength
        self.mean_floor = mean_floor
        self.exclude_mape_threshold = exclude_mape_threshold
        self.val_days = val_days
        self.n_splits = n_splits
        self.naive_column = naive_column
        self.optimization_metric = optimization_metric

        self.model_names = [
            'XGBoost21', 'XGBoost35',
            'Prophet21', 'Prophet35',
            'ARIMA',
            'MovingAverage21', 'MovingAverage35',
        ]

        # Poids
        n_models = len(self.model_names)
        self.default_weights = np.ones(n_models) / n_models
        self.global_weights = self.default_weights.copy()
        self.horizon_weights: Dict[int, np.ndarray] = {}
        self.site_weights: Dict[str, np.ndarray] = {}
        self.site_horizon_weights: Dict[str, Dict[int, np.ndarray]] = {}
        
        self.last_train_dates: Dict[str, pd.Timestamp] = {}
        
        self.site_means: Dict[str, float] = {}
        
        self.excluded_sites: List[str] = []
        
        self.optimization_stats: Dict = {}
        
    def _get_horizon_index(self, days_ahead: int) -> int:
        for i, threshold in enumerate(self.horizon_bins):
            if days_ahead <= threshold:
                return i
        return len(self.horizon_bins)
    
    def _get_horizon_label(self, horizon_idx: int) -> str:
        if horizon_idx == 0:
            return f"1-{self.horizon_bins[0]}j"
        elif horizon_idx < len(self.horizon_bins):
            return f"{self.horizon_bins[horizon_idx-1]+1}-{self.horizon_bins[horizon_idx]}j"
        else:
            return f">{self.horizon_bins[-1]}j"
    
    # !!!!!!!!!!!!!!!!! Les métriques !!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    def _get_naive_predictions(self, df: pd.DataFrame) -> np.ndarray:
        """Récupère les préds naives pour le MASE."""
        naive = df[self.naive_column].copy()
        return naive.fillna(df['efreel'].mean()).values
    
    def _compute_mase(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_naive: np.ndarray,
        epsilon: float = 1e-6
    ) -> float:
        """
        MASE = mean(|y - y_pred|) / mean(|y - y_naive|)
        Args:
            y_true: Valeurs réelles
            y_pred: Prédictions du modèle
            y_naive: Prédictions naïves
            epsilon: pour skip /O
        """
        mae_model = np.mean(np.abs(y_true - y_pred))
        mae_naive = np.mean(np.abs(y_true - y_naive))
        
        if mae_naive < epsilon:
            return mae_model / epsilon
        
        return mae_model / mae_naive

    def _compute_smae(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        site_means: np.ndarray
    ) -> float:
        """
        sMAE = mean(|y - y_pred| / max(mean_site, floor))
        """
        denominators = np.maximum(site_means, self.mean_floor)
        scaled_errors = np.abs(y_true - y_pred) / denominators
        return np.mean(scaled_errors)
    
    def _compute_site_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mask = y_true != 0
        if mask.sum() == 0:
            return np.inf
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    # Le gros fit -----------------------
    
    def fit(self, train_df: pd.DataFrame, optimize_weights: bool = True) -> 'EnsembleForecaster':
        print("=" * 60)
        print(f"ENTRAÎNEMENT DE L'ENSEMBLE (Site + Horizon + {self.optimization_metric})")
        print("=" * 60)
        
        # Calculer et sauvegarder les moyennes par site
        for site in train_df['login_site'].unique():
            site_df = train_df[train_df['login_site'] == site]
            self.site_means[site] = site_df['efreel'].mean()
            self.last_train_dates[site] = site_df['efdate'].max()
        
        print(f"\n  Statistiques des sites:")
        means_array = np.array(list(self.site_means.values()))
        print(f"    Moyenne des effectifs: min={means_array.min():.0f}, "
              f"max={means_array.max():.0f}, median={np.median(means_array):.0f}")
        
        print("\n--- XGBoost 21 ---")
        self.xgb21.fit(train_df)

        print("\n--- XGBoost 35 ---")
        self.xgb35.fit(train_df)

        print("\n--- Prophet 21 ---")
        self.prophet21.fit(train_df)

        print("\n--- Prophet 35 ---")
        self.prophet35.fit(train_df)

        print("\n--- ARIMA ---")
        self.arima.fit(train_df)

        print("\n--- MovingAverage 21 ---")
        self.ma21.fit(train_df)

        print("\n--- MovingAverage 35 ---")
        self.ma35.fit(train_df)
        
        if optimize_weights:
            print(f"\n--- Optimisation des poids (Site × Horizon, métrique: {self.optimization_metric}) ---")
            self._optimize_weights(train_df, val_days=self.val_days)
            
        return self
    
    def _optimize_weights(self, train_df: pd.DataFrame, val_days: int = 30):
        """
        Optimise les poids en utilisant la métrique choisie.
        """
        sites = train_df['login_site'].unique()
        max_date = train_df['efdate'].max()
        n_splits = self.n_splits
        
        all_weights = {
            'global': [],
            'horizon': {h: [] for h in range(self.n_horizons)},
            'site': {s: [] for s in sites},
            'site_horizon': {s: {h: [] for h in range(self.n_horizons)} for s in sites}
        }
        
        # Compteurs pour les stats
        total_horizon_dist = {h: 0 for h in range(self.n_horizons)}
        
        print(f"  Validation croisée temporelle: {n_splits} splits de {val_days} jours")
        print(f"  Métrique pour l'opti: {self.optimization_metric}")
        
        for split_idx in range(n_splits):
            val_end = max_date - pd.Timedelta(days=val_days * split_idx)
            val_start = val_end - pd.Timedelta(days=val_days)
            
            train_sub = train_df[train_df['efdate'] <= val_start].copy()
            val_sub = train_df[
                (train_df['efdate'] > val_start) & 
                (train_df['efdate'] <= val_end)
            ].copy()
            
            print(f"\n  Split {split_idx + 1}/{n_splits}:")
            print(f"    Train: ... -> {val_start.date()} ({len(train_sub)} lignes)")
            print(f"    Val:   {val_start.date()} -> {val_end.date()} ({len(val_sub)} lignes)")
            
            if len(val_sub) < 10:
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Pas assez de données, split ignoré")
                continue
            
            sites_in_train = set(train_sub['login_site'].unique())
            sites_in_val = set(val_sub['login_site'].unique())
            sites_missing = sites_in_val - sites_in_train
            if sites_missing:
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    Sites sans données dans train (exclus du split): {len(sites_missing)}")
                val_sub = val_sub[~val_sub['login_site'].isin(sites_missing)]
            
            # Calculer et dstribuer les horizons
            val_sub = val_sub.copy()
            val_sub['days_ahead'] = (val_sub['efdate'] - val_start).dt.days
            val_sub['horizon_idx'] = val_sub['days_ahead'].apply(self._get_horizon_index)
            val_sub['site_mean'] = val_sub['login_site'].map(self.site_means).fillna(self.mean_floor)
            horizon_dist = val_sub['horizon_idx'].value_counts().sort_index()
            print(f"    Horizons: ", end="")
            for h_idx in range(self.n_horizons):
                count = horizon_dist.get(h_idx, 0)
                total_horizon_dist[h_idx] += count
                label = self._get_horizon_label(h_idx)
                print(f"{label}={count} ", end="")
            
            print(f"    Entraînement des modèles temporaires...")
            xgb21_temp  = XGBoostForecaster(horizon_variant='21').fit(train_sub)
            xgb35_temp  = XGBoostForecaster(horizon_variant='35').fit(train_sub)
            prophet21_temp = ProphetForecaster(horizon_variant='21').fit(train_sub)
            prophet35_temp = ProphetForecaster(horizon_variant='35').fit(train_sub)
            arima_temp  = ARIMAForecaster().fit(train_sub)
            ma21_temp   = MovingAverageForecaster(
                column='daily_mobile_mean_8_shifted_4',
                fallback_column='mobile_mean_42_shifted_21',
            ).fit(train_sub)
            ma35_temp   = MovingAverageForecaster(
                column='daily_mobile_mean_8_shifted_6',
                fallback_column='mobile_mean_42_shifted_35',
            ).fit(train_sub)

            preds_all = {
                'XGBoost21':       xgb21_temp.predict(val_sub),
                'XGBoost35':       xgb35_temp.predict(val_sub),
                'Prophet21':       prophet21_temp.predict(val_sub),
                'Prophet35':       prophet35_temp.predict(val_sub),
                'ARIMA':           arima_temp.predict(val_sub),
                'MovingAverage21': ma21_temp.predict(val_sub),
                'MovingAverage35': ma35_temp.predict(val_sub),
            }
            
            y_true_all = val_sub['efreel'].values
            site_means_all = val_sub['site_mean'].values
            y_naive_all = self._get_naive_predictions(val_sub)
            
            # 1. Poids globaux
            split_global_weights = self._optimize_single(
                preds_all, y_true_all, site_means_all, y_naive_all
            )
            all_weights['global'].append(split_global_weights)
            
            # 2. Poids par horizon
            for h_idx in range(self.n_horizons):
                h_mask = val_sub['horizon_idx'] == h_idx
                n_samples = h_mask.sum()
                
                if n_samples >= self.min_samples_per_cell:
                    horizon_preds = {name: preds_all[name][h_mask] for name in self.model_names}
                    h_weights = self._optimize_single(
                        horizon_preds, 
                        y_true_all[h_mask], 
                        site_means_all[h_mask],
                        y_naive_all[h_mask]
                    )
                    all_weights['horizon'][h_idx].append(h_weights)
            
            # 3. Poids par site
            for site in sites:
                site_mask = val_sub['login_site'] == site
                n_samples = site_mask.sum()
                
                if n_samples >= self.min_samples_per_cell:
                    site_preds = {name: preds_all[name][site_mask] for name in self.model_names}
                    s_weights = self._optimize_single(
                        site_preds, 
                        y_true_all[site_mask], 
                        site_means_all[site_mask],
                        y_naive_all[site_mask]
                    )
                    all_weights['site'][site].append(s_weights)
            
            # 4. Poids par (site × horizon)
            for site in sites:
                for h_idx in range(self.n_horizons):
                    mask = (val_sub['login_site'] == site) & (val_sub['horizon_idx'] == h_idx)
                    n_samples = mask.sum()
                    
                    if n_samples >= self.min_samples_per_cell:
                        cell_preds = {name: preds_all[name][mask] for name in self.model_names}
                        cell_weights = self._optimize_single(
                            cell_preds, 
                            y_true_all[mask], 
                            site_means_all[mask],
                            y_naive_all[mask]
                        )
                        all_weights['site_horizon'][site][h_idx].append(cell_weights)

        print("\n" + "=" * 40)
        print("AGRÉGATION DES POIDS")
        print("=" * 40)
        
        print(f"\n  Distribution totale des horizons:")
        for h_idx, count in total_horizon_dist.items():
            label = self._get_horizon_label(h_idx)
            print(f"    {label}: {count} observations")
        
        if all_weights['global']:
            self.global_weights = np.mean(all_weights['global'], axis=0)
            self.global_weights = self.global_weights / self.global_weights.sum()
        
        print(f"\n  1. Poids globaux ({len(all_weights['global'])} splits):")
        print(f"     {dict(zip(self.model_names, self.global_weights.round(3)))}")
        
        print(f"\n  2. Poids par horizon:")
        for h_idx in range(self.n_horizons):
            if all_weights['horizon'][h_idx]:
                self.horizon_weights[h_idx] = np.mean(all_weights['horizon'][h_idx], axis=0)
                self.horizon_weights[h_idx] = self.horizon_weights[h_idx] / self.horizon_weights[h_idx].sum()
            else:
                self.horizon_weights[h_idx] = self.global_weights.copy()
            
            label = self._get_horizon_label(h_idx)
            n_splits_used = len(all_weights['horizon'][h_idx])
            print(f"     {label}: {n_splits_used} splits, "
                  f"poids=[{', '.join([f'{w:.2f}' for w in self.horizon_weights[h_idx]])}]")
        
        print(f"\n  3. Poids par site ({len(sites)} sites)...")
        for site in sites:
            if all_weights['site'][site]:
                self.site_weights[site] = np.mean(all_weights['site'][site], axis=0)
                self.site_weights[site] = self.site_weights[site] / self.site_weights[site].sum()
            else:
                self.site_weights[site] = self.global_weights.copy()
        
        print(f"\n  4. Poids par (site × horizon)...")
        total_cells = 0
        optimized_cells = 0
        
        for site in sites:
            self.site_horizon_weights[site] = {}
            
            for h_idx in range(self.n_horizons):
                total_cells += 1
                
                if all_weights['site_horizon'][site][h_idx]:
                    optimized_weights = np.mean(all_weights['site_horizon'][site][h_idx], axis=0)
                    optimized_weights = optimized_weights / optimized_weights.sum()
                    
                    n_splits_used = len(all_weights['site_horizon'][site][h_idx])
                    reg_factor = self.regularization_strength * (1.0 / n_splits_used)
                    reg_factor = min(reg_factor, self.regularization_strength)
                    
                    final_weights = (1 - reg_factor) * optimized_weights + reg_factor * self.horizon_weights[h_idx]
                    final_weights = final_weights / final_weights.sum()
                    
                    self.site_horizon_weights[site][h_idx] = final_weights
                    optimized_cells += 1
                else:
                    self.site_horizon_weights[site][h_idx] = self._get_fallback_weights(site, h_idx)
        
        print(f"     Cellules optimisées: {optimized_cells}/{total_cells} "
              f"({optimized_cells/total_cells*100:.1f}%)")
        
        self._print_weights_summary()
    
    def _optimize_single(
        self, 
        preds: Dict[str, np.ndarray], 
        y_true: np.ndarray,
        site_means: np.ndarray,
        y_naive: np.ndarray
    ) -> np.ndarray:
        """Optimise les poids en minimisant MASE ou sMAE."""
        
        def objective(weights):
            weights = np.abs(weights)
            weights = weights / weights.sum()
            ensemble_pred = sum(w * preds[name] for w, name in zip(weights, self.model_names))
            
            if self.optimization_metric == 'mase':
                return self._compute_mase(y_true, ensemble_pred, y_naive)
            else:
                return self._compute_smae(y_true, ensemble_pred, site_means)
        
        result = minimize(
            objective,
            x0=np.ones(len(self.model_names)) / len(self.model_names),
            method='Nelder-Mead',
            options={'maxiter': 500}
        )
        
        weights = np.abs(result.x)
        return weights / weights.sum()
        
    def _get_fallback_weights(self, site: str, horizon_idx: int) -> np.ndarray:
        """Retourne les poids de fallback selon la hiérarchie."""
        if horizon_idx in self.horizon_weights:
            return self.horizon_weights[horizon_idx].copy()
        elif site in self.site_weights:
            return self.site_weights[site].copy()
        return self.global_weights.copy()
    
    def _compute_optimization_stats(self, val_sub, preds_all, y_true_all, site_means_all):
        """Calcule des statistiques pour analyse."""
        model_smae = {}
        for name in self.model_names:
            model_smae[name] = self._compute_smae(y_true_all, preds_all[name], site_means_all)
        
        ensemble_pred = sum(
            w * preds_all[name] for w, name in zip(self.global_weights, self.model_names)
        )
        model_smae['Ensemble_global'] = self._compute_smae(y_true_all, ensemble_pred, site_means_all)
        
        self.optimization_stats = {
            'n_sites': val_sub['login_site'].nunique(),
            'n_sites_excluded': len(self.excluded_sites),
            'n_horizons': self.n_horizons,
            'n_validation_samples': len(val_sub),
            'model_smae': model_smae,
        }
    
    def _print_weights_summary(self):
        """Affiche un résumé des poids optimisés."""
        print("\n" + "=" * 60)
        print(f"RÉSUMÉ DES POIDS OPTIMISÉS (métrique: {self.optimization_metric})")
        print("=" * 60)
        
        if 'model_smae' in self.optimization_stats:
            print("\n  sMAE par modèle (validation):")
            for name, smae in sorted(self.optimization_stats['model_smae'].items(), key=lambda x: x[1]):
                print(f"    {name}: {smae:.4f}")
        
        if self.excluded_sites:
            print(f"\n  Sites exclus: {len(self.excluded_sites)}")
        
        print("\n  Poids globaux:")
        for name, w in zip(self.model_names, self.global_weights):
            print(f"    {name}: {w:.4f}")
        
        print("\n  Poids par horizon:")
        col_w = 12
        header = f"    {'Horizon':<15}" + "".join(f"{n:>{col_w}}" for n in self.model_names)
        print(header)
        print("    " + "-" * (15 + col_w * len(self.model_names)))
        for h_idx in range(self.n_horizons):
            label = self._get_horizon_label(h_idx)
            weights = self.horizon_weights.get(h_idx, self.global_weights)
            row = f"    {label:<15}" + "".join(f"{w:>{col_w}.3f}" for w in weights)
            print(row)
        
        # Modèle dominant par horizon
        print(f"\n  Modèle dominant par horizon:")
        for h_idx in range(self.n_horizons):
            label = self._get_horizon_label(h_idx)
            weights = self.horizon_weights.get(h_idx, self.global_weights)
            dominant = self.model_names[np.argmax(weights)]
            print(f"    {label}: {dominant} ({weights[np.argmax(weights)]:.2f})")
    
    def get_weights(self, site: str, horizon_idx: int = None, days_ahead: int = None) -> np.ndarray:
        """Retourne les poids pour un site et horizon donnés."""
        if days_ahead is not None:
            horizon_idx = self._get_horizon_index(days_ahead)
        if horizon_idx is None:
            horizon_idx = 0
        
        if site in self.site_horizon_weights and horizon_idx in self.site_horizon_weights[site]:
            return self.site_horizon_weights[site][horizon_idx]
        elif horizon_idx in self.horizon_weights:
            return self.horizon_weights[horizon_idx]
        elif site in self.site_weights:
            return self.site_weights[site]
        return self.global_weights
    
    def predict(self, test_df: pd.DataFrame, reference_date: pd.Timestamp = None) -> Dict[str, np.ndarray]:
        """Prédit avec tous les modèles et l'ensemble pondéré."""
        predictions = {}
        
        predictions['XGBoost21']       = self.xgb21.predict(test_df)
        predictions['XGBoost35']       = self.xgb35.predict(test_df)
        predictions['Prophet21']       = self.prophet21.predict(test_df)
        predictions['Prophet35']       = self.prophet35.predict(test_df)
        predictions['ARIMA']           = self.arima.predict(test_df)
        predictions['MovingAverage21'] = self.ma21.predict(test_df)
        predictions['MovingAverage35'] = self.ma35.predict(test_df)
        
        test_df = test_df.copy()
        ensemble_pred = np.zeros(len(test_df))
        
        test_df['days_ahead'] = 0
        test_df['horizon_idx'] = 0
        
        for site in test_df['login_site'].unique():
            site_mask = test_df['login_site'] == site
            
            if reference_date is not None:
                ref_date = reference_date
            elif site in self.last_train_dates:
                ref_date = self.last_train_dates[site]
            else:
                ref_date = test_df.loc[site_mask, 'efdate'].min() - pd.Timedelta(days=1)
            
            test_df.loc[site_mask, 'days_ahead'] = (
                test_df.loc[site_mask, 'efdate'] - ref_date
            ).dt.days
            test_df.loc[site_mask, 'horizon_idx'] = test_df.loc[site_mask, 'days_ahead'].apply(
                self._get_horizon_index
            )
        
        for site in test_df['login_site'].unique():
            for h_idx in range(self.n_horizons):
                mask = (test_df['login_site'] == site) & (test_df['horizon_idx'] == h_idx)
                mask_values = mask.values
                
                if mask_values.sum() == 0:
                    continue
                
                weights = self.get_weights(site, h_idx)
                site_horizon_pred = sum(
                    w * predictions[name][mask_values]
                    for w, name in zip(weights, self.model_names)
                )
                ensemble_pred[mask_values] = site_horizon_pred
        
        predictions['Ensemble'] = ensemble_pred
        predictions['_horizon_idx'] = test_df['horizon_idx'].values
        predictions['_days_ahead'] = test_df['days_ahead'].values
        
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
    
    def evaluate_by_horizon(self, test_df: pd.DataFrame, predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Évalue les performances par horizon."""
        y_true = test_df['efreel'].values
        horizon_idx = predictions.get('_horizon_idx', np.zeros(len(test_df)))
        site_means = test_df['login_site'].map(self.site_means).fillna(self.mean_floor).values
        
        results = []
        for h_idx in range(self.n_horizons):
            h_mask = horizon_idx == h_idx
            if h_mask.sum() == 0:
                continue
            
            label = self._get_horizon_label(h_idx)
            
            for name, y_pred in predictions.items():
                if name.startswith('_'):
                    continue
                    
                h_pred = np.maximum(y_pred[h_mask], 0)
                h_y = y_true[h_mask]
                h_means = site_means[h_mask]
                
                mae = mean_absolute_error(h_y, h_pred)
                smae = self._compute_smae(h_y, h_pred, h_means)
                
                mask = h_y != 0
                mape = np.mean(np.abs((h_y[mask] - h_pred[mask]) / h_y[mask])) * 100 if mask.sum() > 0 else np.nan
                
                results.append({
                    'Horizon': label,
                    'Horizon_idx': h_idx,
                    'N': h_mask.sum(),
                    'Modèle': name,
                    'MAE': mae,
                    'sMAE': smae,
                    'MAPE (%)': mape
                })
        
        return pd.DataFrame(results)
    
    def evaluate_by_site(self, test_df: pd.DataFrame, predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Évalue les performances par site."""
        results = []
        
        for site in test_df['login_site'].unique():
            site_mask = (test_df['login_site'] == site).values
            y_true = test_df.loc[site_mask, 'efreel'].values
            site_mean = self.site_means.get(site, self.mean_floor)
            site_means_arr = np.full(len(y_true), site_mean)
            
            for name, y_pred in predictions.items():
                if name.startswith('_'):
                    continue
                    
                site_pred = np.maximum(y_pred[site_mask], 0)
                
                mae = mean_absolute_error(y_true, site_pred)
                smae = self._compute_smae(y_true, site_pred, site_means_arr)
                
                mask = y_true != 0
                mape = np.mean(np.abs((y_true[mask] - site_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan
                
                results.append({
                    'login_site': site,
                    'site_mean': site_mean,
                    'excluded': site in self.excluded_sites,
                    'Modèle': name,
                    'MAE': mae,
                    'sMAE': smae,
                    'MAPE (%)': mape
                })
        
        return pd.DataFrame(results)
    
    def get_weights_dataframe(self) -> pd.DataFrame:
        """Retourne un DataFrame des poids par site et horizon."""
        rows = []
        
        for site, horizons in self.site_horizon_weights.items():
            site_mean = self.site_means.get(site, self.mean_floor)
            
            for h_idx, weights in horizons.items():
                row = {
                    'login_site': site,
                    'site_mean': site_mean,
                    'excluded': site in self.excluded_sites,
                    'horizon_idx': h_idx,
                    'horizon_label': self._get_horizon_label(h_idx),
                }
                for name, w in zip(self.model_names, weights):
                    row[f'poids_{name}'] = w
                row['modele_dominant'] = self.model_names[np.argmax(weights)]
                rows.append(row)
        
        return pd.DataFrame(rows)

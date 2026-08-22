"""Ensemble de prédictions construit depuis passage_predict.

Lit les prédictions individuelles déjà stockées dans la table,
optimise des poids par (site × horizon_bin) sur les données calibrées
(effectif_reel IS NOT NULL), puis applique ces poids aux prédictions futures.

Chaîne de fallback :
    (site, horizon_bin)  →  site  →  global
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


MODELS = ['ARIMA', 'Prophet21', 'Prophet35', 'XGBoost21', 'XGBoost35', 'MovingAverage', 'GlobalDepXGB']
# Bornes des bins horizon : ≤7 / 8-14 / 15-21 / >21
_HORIZON_BOUNDS = [7, 14, 21]
_INDEX_COLS = ['prediction_date', 'target_date', 'uai', 'service']


def _horizon_bin(horizon: float) -> int:
    """Renvoie l'indice du bin horizon (0=≤7, 1=8-14, 2=15-21, 3=>21)."""
    return int(np.digitize(horizon, bins=_HORIZON_BOUNDS))


_BIN_LABELS = ['≤7j', '8-14j', '15-21j', '>21j']


class EnsembleFromStore:
    """
    Ensemble pondéré basé sur les prédictions stockées dans passage_predict.

    Les poids sont optimisés par (site, horizon_bin) en minimisant la MAE.
    Si un modèle n'a pas de prédiction pour un créneau, son poids est
    redistribué aux modèles présents.

    Fallback :  (site, horizon_bin) → site → global.
    """

    def __init__(self, min_samples: int = 20):
        """
        Args:
            min_samples: nombre minimum de lignes pour optimiser une cellule.
                         En dessous, on remonte dans la chaîne de fallback.
        """
        self.min_samples = min_samples
        n = len(MODELS)
        self.global_weights: np.ndarray = np.ones(n) / n
        self.site_weights: dict[str, np.ndarray] = {}
        self.site_horizon_weights: dict[tuple[str, int], np.ndarray] = {}

    def fit(self, df: pd.DataFrame) -> 'EnsembleFromStore':
        """
        Optimise les poids depuis les données de calibration.

        Args:
            df: DataFrame avec colonnes [model, prediction, effectif_reel, horizon,
                prediction_date, target_date, uai, service].
                Seules les lignes effectif_reel IS NOT NULL sont utilisées.
        """
        calib = df[df['effectif_reel'].notna()].copy()
        if calib.empty:
            print("  EnsembleFromStore.fit : aucune donnée de calibration, poids uniformes")
            return self

        pivot = self._to_pivot(calib, value_col='prediction', extra_cols=['horizon', 'effectif_reel'])
        pivot = pivot[pivot[MODELS].notna().any(axis=1)]
        pivot['_bin'] = np.digitize(pivot['horizon'].values, bins=_HORIZON_BOUNDS)

        n_sites = pivot.index.get_level_values('uai').nunique()
        print(f"  Calibration : {len(pivot)} créneaux × {len(MODELS)} modèles, {n_sites} sites")

        # ── Poids globaux ────────────────────────────────────────────────────
        self.global_weights = self._optimize(pivot)
        print(f"  Poids globaux : {dict(zip(MODELS, self.global_weights.round(3)))}")

        # ── Poids par site ───────────────────────────────────────────────────
        site_opt, site_fb = 0, 0
        for uai, sub_site in pivot.groupby(level='uai'):
            if len(sub_site) >= self.min_samples:
                self.site_weights[uai] = self._optimize(sub_site)
                site_opt += 1
            else:
                site_fb += 1

        print(f"  Poids par site : {site_opt} optimisés, {site_fb} fallback global (< {self.min_samples} obs)")

        # ── Poids par (site, horizon_bin) ────────────────────────────────────
        sh_opt, sh_fb_site, sh_fb_global = 0, 0, 0
        for (uai, bin_idx), sub in pivot.groupby([pivot.index.get_level_values('uai'), '_bin']):
            if len(sub) >= self.min_samples:
                self.site_horizon_weights[(uai, bin_idx)] = self._optimize(sub)
                sh_opt += 1
            elif uai in self.site_weights:
                sh_fb_site += 1
            else:
                sh_fb_global += 1

        print(
            f"  Poids (site × horizon) : {sh_opt} optimisés, "
            f"{sh_fb_site} fallback site, {sh_fb_global} fallback global"
        )
        return self

    def _get_weights(self, uai: str, horizon: float) -> np.ndarray:
        """Résout la chaîne de fallback pour un (site, horizon)."""
        bin_idx = _horizon_bin(horizon)
        key = (uai, bin_idx)
        if key in self.site_horizon_weights:
            return self.site_horizon_weights[key]
        if uai in self.site_weights:
            return self.site_weights[uai]
        return self.global_weights

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Génère les prédictions ensemble pour tous les créneaux (passé + futur).

        Args:
            df: lignes de passage_predict (toutes ou juste le futur).
                Colonnes requises : model, prediction, horizon,
                prediction_date, target_date, uai, service.
                Si effectif_reel est présente, elle est propagée dans le résultat.

        Returns:
            DataFrame avec colonnes (prediction_date, target_date, uai, service,
            model='Ensemble', prediction, horizon, effectif_reel).
        """
        if df.empty:
            return pd.DataFrame(columns=_INDEX_COLS + ['model', 'prediction', 'horizon', 'effectif_reel'])

        extra = ['horizon']
        if 'effectif_reel' in df.columns:
            extra.append('effectif_reel')

        pivot = self._to_pivot(df, value_col='prediction', extra_cols=extra)

        X = pivot[MODELS].values
        uais = pivot.index.get_level_values('uai')
        horizons = pivot['horizon'].values

        weights_per_row = np.array([
            self._get_weights(uai, h) for uai, h in zip(uais, horizons)
        ])

        predictions = _weighted_predict_matrix(X, weights_per_row)

        result = pivot.reset_index()[_INDEX_COLS + extra].copy()
        result['model'] = 'Ensemble'
        result['prediction'] = predictions
        if 'effectif_reel' not in result.columns:
            result['effectif_reel'] = float('nan')
        return result

    def get_weights_df(self) -> pd.DataFrame:
        """Sérialise les poids (global / site / site_horizon) en DataFrame prêt à insérer."""
        rows = []
        for model, w in zip(MODELS, self.global_weights):
            rows.append({'scope': 'global',
                         'uai': None, 'horizon_bin': None, 'model': model, 'weight': float(w)})
        for uai, weights in self.site_weights.items():
            for model, w in zip(MODELS, weights):
                rows.append({'scope': 'site',
                             'uai': uai, 'horizon_bin': None, 'model': model, 'weight': float(w)})
        for (uai, bin_idx), weights in self.site_horizon_weights.items():
            for model, w in zip(MODELS, weights):
                rows.append({'scope': 'site_horizon',
                             'uai': uai, 'horizon_bin': _BIN_LABELS[bin_idx],
                             'model': model, 'weight': float(w)})
        return pd.DataFrame(rows)

    def _optimize(self, pivot: pd.DataFrame) -> np.ndarray:
        y = pivot['effectif_reel'].values
        X = pivot[MODELS].values

        def mae(z: np.ndarray) -> float:
            w = _softmax(z)
            preds = _weighted_predict(X, w)
            valid = ~np.isnan(preds)
            if not valid.any():
                return 1e9
            return float(np.mean(np.abs(preds[valid] - y[valid])))

        z0 = np.zeros(len(MODELS))
        res = minimize(mae, z0, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-4, 'fatol': 1e-4})
        return _softmax(res.x)

    @staticmethod
    def _to_pivot(df: pd.DataFrame, value_col: str, extra_cols: list) -> pd.DataFrame:
        """Pivote sur _INDEX_COLS × model et joint les colonnes extra."""
        piv = df.pivot_table(
            index=_INDEX_COLS,
            columns='model',
            values=value_col,
            aggfunc='first',
        ).reindex(columns=MODELS)
        meta = df.groupby(_INDEX_COLS)[extra_cols].first()
        return piv.join(meta)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def _weighted_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """X (N, M), w (M,) — renormalise par ligne sur les non-NaN. Vectorisé."""
    present = ~np.isnan(X)                              # (N, M)
    W = np.where(present, w[np.newaxis, :], 0.0)        # (N, M) 0 si absent
    W_sum = W.sum(axis=1, keepdims=True)                 # (N, 1)
    W_norm = np.where(W_sum > 0, W / W_sum, 0.0)        # (N, M) normalisé
    return (np.where(present, X, 0.0) * W_norm).sum(axis=1)


def _weighted_predict_matrix(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """X (N, M), W (N, M) — renormalise par ligne sur les non-NaN. Vectorisé."""
    present = ~np.isnan(X)                              # (N, M)
    W_m = np.where(present, W, 0.0)                     # (N, M) 0 si absent
    W_sum = W_m.sum(axis=1, keepdims=True)               # (N, 1)
    W_norm = np.where(W_sum > 0, W_m / W_sum, 0.0)      # (N, M) normalisé
    return (np.where(present, X, 0.0) * W_norm).sum(axis=1)

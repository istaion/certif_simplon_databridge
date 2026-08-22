import numpy as np
import pandas as pd


class MovingAverageForecaster:
    """
    Moyenne des 8 derniers enregistrements avec efreel > 0, par (site, weekday).

    fit() est appelé une seule fois sur l'ensemble du dataset.
    predict() utilise les valeurs pré-calculées pour chaque ligne du DataFrame cible.
    """

    def fit(self, df: pd.DataFrame) -> 'MovingAverageForecaster':
        """
        Calcule la moyenne par (login_site, day) sur les 8 derniers enregistrements
        avec efreel > 0, triés par date.
        """
        self.ma_values: dict = {}
        df_pos = df[df['efreel'] > 0].sort_values('efdate')
        for (site, weekday), grp in df_pos.groupby(['login_site', 'day']):
            last_8 = grp.tail(8)['efreel']
            self.ma_values[(site, int(weekday))] = float(last_8.mean())
        self._global_mean = float(np.mean(list(self.ma_values.values()))) if self.ma_values else 0.0
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Prédit pour chaque ligne via ma_values[(site, weekday)].
        Fallback : weekday_mean de la ligne, puis moyenne globale.
        """
        keys = list(zip(test_df['login_site'], test_df['day'].astype(int)))
        preds = np.array([self.ma_values.get(k, np.nan) for k in keys], dtype=float)

        mask = np.isnan(preds)
        if mask.any():
            fallback = test_df.loc[test_df.index[mask], 'weekday_mean'].fillna(self._global_mean).values
            preds[mask] = fallback

        return preds

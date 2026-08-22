import xgboost as xgb
from typing import Dict
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder

class XGBoostForecaster:
    def __init__(self, horizon_variant: str = '21'):
        """
        Args:
            horizon_variant: '21' pour mobile_mean_42_shifted_21 + daily_mobile_mean_8_shifted_4,
                             '35' pour mobile_mean_42_shifted_35 + daily_mobile_mean_8_shifted_6
        """
        self.models: Dict[str, xgb.XGBRegressor] = {}
        self.horizon_variant = horizon_variant

        mean_col = 'mobile_mean_42_shifted_35' if horizon_variant == '35' else 'mobile_mean_42_shifted_21'
        daily_col = 'daily_mobile_mean_8_shifted_6' if horizon_variant == '35' else 'daily_mobile_mean_8_shifted_4'

        self.numeric_cols = [
            'year', 'month', 'day',
            mean_col,
            daily_col,
            'weekday_mean',
            'is_holyday', 'is_ferie', 'is_bridge',
            'jours_avant_vacance', 'jours_apres_vacance',
            'ips',
            'is_day_usually_open'
        ]

        self.cat_cols = ['type_etablissement']
        self.cat_target_cols = ['login_site'] # on target encode l'id
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.target_encoder = TargetEncoder(cols=self.cat_target_cols)
        self.feature_cols = None
        
    def _prepare_features(self, df: pd.DataFrame, y: pd.Series = None, fit_encoder: bool = False) -> pd.DataFrame:
        """on vérifie qu'on a bien tout et on encode"""
        df = df.copy()

        missing_num = [col for col in self.numeric_cols if col not in df.columns]
        if missing_num:
            raise ValueError(f"Colonnes numériques manquantes dans la df: {missing_num}")

        missing_cat = [col for col in self.cat_cols if col not in df.columns]
        if missing_cat:
            raise ValueError(f"Colonnes catégorielles manquantes dans la df: {missing_cat}")

        if fit_encoder:
            target_encoded = self.target_encoder.fit_transform(df[self.cat_target_cols], y)
            target_encoded.columns = [f'{col}_encoded' for col in self.cat_target_cols]
            if self.cat_cols:
                onehot_encoded = self.encoder.fit_transform(df[self.cat_cols])
                self.encoded_onehot_feature_names = list(self.encoder.get_feature_names_out(self.cat_cols))
            else:
                self.encoded_onehot_feature_names = []
        else:
            target_encoded = self.target_encoder.transform(df[self.cat_target_cols])
            target_encoded.columns = [f'{col}_encoded' for col in self.cat_target_cols]
            if self.cat_cols:
                onehot_encoded = self.encoder.transform(df[self.cat_cols])

        parts = [df[self.numeric_cols], target_encoded]
        if self.encoded_onehot_feature_names:
            onehot_df = pd.DataFrame(onehot_encoded, index=df.index, columns=self.encoded_onehot_feature_names)
            parts.append(onehot_df)

        final_df = pd.concat(parts, axis=1)

        if fit_encoder:
            self.feature_cols = final_df.columns.tolist()

        return final_df
    
    def fit(self, train_df: pd.DataFrame) -> 'XGBoostForecaster':
        X = self._prepare_features(train_df, y=train_df['efreel'], fit_encoder=True)
        y = train_df['efreel']
        
        self.models['global'] = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        self.models['global'].fit(X, y)
        
        print(f"XGBoost entraîné sur {len(X)} échantillons")
        return self
    
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """Prédit avec XGBoost."""
        df = self._prepare_features(test_df, fit_encoder=False)
        X = df[self.feature_cols].fillna(0)
        return self.models['global'].predict(X)
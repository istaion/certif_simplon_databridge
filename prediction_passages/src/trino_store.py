"""Persistance des prédictions dans les tables {prefix}passage_predict (Trino)."""

import datetime
import os
import sys
import numpy as np
import pandas as pd

# TrinoClient se trouve dans data_process, un niveau au-dessus de prediction_passages
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from data_process.db.trino_client import TrinoClient

# service est dans la clé primaire : une prédiction est unique par (date, site, service, modèle)
PRIMARY_KEYS = ["prediction_date", "target_date", "uai", "service", "model"]

# Mapping codss2 → libellé service
CODSS2_TO_SERVICE = {'1': "PETIT DEJEUNER", '2': "DEJEUNER", '4': "DINER"}

FEATURE_COLUMNS = [
    "school_year",
    "is_holyday",          # DOUBLE (0.0/1.0) — TrinoClient convertit les int en float64
    "is_ferie",
    "is_bridge",
    "is_day_usually_open",
    "opening_days",
    "group_size",
    "ips",
    "type_etablissement",
    "vacances_zone",
    "mobile_mean_42_shifted_21",
    "daily_mobile_mean_8_shifted_4",
    "mobile_mean_42_shifted_35",
    "daily_mobile_mean_8_shifted_6",
    "weekday_mean",
]


_PREFIX_TO_ENV: dict[str, str] = {
    "wg_test_": "prodcentre",
    "wg_93_": "prod93",
    "wg_rhone_": "prodrhone",
    "wg_13_": "prod13",
}


def passage_predict_table(prefix: str) -> str:
    """Retourne le nom de la table passage_predict pour un préfixe donné.
    Ex: 'wg_13_' → 'wg_13_passage_predict'
    """
    return f"{prefix}passage_predict"


def _client_for_table(table_name: str, ovh_api_key: str, ovh_secret_key: str) -> TrinoClient:
    """Crée un TrinoClient dans l'env correspondant au préfixe de la table."""
    for suffix in ("passage_predict", "ensemble_weights"):
        if table_name.endswith(suffix):
            prefix = table_name[: -len(suffix)]
            env = _PREFIX_TO_ENV.get(prefix)
            if env:
                return TrinoClient(env, ovh_api_key, ovh_secret_key)
    return TrinoClient("default_dataset", ovh_api_key, ovh_secret_key)


def ensemble_weights_table(prefix: str) -> str:
    """Retourne le nom de la table ensemble_weights pour un préfixe donné."""
    return f"{prefix}ensemble_weights"


def ensure_ensemble_weights_table(db: TrinoClient, table_name: str) -> None:
    """Crée la table {table_name} si elle n'existe pas encore."""
    db.run_query(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            scope        VARCHAR,
            uai          VARCHAR,
            horizon_bin  VARCHAR,
            model        VARCHAR,
            weight       DOUBLE
        )
    """)


def store_ensemble_weights(
    weights_df: pd.DataFrame,
    ovh_api_key: str,
    ovh_secret_key: str,
    table_name: str,
) -> None:
    """Remplace les poids d'ensemble dans {table_name} (DELETE + INSERT)."""
    db = _client_for_table(table_name, ovh_api_key, ovh_secret_key)
    ensure_ensemble_weights_table(db, table_name)
    db.run_query(f"DELETE FROM {table_name} WHERE scope IS NOT NULL OR scope IS NULL")
    rows = db.bulk_insert(table_name, weights_df)
    print(f"  {rows} poids stockés dans {table_name}")


def ensure_passage_predict_table(db: TrinoClient, table_name: str) -> None:
    """Crée la table {table_name} si elle n'existe pas encore."""
    db.run_query(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            prediction_date               DATE,
            target_date                   DATE,
            horizon                       DOUBLE,
            uai                           VARCHAR,
            service                       VARCHAR,
            model                         VARCHAR,
            prediction                    DOUBLE,
            effectif_reel                 DOUBLE,
            school_year                   VARCHAR,
            is_holyday                    DOUBLE,
            is_ferie                      DOUBLE,
            is_bridge                     DOUBLE,
            is_day_usually_open           DOUBLE,
            opening_days                  VARCHAR,
            group_size                    VARCHAR,
            ips                           DOUBLE,
            type_etablissement            VARCHAR,
            vacances_zone                 VARCHAR,
            mobile_mean_42_shifted_21     DOUBLE,
            daily_mobile_mean_8_shifted_4 DOUBLE,
            mobile_mean_42_shifted_35     DOUBLE,
            daily_mobile_mean_8_shifted_6 DOUBLE,
            weekday_mean                  DOUBLE
        )
        WITH (
            partitioning = ARRAY['month(prediction_date)'],
            extra_properties = MAP(
                ARRAY[
                    'write.target-file-size-bytes',
                    'write.metadata.delete-after-commit.enabled',
                    'write.metadata.previous-versions-max'
                ],
                ARRAY[
                    '268435456',
                    'true',
                    '50'
                ]
            )
        )
    """)


def build_predict_df(
    test_df: pd.DataFrame,
    predictions: dict,
    prediction_date: datetime.date,
) -> pd.DataFrame:
    """
    Construit le DataFrame à insérer dans {prefix}passage_predict.

    Filtre sur is_day_usually_open == 1 (jours habituellement ouverts, hors vacances).
    Le calcul horizon = target_date - prediction_date est en jours.
    effectif_reel est renseigné si la colonne efreel existe (cas test-set).
    service est dérivé de codss2 : 1→"PETIT DEJEUNER", 2→"DEJEUNER", 4→"DINER".
    """
    open_mask = test_df["is_day_usually_open"] == 1
    df_open = test_df[open_mask].copy()

    rows = []
    for model_name, preds_array in predictions.items():
        pred_series = pd.Series(preds_array, index=test_df.index)
        sub = df_open.copy()
        sub["prediction"] = pred_series.loc[sub.index].values
        sub["model"] = model_name
        rows.append(sub)

    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)

    combined["prediction_date"] = prediction_date
    combined["target_date"] = combined["efdate"].dt.date
    combined["horizon"] = (combined["efdate"] - pd.Timestamp(prediction_date)).dt.days
    combined["uai"] = combined["login_site"]
    combined["service"] = combined["codss2"].map(CODSS2_TO_SERVICE).fillna("INCONNU")
    combined["effectif_reel"] = combined["efreel"] if "efreel" in combined.columns else float("nan")

    for col in FEATURE_COLUMNS:
        if col not in combined.columns:
            combined[col] = float("nan")

    output_cols = PRIMARY_KEYS + ["horizon", "prediction", "effectif_reel"] + FEATURE_COLUMNS
    return combined[output_cols].reset_index(drop=True)


def get_last_prediction_date(
    ovh_api_key: str,
    ovh_secret_key: str,
    model_name: str = 'ARIMA',
    table_name: str = '',
) -> datetime.date | None:
    """Retourne le dernier prediction_date pour un modèle dans la table donnée."""
    try:
        db = _client_for_table(table_name, ovh_api_key, ovh_secret_key)
        ensure_passage_predict_table(db, table_name)
        result = db.query_as_dataframe(
            f"SELECT MAX(prediction_date) AS last_date FROM {table_name} "
            f"WHERE model = '{model_name}'"
        )
        val = result['last_date'].iloc[0]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return pd.Timestamp(val).date()
    except Exception as e:
        print(f"  Impossible de lire le dernier prediction_date : {e}")
        return None


def delete_predictions_since(
    ovh_api_key: str,
    ovh_secret_key: str,
    model_name: str,
    since_date: str,
    table_name: str = '',
) -> None:
    """Supprime les prédictions d'un modèle à partir de since_date (inclus)."""
    db = _client_for_table(table_name, ovh_api_key, ovh_secret_key)
    db.run_query(
        f"DELETE FROM {table_name} "
        f"WHERE model = '{model_name}' AND prediction_date >= DATE '{since_date}'"
    )
    print(f"  DELETE {model_name} depuis {since_date} ({table_name})")


def build_future_df(
    df_history: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    horizon_days: int = 30,
    vacances_df: pd.DataFrame | None = None,
    jours_feries_df: pd.DataFrame | None = None,
    fallback_zone: str = 'B',
) -> pd.DataFrame:
    """
    Construit un DataFrame synthétique couvrant les horizon_days jours après cutoff_date,
    pour tous les sites présents dans df_history.

    Les features sont estimées à partir des dernières valeurs connues par site :
    - opening_days, vacances_zone, group_size, ips, type_etablissement → dernière valeur
    - weekday_mean, daily_mobile_mean_8_shifted_4 → dernière valeur par (site, jour)
    - mobile_mean_42_shifted_21 → dernière valeur globale du site
    - is_holyday / is_ferie / is_bridge → calculés depuis les tables Trino passées en paramètre
    - efreel → NaN (inconnu pour le futur)
    """
    future_dates = pd.date_range(
        start=cutoff_date + pd.Timedelta(days=1),
        periods=horizon_days,
        freq='D',
    )

    ferie_set: set = set()
    bridge_set: set = set()
    if jours_feries_df is not None:
        ferie_ts = pd.to_datetime(jours_feries_df['date']).dt.normalize()
        ferie_set = set(ferie_ts)
        for fd in ferie_ts:
            if fd.weekday() == 1:
                bridge_set.add(fd - pd.Timedelta(days=1))
            elif fd.weekday() == 3:
                bridge_set.add(fd + pd.Timedelta(days=1))

    vacances_by_zone: dict = {}
    if vacances_df is not None:
        for zone, grp in vacances_df.groupby('zone'):
            vacances_by_zone[zone] = list(zip(grp['date_debut'], grp['date_fin']))

    def _is_holiday(date: pd.Timestamp, zone: str) -> int:
        ranges = vacances_by_zone.get(zone, vacances_by_zone.get(fallback_zone, []))
        return int(any(d1 <= date <= d2 for d1, d2 in ranges))

    cy = cutoff_date.year
    cm = cutoff_date.month
    sy_start = cy - 1 if cm < 8 else cy
    current_sy = f"{sy_start}-{sy_start + 1}"

    site_features: dict = {}
    for site, site_df in df_history.groupby('login_site'):
        site_df = site_df.sort_values('efdate')
        last = site_df.iloc[-1]

        sy_rows = site_df[site_df['school_year'] == current_sy]
        if sy_rows.empty:
            sy_rows = site_df
        opening_days = sy_rows['opening_days'].iloc[-1] if 'opening_days' in sy_rows.columns else "1111100"

        weekday_means: dict = {}
        daily_means_4: dict = {}
        daily_means_6: dict = {}
        for day, day_grp in site_df.groupby('day'):
            if 'weekday_mean' in day_grp.columns:
                weekday_means[day] = day_grp['weekday_mean'].iloc[-1]
            if 'daily_mobile_mean_8_shifted_4' in day_grp.columns:
                daily_means_4[day] = day_grp['daily_mobile_mean_8_shifted_4'].iloc[-1]
            if 'daily_mobile_mean_8_shifted_6' in day_grp.columns:
                daily_means_6[day] = day_grp['daily_mobile_mean_8_shifted_6'].iloc[-1]

        site_features[site] = {
            'codss2': str(last.get('codss2', '2')),
            'opening_days': opening_days,
            'vacances_zone': last.get('vacances_zone', None),
            'group_size': last.get('group_size', None),
            'ips': last.get('ips', None),
            'type_etablissement': last.get('type_etablissement', None),
            'mobile_mean_42_shifted_21': last.get('mobile_mean_42_shifted_21', None),
            'mobile_mean_42_shifted_35': last.get('mobile_mean_42_shifted_35', None),
            'weekday_means': weekday_means,
            'daily_means_4': daily_means_4,
            'daily_means_6': daily_means_6,
        }

    rows = []
    for site, sf in site_features.items():
        zone = sf['vacances_zone'] or fallback_zone
        opening_days = sf['opening_days']

        for date in future_dates:
            weekday = int(date.weekday())
            month = int(date.month)
            year = int(date.year)
            sy_s = year - 1 if month < 8 else year
            school_year = f"{sy_s}-{sy_s + 1}"

            is_hol = _is_holiday(date, zone)
            is_fer = int(date in ferie_set)
            is_bri = int(date in bridge_set)
            day_open = int(opening_days[weekday]) if len(opening_days) > weekday else 0
            is_open = day_open if is_hol == 0 else 0

            rows.append({
                'login_site': site,
                'efdate': date,
                'day': weekday,
                'year': year,
                'month': month,
                'school_year': school_year,
                'codss2': sf['codss2'],
                'efreel': float('nan'),
                'is_holyday': is_hol,
                'is_ferie': is_fer,
                'is_bridge': is_bri,
                'is_day_usually_open': is_open,
                'opening_days': opening_days,
                'group_size': sf['group_size'],
                'ips': sf['ips'],
                'type_etablissement': sf['type_etablissement'],
                'vacances_zone': sf['vacances_zone'],
                'mobile_mean_42_shifted_21': sf['mobile_mean_42_shifted_21'],
                'mobile_mean_42_shifted_35': sf['mobile_mean_42_shifted_35'],
                'daily_mobile_mean_8_shifted_4': sf['daily_means_4'].get(weekday),
                'daily_mobile_mean_8_shifted_6': sf['daily_means_6'].get(weekday),
                'weekday_mean': sf['weekday_means'].get(weekday),
            })

    result = pd.DataFrame(rows)
    if result.empty:
        result['jours_avant_vacance'] = pd.Series(dtype=float)
        result['jours_apres_vacance'] = pd.Series(dtype=float)
        return result

    from src.data_prep import DataPreparation
    result_avant = pd.Series(30.0, index=result.index)
    result_apres = pd.Series(30.0, index=result.index)
    for zone, periods in vacances_by_zone.items():
        mask = result['vacances_zone'] == zone
        if mask.any():
            av, ap = DataPreparation._compute_distance_vacances(result.loc[mask, 'efdate'], periods)
            result_avant[mask] = av.values
            result_apres[mask] = ap.values
    mask_no_zone = result['vacances_zone'].isna()
    if mask_no_zone.any() and 'B' in vacances_by_zone:
        av, ap = DataPreparation._compute_distance_vacances(result.loc[mask_no_zone, 'efdate'], vacances_by_zone['B'])
        result_avant[mask_no_zone] = av.values
        result_apres[mask_no_zone] = ap.values
    result['jours_avant_vacance'] = result_avant
    result['jours_apres_vacance'] = result_apres
    return result


def read_passage_predict(
    ovh_api_key: str,
    ovh_secret_key: str,
    table_name: str,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Lit {table_name} depuis Trino et retourne un DataFrame."""
    db = _client_for_table(table_name, ovh_api_key, ovh_secret_key)
    ensure_passage_predict_table(db, table_name)
    where = ""
    if models:
        quoted = ", ".join(f"'{m}'" for m in models)
        where = f"WHERE model IN ({quoted})"
    return db.query_as_dataframe(f"SELECT * FROM {table_name} {where}")


# Colonnes minimales nécessaires pour EnsembleFromStore (évite SELECT *)
_ENSEMBLE_COLS = "prediction_date, target_date, uai, service, model, prediction, effectif_reel, horizon"


def read_calibration_predictions(
    ovh_api_key: str,
    ovh_secret_key: str,
    models: list[str],
    table_name: str,
    days: int = 180,
) -> pd.DataFrame:
    """Lit les prédictions calibrées (effectif_reel IS NOT NULL) sur les `days` derniers jours."""
    db = _client_for_table(table_name, ovh_api_key, ovh_secret_key)
    ensure_passage_predict_table(db, table_name)
    quoted = ", ".join(f"'{m}'" for m in models)
    cutoff = (pd.Timestamp.today() - pd.Timedelta(days=days)).date()
    sql = f"""
        SELECT {_ENSEMBLE_COLS}
        FROM {table_name}
        WHERE model IN ({quoted})
          AND effectif_reel IS NOT NULL
          AND prediction_date >= DATE '{cutoff}'
    """
    return db.query_as_dataframe(sql)


def read_future_predictions(
    ovh_api_key: str,
    ovh_secret_key: str,
    models: list[str],
    table_name: str,
    after_date=None,
) -> pd.DataFrame:
    """Lit les prédictions futures (effectif_reel IS NULL) après `after_date`."""
    db = _client_for_table(table_name, ovh_api_key, ovh_secret_key)
    ensure_passage_predict_table(db, table_name)
    quoted = ", ".join(f"'{m}'" for m in models)
    where = f"model IN ({quoted}) AND effectif_reel IS NULL"
    if after_date is not None:
        where += f" AND prediction_date > DATE '{after_date}'"
    sql = f"SELECT {_ENSEMBLE_COLS} FROM {table_name} WHERE {where}"
    return db.query_as_dataframe(sql)


def delete_model_predictions(
    ovh_api_key: str,
    ovh_secret_key: str,
    model_name: str,
    table_name: str,
) -> None:
    """Supprime toutes les lignes pour un modèle donné dans {table_name}."""
    db = _client_for_table(table_name, ovh_api_key, ovh_secret_key)
    db.run_query(f"DELETE FROM {table_name} WHERE model = '{model_name}'")


def bulk_insert_predictions(
    predict_df: pd.DataFrame,
    ovh_api_key: str,
    ovh_secret_key: str,
    table_name: str,
    step: int = 500,
) -> int:
    """Insère directement un DataFrame dans {table_name} (sans build_predict_df)."""
    db = _client_for_table(table_name, ovh_api_key, ovh_secret_key)
    ensure_passage_predict_table(db, table_name)
    rows = db.bulk_insert(table_name, predict_df, step=step)
    print(f"  {rows} lignes insérées dans {table_name}")
    return rows


def upsert_predictions(
    predict_df: pd.DataFrame,
    ovh_api_key: str,
    ovh_secret_key: str,
    table_name: str,
) -> int:
    """Upserte un DataFrame dans {table_name} via MERGE (clé = PRIMARY_KEYS)."""
    db = _client_for_table(table_name, ovh_api_key, ovh_secret_key)
    ensure_passage_predict_table(db, table_name)
    rows = db.upsert(table_name, PRIMARY_KEYS, predict_df)
    print(f"  {rows} lignes upsertées dans {table_name}")
    return rows


def store_predictions(
    test_df: pd.DataFrame,
    predictions: dict,
    prediction_date: datetime.date,
    ovh_api_key: str,
    ovh_secret_key: str,
    table_name: str,
) -> int:
    """
    Stocke les prédictions d'un ou plusieurs modèles dans {table_name}.

    Args:
        test_df: DataFrame de test issu de DataPreparation
        predictions: dict {nom_modele: np.ndarray} des prédictions
        prediction_date: date d'exécution (= max(train efdate))
        table_name: nom complet de la table cible (ex: 'wg_13_passage_predict')
    """
    db = _client_for_table(table_name, ovh_api_key, ovh_secret_key)
    ensure_passage_predict_table(db, table_name)

    predict_df = build_predict_df(test_df, predictions, prediction_date)
    if predict_df.empty:
        print("  Aucune prédiction à stocker (aucun jour ouvert dans le test set ?)")
        return 0

    models = list(predictions.keys())
    n_sites = predict_df["uai"].nunique()
    services = predict_df["service"].unique().tolist()
    print(f"  {len(predict_df)} prédictions à stocker — modèle(s): {models}, {n_sites} sites, service(s): {services}")

    rows = db.bulk_insert(table_name, predict_df)
    print(f"  {rows} lignes insérées dans {table_name}")
    return rows

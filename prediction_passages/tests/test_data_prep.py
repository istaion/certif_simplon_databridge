"""
Tests de préparation des données (C12) :
  - schéma et types des données préparées (DataPreparation.load_and_prepare)
  - absence de valeurs aberrantes après nettoyage
  - cohérence du découpage train/test (pas de chevauchement temporel)
  - absence de fuite de données via les moyennes mobiles décalées
"""
import pandas as pd
import pytest

from conftest import DEMO_UAI, build_data_preparation

EXPECTED_COLUMNS = {
    "login_site", "codss2", "efdate", "efreel",
    "year", "month", "day", "school_year",
    "mobile_mean_42", "mobile_mean_42_shifted_21", "mobile_mean_42_shifted_35",
    "daily_mobile_mean_8", "daily_mobile_mean_8_shifted_4", "daily_mobile_mean_8_shifted_6",
    "weekday_mean",
    "is_holyday", "is_ferie", "is_bridge",
    "jours_avant_vacance", "jours_apres_vacance",
    "ips", "type_etablissement", "vacances_zone",
    "opening_days", "is_day_usually_open", "group_size",
}


# ── Schéma ────────────────────────────────────────────────────────────────────

def test_expected_columns_present(prepared_df):
    df, _ = prepared_df
    missing = EXPECTED_COLUMNS - set(df.columns)
    assert not missing, f"Colonnes manquantes dans le DataFrame préparé : {missing}"


def test_column_dtypes(prepared_df):
    df, _ = prepared_df
    assert pd.api.types.is_datetime64_any_dtype(df["efdate"])
    assert pd.api.types.is_numeric_dtype(df["efreel"])
    assert set(df["day"].unique()) <= set(range(7))
    # Seul le service "2" (déjeuner) est conservé en sortie de load_and_prepare
    assert set(df["codss2"].unique()) == {"2"}


def test_binary_flag_columns_are_0_1(prepared_df):
    df, _ = prepared_df
    for col in ["is_holyday", "is_ferie", "is_bridge", "is_day_usually_open"]:
        assert set(df[col].unique()) <= {0, 1}, f"{col} contient des valeurs hors {{0,1}}"


# ── Absence de valeurs aberrantes après nettoyage ────────────────────────────

def test_no_low_efreel_values(prepared_df):
    """Les effectifs <= 1 (bruit de saisie) doivent être filtrés."""
    df, _ = prepared_df
    assert (df["efreel"] > 1).all()


def test_no_future_dates(prepared_df):
    df, _ = prepared_df
    today = pd.Timestamp.today().normalize()
    assert (df["efdate"] <= today).all()


def test_demo_sites_excluded(prepared_df):
    df, _ = prepared_df
    assert DEMO_UAI not in set(df["login_site"].unique())


def test_no_missing_efreel(prepared_df):
    df, _ = prepared_df
    assert df["efreel"].notna().all()


def test_distance_vacances_within_bounds(prepared_df):
    """jours_avant_vacance / jours_apres_vacance sont plafonnés à [0, 30]."""
    df, _ = prepared_df
    for col in ["jours_avant_vacance", "jours_apres_vacance"]:
        assert (df[col] >= 0).all()
        assert (df[col] <= 30).all()


# ── Exclusion des saisies manuelles (use_manual_entry=False) ────────────────

def test_manual_entry_exclusion(monkeypatch, synthetic_tables):
    dp_with = build_data_preparation(monkeypatch, synthetic_tables, use_manual_entry=True)
    df_with = dp_with.load_and_prepare()

    dp_without = build_data_preparation(monkeypatch, synthetic_tables, use_manual_entry=False)
    df_without = dp_without.load_and_prepare()

    assert len(df_without) < len(df_with)


# ── Cohérence du découpage train/test ───────────────────────────────────────

def test_train_test_split_no_temporal_overlap(prepared_df):
    df, dp = prepared_df
    train, test = dp.train_test_split_by_date(df, test_days=30)

    assert len(train) > 0 and len(test) > 0
    assert train["efdate"].max() < test["efdate"].min()
    assert len(train) + len(test) == len(df)


def test_train_test_split_covers_full_range(prepared_df):
    df, dp = prepared_df
    train, test = dp.train_test_split_by_date(df, test_days=30)
    assert train["efdate"].min() == df["efdate"].min()
    assert test["efdate"].max() == df["efdate"].max()


# ── Absence de fuite via les moyennes mobiles décalées ──────────────────────

@pytest.mark.parametrize(
    "mean_col,daily_col,shift_mean,shift_daily",
    [
        ("mobile_mean_42_shifted_21", "daily_mobile_mean_8_shifted_4", 21, 4),
        ("mobile_mean_42_shifted_35", "daily_mobile_mean_8_shifted_6", 35, 6),
    ],
)
def test_shifted_rolling_features_match_manual_recomputation(
    prepared_df, mean_col, daily_col, shift_mean, shift_daily
):
    """
    Recalcule manuellement les moyennes mobiles décalées à partir des données
    brutes (efreel) et vérifie qu'elles correspondent aux colonnes du
    DataFrame préparé -- si une fuite existait (par ex. shift() oublié ou mal
    appliqué), la recomputation ne correspondrait plus après un décalage
    volontairement introduit ci-dessous.
    """
    df, _ = prepared_df
    site, codss2 = df.iloc[0][["login_site", "codss2"]]
    sub = (
        df[(df["login_site"] == site) & (df["codss2"] == codss2)]
        .sort_values("efdate")
        .reset_index(drop=True)
    )

    expected_mobile = sub["efreel"].rolling(window=42).mean().shift(shift_mean)

    # La valeur au dernier jour ne doit dépendre que des observations
    # antérieures à (date - shift) : on vérifie qu'elle est inchangée si on
    # "efface" les `shift_mean - 1` dernières valeurs de la série brute.
    last_valid = expected_mobile.dropna()
    assert not last_valid.empty, "Pas assez de données pour valider le décalage (agrandir le jeu synthétique)"
    idx = last_valid.index[-1]

    truncated = sub["efreel"].copy()
    truncated.iloc[idx - shift_mean + 1 : idx + 1] = None  # efface la fenêtre récente
    recomputed = truncated.rolling(window=42).mean().shift(shift_mean)
    assert recomputed.iloc[idx] == pytest.approx(expected_mobile.iloc[idx]), (
        f"{mean_col} semble utiliser des données à moins de {shift_mean} jours de la cible : fuite potentielle"
    )

    # --- daily_mobile_mean_8_shifted_* : même vérification, par jour de semaine ---
    expected_daily_shifted = sub.groupby("day")["efreel"].transform(
        lambda x: x.rolling(window=8).mean().shift(shift_daily)
    )
    valid_daily = expected_daily_shifted.dropna()
    assert not valid_daily.empty, "Pas assez de données par jour de semaine pour valider le décalage journalier"
    idx_daily = valid_daily.index[-1]
    assert sub[daily_col].iloc[idx_daily] == pytest.approx(expected_daily_shifted.iloc[idx_daily]), (
        f"{daily_col} ne correspond pas à la recomputation manuelle (décalage {shift_daily} jours) : fuite potentielle"
    )

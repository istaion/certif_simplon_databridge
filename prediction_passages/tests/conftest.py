"""
Fixtures partagées pour les tests du modèle de prédiction (C12 -- E3).

Stratégie générale : `DataPreparation.load_and_prepare()` ouvre une vraie
connexion Trino dès sa première ligne. Plutôt que de la contourner en
réimplémentant la logique de préparation (ce qui ne testerait plus le vrai
code), on mocke uniquement la frontière I/O -- `connect()` -- avec un faux
curseur qui sert des tables synthétiques déterministes. Tout le reste du
pipeline (nettoyage, enrichissement, feature engineering) tourne pour de
vrai sur ces données de test.

Les tables synthétiques sont volontairement construites pour rester dans le
même bloc de vacances scolaires (aucune période de vacances/jour férié dans
la plage de dates utilisée) : ça simplifie les assertions sur les colonnes
calendaires sans avoir à recalculer ces règles dans les tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Déclenche prediction_passages/__init__.py (ajoute prediction_passages/ à
# sys.path pour que les imports internes "from src.xxx import ..." marchent).
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import prediction_passages  # noqa: F401,E402

SITES = ("SITE_A", "SITE_B")
DEMO_UAI = "0180000X"  # UAI de démonstration réellement exclu par DataPreparation


def _school_year_label(date: pd.Timestamp) -> str:
    y = date.year - 1 if date.month < 8 else date.year
    return f"{y}-{y + 1}"


def make_synthetic_effect_df(n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Construit un jeu de données synthétique "propre" (2 sites, jours ouvrés
    uniquement, saisonnalité hebdomadaire déterministe + faible bruit) ainsi
    que quelques lignes délibérément aberrantes, pour vérifier qu'elles sont
    bien éliminées par le nettoyage (`load_and_prepare`).
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01")  # lundi
    all_days = pd.date_range(start, periods=n_days, freq="D")
    weekdays = all_days[all_days.weekday < 5]

    weekday_effect = {0: -5, 1: 0, 2: 5, 3: 0, 4: -10}
    rows = []
    for site_idx, site in enumerate(SITES):
        base = 80 + site_idx * 40
        for i, d in enumerate(weekdays):
            noise = rng.normal(0, 2)
            efreel = max(2.0, round(base + weekday_effect[d.weekday()] + noise))
            origine = "MANUEL" if (site_idx + i) % 5 == 0 else "AUTO"
            rows.append({
                "efdate": d, "origine": origine, "codss2": "2",
                "login_site": site, "efreel": float(efreel),
            })

    df = pd.DataFrame(rows, columns=["efdate", "origine", "codss2", "login_site", "efreel"])

    # Lignes aberrantes, volontairement injectées pour être vérifiées comme exclues :
    aberrant = pd.DataFrame([
        # établissement de démonstration -> doit être exclu
        {"efdate": weekdays[10], "origine": "AUTO", "codss2": "2", "login_site": DEMO_UAI, "efreel": 50.0},
        # effectif <= 1 (bruit de saisie) -> doit être exclu
        {"efdate": weekdays[20], "origine": "AUTO", "codss2": "2", "login_site": SITES[0], "efreel": 1.0},
        # date future -> doit être exclue
        {"efdate": pd.Timestamp.today().normalize() + pd.Timedelta(days=5),
         "origine": "AUTO", "codss2": "2", "login_site": SITES[0], "efreel": 90.0},
    ], columns=df.columns)

    return pd.concat([df, aberrant], ignore_index=True)


def make_synthetic_tables(effect_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    school_years = sorted({_school_year_label(d) for d in effect_df["efdate"]})

    vacances_df = pd.DataFrame({
        "zone": ["B"],
        "school_year": [school_years[0]],
        "type_vacances": ["ete"],
        "date_debut": [pd.Timestamp("2022-07-01")],
        "date_fin": [pd.Timestamp("2022-08-31")],
    })
    jours_feries_df = pd.DataFrame({
        "date": pd.Series(dtype="datetime64[ns]"),
        "nom_jour_ferie": pd.Series(dtype="object"),
    })
    etab_rows = [
        {"uai": site, "school_year": sy, "ips": 100.0 + i * 10,
         "type_etablissement": "college", "vacances_zone": "B"}
        for i, site in enumerate(SITES)
        for sy in school_years
    ]
    etablissement_detail_df = pd.DataFrame(etab_rows)

    return {
        "effect": effect_df,
        "vacances": vacances_df,
        "jours_feries": jours_feries_df,
        "etablissement_detail": etablissement_detail_df,
    }


class _FakeCursor:
    """Faux curseur DB-API : associe chaque requête à une table synthétique via un mot-clé."""

    def __init__(self, tables: dict[str, pd.DataFrame]):
        self._tables = tables
        self._current: pd.DataFrame | None = None

    def execute(self, query, *args, **kwargs):
        q = query.lower()
        if "etablissement_detail" in q:
            self._current = self._tables["etablissement_detail"]
        elif "jours_feries" in q:
            self._current = self._tables["jours_feries"]
        elif "vacances" in q:
            self._current = self._tables["vacances"]
        elif "effect" in q:
            self._current = self._tables["effect"]
        else:
            raise AssertionError(f"Requête Trino inattendue dans les tests : {query[:200]!r}")

    @property
    def description(self):
        return [(col,) for col in self._current.columns]

    def fetchall(self):
        return list(self._current.itertuples(index=False, name=None))


class _FakeConnection:
    def __init__(self, tables: dict[str, pd.DataFrame]):
        self._tables = tables

    def cursor(self):
        return _FakeCursor(self._tables)


@pytest.fixture()
def synthetic_effect_df() -> pd.DataFrame:
    return make_synthetic_effect_df()


@pytest.fixture()
def synthetic_tables(synthetic_effect_df) -> dict[str, pd.DataFrame]:
    return make_synthetic_tables(synthetic_effect_df)


def build_data_preparation(monkeypatch, synthetic_tables, **dp_kwargs):
    """Construit un DataPreparation dont la connexion Trino est mockée."""
    import prediction_passages.src.data_prep as data_prep_module

    conn = _FakeConnection(synthetic_tables)
    monkeypatch.setattr(data_prep_module, "connect", lambda **kwargs: conn)
    monkeypatch.setenv("OVH_API_KEY", "test-key")
    monkeypatch.setenv("OVH_SECRET_KEY", "test-secret")

    dp_kwargs.setdefault("prefix", "wg_test_")
    dp_kwargs.setdefault("env", "prodcentre")
    return data_prep_module.DataPreparation(**dp_kwargs)


@pytest.fixture()
def prepared_df(monkeypatch, synthetic_tables):
    """DataFrame préparé (use_manual_entry=True, comportement par défaut)."""
    dp = build_data_preparation(monkeypatch, synthetic_tables, use_manual_entry=True)
    return dp.load_and_prepare(), dp

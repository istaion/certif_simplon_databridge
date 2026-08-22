"""Requêtes Trino pour les routes API stats WeResto."""

from __future__ import annotations

import math
import threading
import time as _time

import pandas as pd

from data_process.db.trino_client import TrinoClient


# ── Helpers SQL ───────────────────────────────────────────────────────────────

def _esc(v: str) -> str:
    return v.replace("'", "''")


def _in_clause(col: str, values: list[str]) -> str:
    escaped = ", ".join(f"'{_esc(v)}'" for v in values)
    return f"{col} IN ({escaped})"


# ── Builders de clause WHERE ──────────────────────────────────────────────────

def _where_tarif3(
    school_year: str,
    nom_etablissement: list[str] | None,
    facturation_type: list[str] | None,
    access_software: list[str] | None,
    ips_min: float | None,
    ips_max: float | None,
    date_debut: str | None,
    date_fin: str | None,
) -> str:
    """tarification_3 n'a pas de colonnes department ni type."""
    conds = [f"school_year = '{_esc(school_year)}'"]
    if nom_etablissement:
        conds.append(_in_clause("nom_etablissement", nom_etablissement))
    if facturation_type:
        conds.append(_in_clause("facturation_type", facturation_type))
    if access_software:
        conds.append(_in_clause("access_software", access_software))
    if ips_min is not None:
        conds.append(f"ips >= {ips_min}")
    if ips_max is not None:
        conds.append(f"ips <= {ips_max}")
    if date_debut:
        conds.append(f"date >= TIMESTAMP '{_esc(date_debut)} 00:00:00'")
    if date_fin:
        conds.append(f"date < TIMESTAMP '{_esc(date_fin)} 00:00:00'")
    return "WHERE " + " AND ".join(conds)


def _where_suivi(
    school_year: str,
    nom_etablissement: list[str] | None,
    department: list[str] | None,
    type_orga: list[str] | None,
    access_software: list[str] | None,
    ips_min: float | None,
    ips_max: float | None,
) -> str:
    """suivi_inscriptions/validations n'ont pas de colonne facturation_type."""
    conds = [f"school_year = '{_esc(school_year)}'"]
    if nom_etablissement:
        conds.append(_in_clause("nom_etablissement", nom_etablissement))
    if department:
        conds.append(_in_clause("department", department))
    if type_orga:
        conds.append(_in_clause("type", type_orga))
    if access_software:
        conds.append(_in_clause("access_software", access_software))
    if ips_min is not None:
        conds.append(f"ips >= {ips_min}")
    if ips_max is not None:
        conds.append(f"ips <= {ips_max}")
    return "WHERE " + " AND ".join(conds)


# ── Utilitaire DataFrame → JSON ───────────────────────────────────────────────

def _df_to_records(df: pd.DataFrame) -> list[dict]:
    def _clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


# ── Helpers table ────────────────────────────────────────────────────────────

def _resolve_school_year_id(db: TrinoClient, prefix: str, school_year: str) -> int:
    df = db.query_as_dataframe(
        f"SELECT school_year_id FROM {prefix}school_year WHERE label = '{_esc(school_year)}'"
    )
    if df is None or df.empty:
        raise ValueError(f"school_year {school_year!r} introuvable dans {prefix}school_year")
    return int(df["school_year_id"].iloc[0])


_sy_id_cache: dict[tuple, int] = {}


def resolve_school_year_id(
    ec: str, prefix: str, ovh_api_key: str, ovh_secret_key: str, school_year: str
) -> int:
    key = (ec, prefix, school_year)
    if key not in _sy_id_cache:
        _sy_id_cache[key] = _resolve_school_year_id(
            TrinoClient(ec, ovh_api_key, ovh_secret_key), prefix, school_year
        )
    return _sy_id_cache[key]


# ── Cache réponse TTL ─────────────────────────────────────────────────────────

_CACHE_TTL = 72000.0  # 20h — données rafraîchies uniquement la nuit
_response_cache: dict[tuple, tuple] = {}   # key → (payload, timestamp)
_cache_lock = threading.Lock()


def _cache_key(*args) -> tuple:
    def _freeze(v):
        return tuple(sorted(v)) if isinstance(v, list) else v
    return tuple(_freeze(a) for a in args)


def get_cached_response(key: tuple) -> dict | None:
    with _cache_lock:
        entry = _response_cache.get(key)
        if entry and (_time.time() - entry[1]) < _CACHE_TTL:
            return entry[0]
    return None


def set_cached_response(key: tuple, payload: dict) -> None:
    with _cache_lock:
        _response_cache[key] = (payload, _time.time())


# ── Cache table TTL ───────────────────────────────────────────────────────────
# Charge la table complète une seule fois par school_year_id (TTL 10 min).
# Toutes les combinaisons de filtres sont ensuite appliquées en pandas (~1 ms).

_TABLE_CACHE_TTL = 72000.0  # 20h — données rafraîchies uniquement la nuit
_table_cache: dict[tuple, tuple] = {}  # (ec, prefix, sy_id, name) → (DataFrame, ts)
_table_lock = threading.Lock()


def _load_table_cached(
    ec: str, prefix: str, ovh_api_key: str, ovh_secret_key: str,
    cache_key: tuple, sql: str,
) -> pd.DataFrame:
    with _table_lock:
        entry = _table_cache.get(cache_key)
        if entry and (_time.time() - entry[1]) < _TABLE_CACHE_TTL:
            return entry[0]
    df = TrinoClient(ec, ovh_api_key, ovh_secret_key).query_as_dataframe(sql)
    df = df if df is not None else pd.DataFrame()
    with _table_lock:
        _table_cache[cache_key] = (df, _time.time())
    return df


def _apply_filters(
    df: pd.DataFrame,
    nom_etablissement: list[str] | None = None,
    facturation_type: list[str] | None = None,
    department: list[str] | None = None,
    type_orga: list[str] | None = None,
    access_software: list[str] | None = None,
    ips_min: float | None = None,
    ips_max: float | None = None,
    id_organization: list[int] | None = None,
    service: list[str] | None = None,
    date_debut: str | None = None,
    date_fin: str | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df
    # id_organization et nom_etablissement sont deux façons d'identifier le même
    # établissement — on les union pour que les deux ne soient pas exclusifs.
    if id_organization or nom_etablissement:
        if "id_organization" in df.columns:
            combined: set = set(id_organization or [])
            if nom_etablissement and "nom_etablissement" in df.columns:
                combined |= set(
                    df.loc[df["nom_etablissement"].isin(nom_etablissement), "id_organization"]
                    .dropna().unique()
                )
            if combined:
                df = df[df["id_organization"].isin(combined)]
        elif nom_etablissement and "nom_etablissement" in df.columns:
            df = df[df["nom_etablissement"].isin(nom_etablissement)]
    if facturation_type and "facturation_type" in df.columns:
        df = df[df["facturation_type"].isin(facturation_type)]
    if department and "department" in df.columns:
        df = df[df["department"].isin(department)]
    if type_orga and "type" in df.columns:
        df = df[df["type"].isin(type_orga)]
    if access_software and "access_software" in df.columns:
        df = df[df["access_software"].isin(access_software)]
    if ips_min is not None and "ips" in df.columns:
        df = df[df["ips"] >= ips_min]
    if ips_max is not None and "ips" in df.columns:
        df = df[df["ips"] <= ips_max]
    if service and "service" in df.columns:
        df = df[df["service"].isin(service)]
    if date_debut and "date" in df.columns:
        df = df[df["date"] >= pd.Timestamp(date_debut)]
    if date_fin and "date" in df.columns:
        df = df[df["date"] < pd.Timestamp(date_fin)]
    return df


_KPI_COLS = [
    "dossiers_deposes", "connexion_sans_depot", "dossiers_valides",
    "dossiers_en_attente", "dossiers_a_corriger",
    "transmission_automatique", "transmission_manuelle", "refus_transmission",
    "pas_donnees_fournies", "import_gestionnaire", "avis_impot",
    "identite_pivot", "import_departemental",
]

_FILTER_COLS_CATEGORICAL = [
    "facturation_type", "department", "type",
    "access_software", "label_group",
]


def query_available_filters(
    ec: str, prefix: str, ovh_api_key: str, ovh_secret_key: str,
    school_year_id: int,
    page: str,
) -> dict:
    if page != "inscription":
        raise ValueError(f"page={page!r} non supportée")

    df = _load_table_cached(
        ec, prefix, ovh_api_key, ovh_secret_key,
        (ec, prefix, school_year_id, "tarif1"),
        f"SELECT * FROM {prefix}tarification_1_sc{school_year_id}",
    )

    result: dict = {}

    if "id_organization" in df.columns and "nom_etablissement" in df.columns:
        result["etablissement"] = (
            df[["id_organization", "nom_etablissement"]]
            .drop_duplicates()
            .dropna()
            .sort_values("nom_etablissement")
            .apply(lambda r: [int(r["id_organization"]), r["nom_etablissement"]], axis=1)
            .tolist()
        )

    for col in _FILTER_COLS_CATEGORICAL:
        if col in df.columns:
            result[col] = sorted(df[col].dropna().unique().tolist())

    if "ips" in df.columns:
        ips = df["ips"].dropna()
        result["ips"] = {"min": float(ips.min()), "max": float(ips.max())} if not ips.empty else None

    return result


# ── Requêtes individuelles ────────────────────────────────────────────────────

def query_kpis_and_par_tranche(
    ec: str, prefix: str, ovh_api_key: str, ovh_secret_key: str,
    school_year_id: int,
    nom_etablissement: list[str] | None,
    facturation_type: list[str] | None,
    department: list[str] | None,
    type_orga: list[str] | None,
    access_software: list[str] | None,
    ips_min: float | None,
    ips_max: float | None,
    id_organization: list[int] | None = None,
) -> tuple[dict, pd.DataFrame]:
    df = _load_table_cached(
        ec, prefix, ovh_api_key, ovh_secret_key,
        (ec, prefix, school_year_id, "tarif1"),
        f"SELECT * FROM {prefix}tarification_1_sc{school_year_id}",
    )
    df = _apply_filters(df, nom_etablissement, facturation_type, department,
                        type_orga, access_software, ips_min, ips_max, id_organization)

    kpis = (
        {c: int(df[c].sum()) for c in _KPI_COLS}
        if not df.empty
        else {c: 0 for c in _KPI_COLS}
    )
    if df.empty:
        par_tranche = pd.DataFrame(columns=["tranche", "facturation_type", "dossiers_valides_sum"])
    else:
        par_tranche = (
            df.groupby(["tranche", "facturation_type"], as_index=False)["dossiers_valides"]
            .sum()
            .rename(columns={"dossiers_valides": "dossiers_valides_sum"})
        )
    return kpis, par_tranche


_RECOURS_FACTURATION = ['ticket', 'interne']


def query_recours_kpis_par_tranche(
    ec: str, prefix: str, ovh_api_key: str, ovh_secret_key: str,
    school_year_id: int,
    nom_etablissement: list[str] | None,
    department: list[str] | None,
    type_orga: list[str] | None,
    access_software: list[str] | None,
    ips_min: float | None,
    ips_max: float | None,
    id_organization: list[int] | None = None,
    facturation_type: list[str] | None = None,
) -> tuple[dict, pd.DataFrame]:
    df = _load_table_cached(
        ec, prefix, ovh_api_key, ovh_secret_key,
        (ec, prefix, school_year_id, "tarif1"),
        f"SELECT * FROM {prefix}tarification_1_sc{school_year_id}",
    )
    if "facturation_type" in df.columns:
        df = df[df["facturation_type"].isin(_RECOURS_FACTURATION)]
    df = _apply_filters(df, nom_etablissement, facturation_type=facturation_type,
                        department=department, type_orga=type_orga,
                        access_software=access_software,
                        ips_min=ips_min, ips_max=ips_max, id_organization=id_organization)

    if df.empty:
        return (
            {"dossiers_valides": 0, "dossiers_valides_interne": 0},
            pd.DataFrame(columns=["tranche", "facturation_type", "dossiers_valides_sum"]),
        )

    kpis = {
        "dossiers_valides": int(df["dossiers_valides"].sum()),
        "dossiers_valides_interne": int(
            df.loc[df["facturation_type"] == "interne", "dossiers_valides"].sum()
        ),
    }
    par_tranche = (
        df.groupby(["tranche", "facturation_type"], as_index=False)["dossiers_valides"]
        .sum()
        .rename(columns={"dossiers_valides": "dossiers_valides_sum"})
    )
    return kpis, par_tranche


def query_general_tarif2(
    ec: str, prefix: str, ovh_api_key: str, ovh_secret_key: str,
    school_year: str,
    nom_etablissement: list[str] | None,
    department: list[str] | None,
    type_orga: list[str] | None,
    access_software: list[str] | None,
    school_year_id: int | None = None,
    id_organization: list[int] | None = None,
) -> pd.DataFrame:
    if school_year_id is None:
        school_year_id = resolve_school_year_id(
            ec, prefix, ovh_api_key, ovh_secret_key, school_year,
        )
    df = _load_table_cached(
        ec, prefix, ovh_api_key, ovh_secret_key,
        (ec, prefix, school_year_id, "tarif2"),
        f"SELECT * FROM {prefix}tarification_2_sy{school_year_id}",
    )
    df = _apply_filters(df, nom_etablissement, department=department,
                        type_orga=type_orga, access_software=access_software,
                        id_organization=id_organization)
    if df.empty:
        return pd.DataFrame(columns=["tranche", "nom_sous_groupe", "dossiers_valides_sum"])
    return (
        df.assign(_valides=df["nb_validated"] + df["nb_merged"])
        .groupby(["tranche", "nom_sous_groupe"], as_index=False)["_valides"]
        .sum()
        .rename(columns={"_valides": "dossiers_valides_sum"})
    )


def query_general_enrollment(
    ec: str, prefix: str, ovh_api_key: str, ovh_secret_key: str,
    school_year: str,
    nom_etablissement: list[str] | None,
    department: list[str] | None,
    type_orga: list[str] | None,
    access_software: list[str] | None,
    ips_min: float | None,
    ips_max: float | None,
    school_year_id: int | None = None,
    id_organization: list[int] | None = None,
) -> pd.DataFrame:
    if school_year_id is None:
        school_year_id = resolve_school_year_id(
            ec, prefix, ovh_api_key, ovh_secret_key, school_year,
        )
    df = _load_table_cached(
        ec, prefix, ovh_api_key, ovh_secret_key,
        (ec, prefix, school_year_id, "enrollment"),
        f"""SELECT
          CAST(oe.organization_id AS DOUBLE) AS id_organization,
          o.name AS nom_etablissement,
          o.department, o.type, o.access_software, o.ips,
          oe.total_enrollment, oe.social_tarif_beneficiaries, oe.intern_count
        FROM {prefix}organization_enrollment oe
        JOIN {prefix}organization o
          ON CAST(oe.organization_id AS DOUBLE) = o.id_organization
        WHERE oe.school_year_id = {school_year_id}""",
    )
    return _apply_filters(df, nom_etablissement, department=department,
                          type_orga=type_orga, access_software=access_software,
                          ips_min=ips_min, ips_max=ips_max, id_organization=id_organization)


_SUIVI_CUMUL_COLS = [
    "total_connexions_cumul", "dossiers_deposes_cumul", "dossiers_valides_cumul",
    "dossiers_valides_cumul_tranche1", "dossiers_valides_cumul_tranche2",
    "dossiers_valides_cumul_tranche3", "dossiers_valides_cumul_tranche4",
    "dossiers_valides_cumul_hors_tranche",
]


def query_recours_inscriptions(
    ec: str, prefix: str, ovh_api_key: str, ovh_secret_key: str,
    school_year: str,
    nom_etablissement: list[str] | None,
    department: list[str] | None,
    type_orga: list[str] | None,
    access_software: list[str] | None,
    ips_min: float | None,
    ips_max: float | None,
    school_year_id: int | None = None,
    id_organization: list[int] | None = None,
) -> pd.DataFrame:
    if school_year_id is None:
        school_year_id = resolve_school_year_id(
            ec, prefix, ovh_api_key, ovh_secret_key, school_year,
        )
    df = _load_table_cached(
        ec, prefix, ovh_api_key, ovh_secret_key,
        (ec, prefix, school_year_id, "suivi_inscriptions"),
        f"SELECT * FROM {prefix}suivi_inscriptions_sy{school_year_id}",
    )
    df = _apply_filters(df, nom_etablissement, department=department,
                        type_orga=type_orga, access_software=access_software,
                        ips_min=ips_min, ips_max=ips_max, id_organization=id_organization)
    if df.empty:
        return pd.DataFrame(columns=["jour"] + _SUIVI_CUMUL_COLS)
    existing = [c for c in _SUIVI_CUMUL_COLS if c in df.columns]
    return (
        df.groupby("jour", as_index=False)[existing]
        .sum()
        .sort_values("jour")
        .reset_index(drop=True)
    )


_SUIVI_VALID_CUMUL_COLS = [
    "dossiers_valides_cumul",
    "dossiers_valides_tranche1_cumul", "dossiers_valides_tranche2_cumul",
    "dossiers_valides_tranche3_cumul", "dossiers_valides_tranche4_cumul",
]


def query_recours_validations(
    ec: str, prefix: str, ovh_api_key: str, ovh_secret_key: str,
    school_year: str,
    nom_etablissement: list[str] | None,
    department: list[str] | None,
    type_orga: list[str] | None,
    access_software: list[str] | None,
    ips_min: float | None,
    ips_max: float | None,
    school_year_id: int | None = None,
    id_organization: list[int] | None = None,
) -> pd.DataFrame:
    if school_year_id is None:
        school_year_id = resolve_school_year_id(
            ec, prefix, ovh_api_key, ovh_secret_key, school_year,
        )
    df = _load_table_cached(
        ec, prefix, ovh_api_key, ovh_secret_key,
        (ec, prefix, school_year_id, "suivi_validations"),
        f"SELECT * FROM {prefix}suivi_validations_sy{school_year_id}",
    )
    df = _apply_filters(df, nom_etablissement, department=department,
                        type_orga=type_orga, access_software=access_software,
                        ips_min=ips_min, ips_max=ips_max, id_organization=id_organization)
    if df.empty:
        return pd.DataFrame(columns=["jour"] + _SUIVI_VALID_CUMUL_COLS)
    existing = [c for c in _SUIVI_VALID_CUMUL_COLS if c in df.columns]
    return (
        df.groupby("jour", as_index=False)[existing]
        .sum()
        .sort_values("jour")
        .reset_index(drop=True)
    )


def query_passages_tarif3(
    ec: str, prefix: str, ovh_api_key: str, ovh_secret_key: str,
    school_year: str,
    nom_etablissement: list[str] | None,
    facturation_type: list[str] | None,
    access_software: list[str] | None,
    ips_min: float | None,
    ips_max: float | None,
    date_debut: str | None,
    date_fin: str | None,
) -> pd.DataFrame:
    where = _where_tarif3(
        school_year, nom_etablissement, facturation_type,
        access_software, ips_min, ips_max, date_debut, date_fin,
    )
    db = TrinoClient(ec, ovh_api_key, ovh_secret_key)
    return db.query_as_dataframe(f"""
        SELECT
          date, tranche, facturation_type, nom_etablissement, id_organization,
          SUM(nb_passages_total) AS nb_passages_total,
          MAX(effectif) AS effectif,
          MAX(effectif_cible) AS effectif_cible,
          MAX(effectif_interne) AS effectif_interne
        FROM {prefix}tarification_3
        {where}
        GROUP BY date, tranche, facturation_type, nom_etablissement, id_organization
        ORDER BY date, nom_etablissement, tranche
    """)


_PASSAGES_FACTURATION = ['ticket', 'interne']


def query_passages_tarif3_cached(
    ec: str, prefix: str, ovh_api_key: str, ovh_secret_key: str,
    school_year_id: int,
    nom_etablissement: list[str] | None,
    id_organization: list[int] | None,
    facturation_type: list[str] | None,
    access_software: list[str] | None,
    ips_min: float | None,
    ips_max: float | None,
    service: list[str] | None,
    date_debut: str | None,
    date_fin: str | None,
) -> pd.DataFrame:
    df = _load_table_cached(
        ec, prefix, ovh_api_key, ovh_secret_key,
        (ec, prefix, school_year_id, "tarif3"),
        f"SELECT * FROM {prefix}tarification_3_sy{school_year_id}",
    )
    if "facturation_type" in df.columns:
        df = df[df["facturation_type"].isin(_PASSAGES_FACTURATION)]
    return _apply_filters(
        df, nom_etablissement=nom_etablissement, id_organization=id_organization,
        facturation_type=facturation_type,
        access_software=access_software, ips_min=ips_min, ips_max=ips_max,
        service=service, date_debut=date_debut, date_fin=date_fin,
    )


def query_export_tarif1(
    ec: str, prefix: str, ovh_api_key: str, ovh_secret_key: str,
    school_year: str,
    nom_etablissement: list[str] | None,
    facturation_type: list[str] | None,
    department: list[str] | None,
    type_orga: list[str] | None,
    access_software: list[str] | None,
    ips_min: float | None,
    ips_max: float | None,
    id_organization: list[int] | None = None,
) -> pd.DataFrame:
    sy_id = resolve_school_year_id(ec, prefix, ovh_api_key, ovh_secret_key, school_year)
    df = _load_table_cached(
        ec, prefix, ovh_api_key, ovh_secret_key,
        (ec, prefix, sy_id, "tarif1"),
        f"SELECT * FROM {prefix}tarification_1_sc{sy_id}",
    )
    return _apply_filters(df, nom_etablissement, facturation_type, department,
                          type_orga, access_software, ips_min, ips_max, id_organization)


# ── Assemblage des réponses ───────────────────────────────────────────────────

def assemble_general(
    kpis: dict,
    par_tranche: pd.DataFrame,
    par_categorie: pd.DataFrame,
    df_enrollment: pd.DataFrame,
    df_suivi: pd.DataFrame,
) -> dict:
    return {
        "kpis": kpis,
        "effectif_par_etablissement": {
            "total_enrollment": int(df_enrollment["total_enrollment"].sum()),
            "social_tarif_beneficiaries": int(df_enrollment["social_tarif_beneficiaries"].sum()),
            "intern_count": int(df_enrollment["intern_count"].sum()),
        },
        "par_tranche": _df_to_records(par_tranche),
        "par_categorie": _df_to_records(par_categorie),
        "suivi_inscriptions": _df_to_records(df_suivi),
    }


def assemble_recours(
    kpis: dict,
    par_tranche: pd.DataFrame,
    df_enrollment: pd.DataFrame,
    df_validations: pd.DataFrame,
) -> dict:
    enrollment = (
        {
            "total_enrollment": int(df_enrollment["total_enrollment"].sum()),
            "social_tarif_beneficiaries": int(df_enrollment["social_tarif_beneficiaries"].sum()),
            "intern_count": int(df_enrollment["intern_count"].sum()),
        }
        if not df_enrollment.empty
        else {"total_enrollment": 0, "social_tarif_beneficiaries": 0, "intern_count": 0}
    )
    return {
        "kpis": kpis,
        "effectif_par_etablissement": enrollment,
        "par_tranche": _df_to_records(par_tranche),
        "suivi_validations": _df_to_records(df_validations),
    }


EXPORT_COL_LABELS: dict[str, str] = {
    "rne":                     "UAI",
    "nom_etablissement":       "Etablissement",
    "access_software":         "Solution tierce",
    "ips":                     "ips",
    "effectif":                "effectif",
    "effectif_cible":          "effectif cible",
    "effectif_interne":        "effectif interne",
    "dossiers_deposes":        "nombre de dossiers déposés",
    "dossiers_valides":        "nombre de dossiers validés",
    "dossiers_a_corriger":     "nombre de dossiers à corriger",
    "dossiers_en_attente":     "nombre de dossier en attente",
    "connexion_sans_depot":    "connexion sans dépôt de dossier",
    "tranche_1":               "Tranche 1",
    "tranche_2":               "Tranche 2",
    "tranche_3":               "Tranche 3",
    "tranche_4":               "Tranche 4",
    "hors_tranche":            "Hors tranche",
    "transmission_automatique":"Transmission automatique",
    "transmission_manuelle":   "Transmission manuelle",
    "pas_donnees_fournies":    "Pas de données fournies",
    "refus_transmission":      "Refus de transmission",
    "import_departemental":    "Import départemental",
    "taux_recours_effectif":        "Taux de recours (effectif)",
    "taux_recours_interne":         "Taux de recours (internes)",
    "taux_recours_effectif_cible":  "Taux de recours (effectif cible)",
}


_EXPORT_COL_ORDER = [
    "rne", "nom_etablissement", "access_software", "ips",
    "effectif", "effectif_cible", "effectif_interne",
    "dossiers_deposes", "dossiers_valides", "dossiers_a_corriger",
    "dossiers_en_attente", "connexion_sans_depot",
    "tranche_1", "tranche_2", "tranche_3", "tranche_4", "hors_tranche",
    "transmission_automatique", "transmission_manuelle", "pas_donnees_fournies",
    "refus_transmission", "import_departemental",
    "taux_recours_effectif", "taux_recours_interne", "taux_recours_effectif_cible",
]


def assemble_export(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    g_key = ["id_organization", "nom_etablissement"]

    agg_spec: dict = dict(
        dossiers_deposes=        ("dossiers_deposes",        "sum"),
        dossiers_valides=        ("dossiers_valides",        "sum"),
        dossiers_a_corriger=     ("dossiers_a_corriger",     "sum"),
        dossiers_en_attente=     ("dossiers_en_attente",     "sum"),
        connexion_sans_depot=    ("connexion_sans_depot",    "sum"),
        transmission_automatique=("transmission_automatique","sum"),
        transmission_manuelle=   ("transmission_manuelle",   "sum"),
        pas_donnees_fournies=    ("pas_donnees_fournies",    "sum"),
        refus_transmission=      ("refus_transmission",      "sum"),
        import_departemental=    ("import_departemental",    "sum"),
    )
    for col, src in [("rne", "rne"), ("access_software", "access_software")]:
        if src in df.columns:
            agg_spec[col] = (src, "first")
    for col, src in [("ips", "ips"), ("effectif", "effectif"),
                     ("effectif_cible", "effectif_cible"), ("effectif_interne", "effectif_interne")]:
        if src in df.columns:
            agg_spec[col] = (src, "max")

    base = df.groupby(g_key, as_index=False).agg(**agg_spec)

    for t, col in [
        ("Tranche 1", "tranche_1"), ("Tranche 2", "tranche_2"),
        ("Tranche 3", "tranche_3"), ("Tranche 4", "tranche_4"),
        ("Hors tranche", "hors_tranche"),
    ]:
        sub = (
            df[df["tranche"] == t]
            .groupby(g_key, as_index=False)["dossiers_valides"]
            .sum()
            .rename(columns={"dossiers_valides": col})
        )
        base = base.merge(sub, on=g_key, how="left")
        base[col] = base[col].fillna(0).astype(int)

    valides_interne = pd.Series(0, index=base.index)
    if "facturation_type" in df.columns:
        sub_i = (
            df[df["facturation_type"] == "interne"]
            .groupby(g_key, as_index=False)["dossiers_valides"]
            .sum()
            .rename(columns={"dossiers_valides": "_vi"})
        )
        base = base.merge(sub_i, on=g_key, how="left")
        valides_interne = base.pop("_vi").fillna(0)

    eff   = base["effectif"].replace(0, float("nan"))       if "effectif"         in base.columns else None
    eff_c = base["effectif_cible"].replace(0, float("nan")) if "effectif_cible"   in base.columns else None
    eff_i = base["effectif_interne"].replace(0, float("nan")) if "effectif_interne" in base.columns else None
    base["taux_recours_effectif"]       = (base["dossiers_valides"] / eff).round(4)       if eff   is not None else float("nan")
    base["taux_recours_effectif_cible"] = (base["dossiers_valides"] / eff_c).round(4)     if eff_c is not None else float("nan")
    base["taux_recours_interne"]        = (valides_interne / eff_i).round(4)               if eff_i is not None else float("nan")

    base = base.drop(columns=["id_organization"])
    available = [c for c in _EXPORT_COL_ORDER if c in base.columns]
    return base[available]


def assemble_passages(df: pd.DataFrame, df_enrollment: pd.DataFrame) -> dict:
    enrollment = (
        {
            "total_enrollment": int(df_enrollment["total_enrollment"].sum()),
            "social_tarif_beneficiaries": int(df_enrollment["social_tarif_beneficiaries"].sum()),
            "intern_count": int(df_enrollment["intern_count"].sum()),
        }
        if not df_enrollment.empty
        else {"total_enrollment": 0, "social_tarif_beneficiaries": 0, "intern_count": 0}
    )

    if df.empty:
        return {
            "kpis": {"nb_passages_total": 0},
            "effectif_par_etablissement": enrollment,
            "par_tranche": [],
            "par_tranche_et_facturation": [],
            "evolution": [],
        }

    nb_total = int(df["nb_passages_total"].sum())

    par_tranche = (
        df.groupby("tranche", as_index=False)["nb_passages_total"].sum()
        .rename(columns={"nb_passages_total": "nb_passages"})
    )

    par_tranche_et_facturation = (
        df.groupby(["tranche", "facturation_type"], as_index=False)["nb_passages_total"].sum()
        .rename(columns={"nb_passages_total": "nb_passages"})
    )

    evolution_long = df.groupby(["date", "tranche"], as_index=False)["nb_passages_total"].sum()
    evolution = (
        evolution_long
        .pivot(index="date", columns="tranche", values="nb_passages_total")
        .fillna(0).astype(int)
        .reset_index()
        .rename(columns={
            "Tranche 1": "tranche_1", "Tranche 2": "tranche_2",
            "Tranche 3": "tranche_3", "Tranche 4": "tranche_4",
            "Hors tranche": "hors_tranche",
        })
        .sort_values("date")
    )
    evolution["date"] = evolution["date"].astype(str)

    return {
        "kpis": {"nb_passages_total": nb_total},
        "effectif_par_etablissement": enrollment,
        "par_tranche": _df_to_records(par_tranche),
        "par_tranche_et_facturation": _df_to_records(par_tranche_et_facturation),
        "evolution": _df_to_records(evolution),
    }

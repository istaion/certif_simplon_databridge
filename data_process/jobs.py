"""
Orchestrateur : lance tous les jobs de synchronisation Webresto → Trino.

Usage :
    from data_process.jobs import run_all_jobs

    results = run_all_jobs(
        base_url=os.environ["BASE_URL"],
        secret_key_webresto=os.environ["SECRET_KEY_WEBRESTO"],
        environnement_client=os.environ["ENVIRONNEMENT_CLIENT"],
        prefix_table=os.environ["PREFIX_TABLE"],
    )
    for name, r in results.items():
        print(name, r.status, r.rows_upserted, r.errors)
"""

import json
import logging
import os
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Callable, Optional

import pandas as pd
import requests

from dateutil.relativedelta import relativedelta

from data_process.db.trino_client import TrinoClient
from data_process.fetch.fetch_webresto import WebrestoFetcher
from data_process.process.access_control import (
    preprocess_passage,
    transform_passage,
    transform_passage_partner,
    transform_subgroup_mapping,
)
from data_process.process.organization import (
    preprocess_subgroup,
    transform_organization,
    transform_service,
    transform_subgroup,
)
from data_process.process.registrations import (
    preprocess_registration,
    preprocess_session,
    transform_history,
    transform_registration,
    transform_session,
    transform_vague,
)
from data_process.process.reporting import (
    build_tarification_1_sql,
    build_tarification_2_sql,
    build_tarification_3_sql,
    build_suivi_inscriptions_sql,
    build_suivi_validations_sql,
    build_constatation_sql,
    build_tarification_passages_sql,
    build_passages_detail_sql,
    build_tarification_filter_sql,
    build_dernier_passage_sql,
)
from data_process.process.finances import (
    transform_transaction,
)
from data_process.process.users import (
    preprocess_bankdetail,
    transform_bankdetail,
    transform_user,
)

logger = logging.getLogger(__name__)


# ── Noms de jobs (enum pour validation FastAPI) ───────────────────────────────

class JobName(str, Enum):
    subgroup_mapping     = "subgroup_mapping"
    organization         = "organization"
    service              = "service"
    group                = "group"
    tranche              = "tranche"
    subgroup             = "subgroup"
    vague                = "vague"
    session              = "session"
    passage              = "passage"
    passage_partner      = "passage_partner"
    history              = "history"
    registration         = "registration"
    bankdetail           = "bankdetail"
    user                 = "user"
    tarification_1          = "tarification_1"
    tarification_2          = "tarification_2"
    tarification_3          = "tarification_3"
    suivi_inscriptions      = "suivi_inscriptions"
    suivi_validations       = "suivi_validations"
    constatation_reporting   = "constatation_reporting"
    tarification_passages    = "tarification_passages"
    passages_detail          = "passages_detail"
    tarification_filter      = "tarification_filter"
    dernier_passage          = "dernier_passage"
    user_info_school_year   = "user_info_school_year"
    registration_enrich     = "registration_enrich"
    etablissement_detail    = "etablissement_detail"
    vacances_scolaires      = "vacances_scolaires"
    jours_feries            = "jours_feries"
    booking                 = "booking"


# ── Résultat d'un job ─────────────────────────────────────────────────────────

@dataclass
class JobResult:
    success: bool = False
    status: str = "pending"         # complete_success | partial_success | failed
    rows_upserted: int = 0
    periods_processed: int = 0      # pertinent pour les jobs incrémentaux
    last_successful_date: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def summary(self) -> str:
        parts = [f"status={self.status}", f"rows={self.rows_upserted}"]
        if self.periods_processed:
            parts.append(f"periods={self.periods_processed}")
        if self.errors:
            parts.append(f"errors={len(self.errors)}")
        if self.warnings:
            parts.append(f"warnings={len(self.warnings)}")
        parts.append(f"duration={self.duration_seconds:.1f}s")
        return " | ".join(parts)


# ── Runners internes ──────────────────────────────────────────────────────────

def _run_full_reload_job(
    name: str,
    fetcher: WebrestoFetcher,
    db: TrinoClient,
    endpoint: str,
    table: str,
    method: str,
    body: Optional[dict],
    preprocess: Optional[Callable],
    transform: Callable,
    environnement_client: str,
    empty_on_400: bool = False,
) -> JobResult:
    """Full reload : TRUNCATE puis bulk_insert de toutes les données."""
    result = JobResult()
    warnings: list[str] = []
    t0 = time.time()

    try:
        logger.info(f"[{name}] Démarrage (full reload) → {endpoint}")

        effective_preprocess = (
            (lambda items: preprocess(items, warnings)) if preprocess is not None else None
        )

        try:
            df = fetcher.fetch_as_dataframe(
                endpoint, method=method, body=body, preprocess=effective_preprocess
            )
        except requests.HTTPError as http_err:
            if empty_on_400 and http_err.response is not None and http_err.response.status_code == 400:
                logger.warning(f"[{name}] HTTP 400 traité comme liste vide (gateway sans données)")
                df = pd.DataFrame()
            else:
                raise

        if df.empty:
            msg = "DataFrame vide retourné par l'API"
            logger.error(f"[{name}] {msg}")
            result.errors.append(msg)
            result.status = "failed"
        else:
            df = transform(df, environnement_client)

            if df.empty:
                msg = "DataFrame vide après transformation"
                logger.error(f"[{name}] {msg}")
                result.errors.append(msg)
                result.status = "failed"
            else:
                db.truncate(table)
                rows = db.bulk_insert(table, df)
                result.rows_upserted = rows
                result.success = True
                result.status = "complete_success"
                logger.info(f"[{name}] OK — {rows} lignes insérées")

    except Exception as e:
        logger.exception(f"[{name}] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"

    result.warnings = warnings
    result.duration_seconds = round(time.time() - t0, 2)
    return result


def _run_incremental_job(
    name: str,
    fetcher: WebrestoFetcher,
    db: TrinoClient,
    endpoint: str,
    table: str,
    method: str,
    body_builder: Callable[[str, str], dict],
    preprocess: Optional[Callable],
    transform: Callable,
    environnement_client: str,
    column_updates: str,
    primary_keys: list[str],
    chunk_months: int = 4,
) -> JobResult:
    """
    Sync incrémentale : récupère depuis la dernière date connue jusqu'à aujourd'hui,
    par tranches de chunk_months mois. Chaque tranche est upsertée immédiatement
    (checkpoint progressif) — en cas d'échec partiel, les données déjà sauvegardées
    sont conservées.
    """
    result = JobResult()
    warnings: list[str] = []
    t0 = time.time()

    effective_preprocess = (
        (lambda items: preprocess(items, warnings)) if preprocess is not None else None
    )

    try:
        logger.info(f"[{name}] Démarrage (incrémental, chunks={chunk_months}m) → {endpoint}")

        t_max = time.time()
        last_updated = db.get_last_updated_at(table, column_updates)
        logger.info(f"[{name}] MAX({column_updates}) en {time.time() - t_max:.1f}s → {last_updated}")

        updated_before = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info(f"[{name}] Plage : {last_updated} → {updated_before}")

        current_start = datetime.strptime(last_updated, "%Y-%m-%d")
        end_date = datetime.strptime(updated_before, "%Y-%m-%d")

        total_upserted = 0
        periods_processed = 0
        last_successful_date = last_updated

        while current_start < end_date:
            current_end = min(current_start + relativedelta(months=chunk_months), end_date)
            period_str = (
                f"{current_start.strftime('%Y-%m-%d')} → {current_end.strftime('%Y-%m-%d')}"
            )

            try:
                logger.info(f"[{name}] Période {periods_processed + 1}: {period_str}")
                body = body_builder(
                    current_start.strftime("%Y-%m-%d"),
                    current_end.strftime("%Y-%m-%d"),
                )
                t_fetch = time.time()
                df = fetcher.fetch_as_dataframe(
                    endpoint, method=method, body=body, preprocess=effective_preprocess
                )
                logger.info(f"[{name}] Fetch en {time.time() - t_fetch:.1f}s")

                if df.empty:
                    logger.info(f"[{name}] Aucune donnée pour cette période")
                else:
                    logger.info(f"[{name}] {len(df)} lignes brutes récupérées")

                    # ── Soft deletes ──────────────────────────────────────────
                    if "deletedAt" in df.columns:
                        deleted_mask = df["deletedAt"].notna()
                        n_deleted = int(deleted_mask.sum())
                        if n_deleted > 0:
                            logger.info(f"[{name}] {n_deleted} soft-deleted détectés → suppression Trino")
                            try:
                                df_to_delete = transform(df[deleted_mask].copy(), environnement_client)
                                if not df_to_delete.empty:
                                    t_del = time.time()
                                    suppressed = db.delete_rows(table, primary_keys, df_to_delete)
                                    logger.info(f"[{name}] {suppressed} lignes supprimées en {time.time() - t_del:.1f}s")
                            except Exception as del_err:
                                logger.warning(f"[{name}] Échec suppression soft-deleted : {del_err}")
                        df = df[~deleted_mask].copy()

                    # ── Upsert des lignes actives ─────────────────────────────
                    df = transform(df, environnement_client)
                    if not df.empty:
                        t_upsert = time.time()
                        upserted = db.upsert(table, primary_keys, df)
                        total_upserted += upserted
                        logger.info(f"[{name}] {upserted} lignes upsertées en {time.time() - t_upsert:.1f}s")

                last_successful_date = current_end.strftime("%Y-%m-%d")
                periods_processed += 1
                logger.info(f"[{name}] Checkpoint : données à jour jusqu'au {last_successful_date}")

            except Exception as e:
                logger.error(f"[{name}] Erreur sur période {period_str}: {e}")
                result.errors.append(f"Période {period_str} : {e}")
                result.status = "partial_success"
                result.rows_upserted = total_upserted
                result.periods_processed = periods_processed
                result.last_successful_date = last_successful_date
                result.warnings = warnings
                result.duration_seconds = round(time.time() - t0, 2)
                logger.warning(
                    f"[{name}] Arrêt anticipé. Données conservées jusqu'au {last_successful_date}"
                )
                return result

            current_start = current_end

        result.success = True
        result.status = "complete_success"
        result.rows_upserted = total_upserted
        result.periods_processed = periods_processed
        result.last_successful_date = last_successful_date
        logger.info(
            f"[{name}] OK — {periods_processed} périodes, {total_upserted} lignes upsertées"
        )

    except Exception as e:
        logger.exception(f"[{name}] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"

    result.warnings = warnings
    result.duration_seconds = round(time.time() - t0, 2)
    return result


def _run_full_reload_incremental_job(
    name: str,
    fetcher: WebrestoFetcher,
    db: TrinoClient,
    endpoint: str,
    table: str,
    method: str,
    body_builder: Callable[[str, str], dict],
    preprocess: Optional[Callable],
    transform: Callable,
    environnement_client: str,
    chunk_months: int = 4,
    start_date: str = "2020-01-01",
) -> JobResult:
    """
    Full reload incrémental : TRUNCATE une fois, puis bulk_insert par chunks depuis start_date.
    La taille des batches Trino est calculée automatiquement selon la limite de 1 MB par requête.
    En cas d'échec sur un chunk, logue l'erreur et continue (partial_success).
    """
    result = JobResult()
    warnings: list[str] = []
    t0 = time.time()

    effective_preprocess = (
        (lambda items: preprocess(items, warnings)) if preprocess is not None else None
    )

    try:
        logger.info(
            f"[{name}] Démarrage (full reload incrémental, chunks={chunk_months}m, start={start_date})"
        )

        db.truncate(table)
        logger.info(f"[{name}] Table {table} tronquée")

        updated_before = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        current_start = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_dt = datetime.strptime(updated_before, "%Y-%m-%d")

        # Pré-calcul du nombre total de chunks pour l'affichage
        _tmp = current_start
        _n_chunks = 0
        while _tmp < end_date_dt:
            _tmp = min(_tmp + relativedelta(months=chunk_months), end_date_dt)
            _n_chunks += 1

        total_inserted = 0
        periods_processed = 0
        last_successful_date = start_date

        while current_start < end_date_dt:
            current_end = min(current_start + relativedelta(months=chunk_months), end_date_dt)
            period_str = (
                f"{current_start.strftime('%Y-%m-%d')} → {current_end.strftime('%Y-%m-%d')}"
            )

            try:
                chunk_num = periods_processed + 1
                print(
                    f"  [{name}] chunk {chunk_num}/{_n_chunks}  {period_str} ...",
                    end=" ", flush=True,
                )
                logger.info(f"[{name}] Période {chunk_num}: {period_str}")
                body = body_builder(
                    current_start.strftime("%Y-%m-%d"),
                    current_end.strftime("%Y-%m-%d"),
                )
                df = fetcher.fetch_as_dataframe(
                    endpoint, method=method, body=body, preprocess=effective_preprocess
                )

                if df.empty:
                    print("0 lignes")
                    logger.info(f"[{name}] Aucune donnée pour cette période")
                else:
                    logger.info(f"[{name}] {len(df)} lignes brutes récupérées")
                    df = transform(df, environnement_client)
                    if not df.empty:
                        inserted = db.bulk_insert(table, df)
                        total_inserted += inserted
                        print(f"{inserted} lignes insérées")
                        logger.info(f"[{name}] {inserted} lignes insérées")
                    else:
                        print("0 lignes après transform")

                last_successful_date = current_end.strftime("%Y-%m-%d")
                periods_processed += 1
                logger.info(f"[{name}] Checkpoint : données à jour jusqu'au {last_successful_date}")

            except Exception as e:
                print(f"ERREUR: {e}")
                logger.error(f"[{name}] Erreur sur période {period_str}: {e}")
                result.errors.append(f"Période {period_str} : {e}")

            current_start = current_end

        if result.errors:
            result.status = "partial_success"
        else:
            result.success = True
            result.status = "complete_success"

        result.rows_upserted = total_inserted
        result.periods_processed = periods_processed
        result.last_successful_date = last_successful_date
        logger.info(
            f"[{name}] Terminé — {periods_processed} périodes, {total_inserted} lignes insérées"
        )

    except Exception as e:
        logger.exception(f"[{name}] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"

    result.warnings = warnings
    result.duration_seconds = round(time.time() - t0, 2)
    return result


def _registration_runner(
    fetcher: WebrestoFetcher, db: TrinoClient, p: str, environnement_client: str
) -> JobResult:
    """Lance le job registration en résolvant id_school_year via la table vague."""
    vague_df = db.query_as_dataframe(f"SELECT id_vague, id_school_year FROM {p}vague")
    vague_map = vague_df.set_index("id_vague")["id_school_year"].to_dict()
    logger.info(f"[registration] vague_map : {vague_map}")
    return _run_incremental_job(
        name="registration", fetcher=fetcher, db=db,
        endpoint="/findAll/registrations", table=f"{p}registration",
        method="POST",
        body_builder=lambda s, e: {"updatedSince": s, "updatedBefore": e},
        preprocess=preprocess_registration,
        transform=lambda df, env: transform_registration(df, env, vague_map=vague_map),
        environnement_client=environnement_client,
        column_updates="updated_at", primary_keys=["id"], chunk_months=2,
    )


def _run_sql_job(name: str, db: TrinoClient, sql: str) -> JobResult:
    """Exécute une requête SQL pure (ex: CREATE OR REPLACE TABLE AS)."""
    result = JobResult()
    t0 = time.time()
    try:
        logger.info(f"[{name}] Démarrage (SQL)")
        rows = db.run_query(sql)
        result.rows_upserted = rows
        result.success = True
        result.status = "complete_success"
        logger.info(f"[{name}] OK — {rows} lignes")
    except Exception as e:
        logger.exception(f"[{name}] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"
    result.duration_seconds = round(time.time() - t0, 2)
    return result


def _run_multi_sql_job(name: str, db: TrinoClient, sql_list: list[str]) -> JobResult:
    """Exécute une liste de requêtes SQL séquentiellement (ex: plusieurs CREATE OR REPLACE)."""
    result = JobResult()
    t0 = time.time()
    total_rows = 0
    try:
        logger.info(f"[{name}] Démarrage ({len(sql_list)} requêtes)")
        for i, sql in enumerate(sql_list):
            rows = db.run_query(sql)
            total_rows += rows or 0
            logger.info(f"[{name}] Requête {i + 1}/{len(sql_list)} OK — {rows} lignes")
        result.rows_upserted = total_rows
        result.success = True
        result.status = "complete_success"
        logger.info(f"[{name}] Terminé — {total_rows} lignes au total")
    except Exception as e:
        logger.exception(f"[{name}] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"
    result.duration_seconds = round(time.time() - t0, 2)
    return result


def _run_registration_enrich_job(db: TrinoClient, p: str) -> JobResult:
    """registration_enrich avec logs de debug pour diagnostiquer les 0 lignes."""
    result = JobResult()
    t0 = time.time()
    name = "registration_enrich"

    def _count(sql: str) -> int:
        try:
            df = db.query_as_dataframe(sql)
            return int(df.iloc[0, 0]) if not df.empty else -1
        except Exception as e:
            logger.info(f"[{name}] count error: {e}")
            return -1

    try:
        logger.info(f"[{name}] Démarrage (SQL MERGE)")

        n_to_enrich = _count(
            f"SELECT COUNT(*) FROM {p}registration WHERE id_school_year IS NULL OR subgroup_id IS NULL"
        )
        logger.info(f"[{name}] Registrations à enrichir (id_school_year NULL ou subgroup_id NULL) : {n_to_enrich}")

        n_uisy = _count(f"SELECT COUNT(*) FROM {p}user_info_school_year")
        logger.info(f"[{name}] Lignes dans user_info_school_year : {n_uisy}")

        n_matchable = _count(f"""
            SELECT COUNT(*) FROM {p}registration reg
            JOIN {p}session ses ON reg.id_session = ses.id
            JOIN {p}vague v     ON ses.id_vague   = v.id_vague
            JOIN {p}user_info_school_year uisy
                ON  reg.id_user      = uisy.id_user
                AND v.id_school_year = uisy.school_year_id
            WHERE reg.id_school_year IS NULL OR reg.subgroup_id IS NULL
        """)
        logger.info(f"[{name}] Registrations matchables avec uisy : {n_matchable}")

        rows = db.run_query(f"""
            MERGE INTO {p}registration r
            USING (
                SELECT id, id_school_year, tranche_id, subgroup_id, choice_bank_detail
                FROM (
                    SELECT
                        reg.id,
                        v.id_school_year,
                        COALESCE(reg.tranche_id, uisy.id_tranche)   AS tranche_id,
                        COALESCE(reg.subgroup_id, uisy.id_subgroup) AS subgroup_id,
                        COALESCE(reg.choice_bank_detail, uisy.choice_bank_details) AS choice_bank_detail,
                        ROW_NUMBER() OVER (PARTITION BY reg.id ORDER BY v.id_school_year DESC) AS _rn
                    FROM {p}registration reg
                    JOIN {p}session ses ON reg.id_session = ses.id
                    JOIN {p}vague v     ON ses.id_vague   = v.id_vague
                    LEFT JOIN (
                        SELECT *
                        FROM (
                            SELECT *,
                                   ROW_NUMBER() OVER (
                                       PARTITION BY id_user, school_year_id
                                       ORDER BY bank_detail_id DESC NULLS LAST
                                   ) AS _rn
                            FROM {p}user_info_school_year
                        ) WHERE _rn = 1
                    ) uisy
                        ON  reg.id_user      = uisy.id_user
                        AND v.id_school_year = uisy.school_year_id
                    WHERE reg.id_school_year IS NULL OR reg.subgroup_id IS NULL
                ) WHERE _rn = 1
            ) src
            ON r.id = src.id
            WHEN MATCHED THEN UPDATE SET
                id_school_year     = src.id_school_year,
                tranche_id         = src.tranche_id,
                subgroup_id        = src.subgroup_id,
                choice_bank_detail = src.choice_bank_detail
        """)
        logger.info(f"[{name}] rowcount MERGE (peut être 0 sur Trino) : {rows}")

        n_enriched = _count(
            f"SELECT COUNT(*) FROM {p}registration WHERE id_school_year IS NOT NULL"
        )
        logger.info(f"[{name}] Registrations avec id_school_year peuplé (total) : {n_enriched}")

        result.rows_upserted = rows
        result.success = True
        result.status = "complete_success"
        logger.info(f"[{name}] OK — cibles={n_to_enrich}, matchables={n_matchable}, enrichies_total={n_enriched}")

    except Exception as e:
        logger.exception(f"[{name}] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"

    result.duration_seconds = round(time.time() - t0, 2)
    return result


# ── Config school_year par environnement ──────────────────────────────────────

def _school_year_config_path(environnement_client: str) -> str:
    if "93" in environnement_client:
        return "data_process/config/93/school_year.json"
    return "data_process/config/centre/school_year.json"


def _load_school_year_config(environnement_client: str) -> int:
    path = _school_year_config_path(environnement_client)
    with open(path) as f:
        return int(json.load(f)["current_school_year_id"])


def _save_school_year_config(environnement_client: str, school_year_id: int) -> None:
    path = _school_year_config_path(environnement_client)
    with open(path, "w") as f:
        json.dump({"current_school_year_id": school_year_id}, f)


def _check_school_year_transition(
    db: TrinoClient, p: str, current_sy_id: int
) -> Optional[int]:
    df = db.query_as_dataframe(f"SELECT MAX(id_school_year) AS max_sy FROM {p}vague")
    if df.empty or df["max_sy"].iloc[0] is None:
        return None
    max_id = int(df["max_sy"].iloc[0])
    return max_id if max_id > current_sy_id else None


def _backup_user_info_schoolyear(
    db: TrinoClient, p: str, school_year_id: int, environnement_client: str
) -> None:
    env_suffix = "dep93" if "93" in environnement_client else "centre"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_table = (
        f"db_mg6jk45h_backup.backup"
        f".{env_suffix}_user_info_schoolyear_sy{school_year_id}_{ts}"
    )
    db.run_query(
        f"CREATE TABLE {backup_table} AS"
        f" SELECT * FROM {p}user_info_school_year"
        f" WHERE school_year_id = {school_year_id}"
    )
    logger.info(f"[user_info_school_year] Backup sy={school_year_id} → {backup_table}")


def _run_user_info_school_year_job(
    db: TrinoClient, p: str, environnement_client: str
) -> JobResult:
    current_sy_id = _load_school_year_config(environnement_client)

    new_sy_id = _check_school_year_transition(db, p, current_sy_id)
    if new_sy_id:
        _backup_user_info_schoolyear(db, p, current_sy_id, environnement_client)
        _save_school_year_config(environnement_client, new_sy_id)
        logger.info(
            f"[user_info_school_year] Transition school_year"
            f" {current_sy_id} → {new_sy_id}"
        )
        current_sy_id = new_sy_id

    return _run_sql_job(
        name="user_info_school_year",
        db=db,
        sql=f"""
            MERGE INTO {p}user_info_school_year target
            USING (
                SELECT
                    CAST(CONCAT(
                        CAST({current_sy_id} AS VARCHAR),
                        CAST(CAST(u.id_user AS BIGINT) AS VARCHAR)
                    ) AS BIGINT) AS id,
                    {current_sy_id} AS school_year_id,
                    u.id_user,
                    u.id_subgroup,
                    b.bank_detail_id,
                    b.choice_bank_details,
                    b.id_tranche,
                    t.label   AS label_tranche,
                    sg.label  AS label_subgroup,
                    grp.label AS label_group,
                    u.created_at,
                    GREATEST(u.updated_at, COALESCE(b.updated_at, u.updated_at)) AS updated_at
                FROM {p}user u
                LEFT JOIN (
                    SELECT *
                    FROM (
                        SELECT *,
                               ROW_NUMBER() OVER (
                                   PARTITION BY id_user
                                   ORDER BY bank_detail_id DESC NULLS LAST
                               ) AS _rn
                        FROM {p}bankdetail
                    ) WHERE _rn = 1
                ) b ON u.id_user = b.id_user
                LEFT JOIN {p}subgroup  sg  ON u.id_subgroup = sg.id_subgroup
                LEFT JOIN {p}group     grp ON sg.id_group   = grp.id_group
                LEFT JOIN {p}tranche   t   ON b.id_tranche  = t.id_tranche
            ) src
            ON target.id = src.id
            WHEN MATCHED THEN UPDATE SET
                id_subgroup         = src.id_subgroup,
                bank_detail_id      = src.bank_detail_id,
                choice_bank_details = src.choice_bank_details,
                id_tranche          = src.id_tranche,
                label_tranche       = src.label_tranche,
                label_subgroup      = src.label_subgroup,
                label_group         = src.label_group,
                updated_at          = src.updated_at
            WHEN NOT MATCHED THEN INSERT
                (id, school_year_id, id_user, id_subgroup, bank_detail_id,
                 choice_bank_details, id_tranche, label_tranche, label_subgroup,
                 label_group, created_at, updated_at)
            VALUES
                (src.id, src.school_year_id, src.id_user, src.id_subgroup,
                 src.bank_detail_id, src.choice_bank_details, src.id_tranche,
                 src.label_tranche, src.label_subgroup, src.label_group,
                 src.created_at, src.updated_at)
        """,
    )


# ── Registre des jobs ─────────────────────────────────────────────────────────

def _make_runners(
    fetcher: WebrestoFetcher,
    db: TrinoClient,
    p: str,
    environnement_client: str,
) -> dict[str, Callable[[], JobResult]]:
    """
    Construit le registre complet des jobs sous forme de callables () → JobResult.
    Utilisé par run_all_jobs (tous) et run_job (un seul).
    """
    return {
        # ── Full reload ───────────────────────────────────────────────────────
        "subgroup_mapping": lambda: _run_full_reload_job(
            name="subgroup_mapping", fetcher=fetcher, db=db,
            endpoint="/getSubgroupMapping", table=f"{p}subgroup_mapping",
            method="GET", body=None, preprocess=None,
            transform=transform_subgroup_mapping, environnement_client=environnement_client,
        ),
        "organization": lambda: _run_full_reload_job(
            name="organization", fetcher=fetcher, db=db,
            endpoint="/findAll/organizations", table=f"{p}organization",
            method="GET",
            body={
                "isHideDemo": "true",
                "select": "organizationId, name, rne, city, accessSoftware, type, ips, academy, vague, department",
            },
            preprocess=None,
            transform=transform_organization, environnement_client=environnement_client,
        ),
        "service": lambda: _run_full_reload_job(
            name="service", fetcher=fetcher, db=db,
            endpoint="/findAll/services", table=f"{p}service",
            method="GET", body=None, preprocess=None,
            transform=transform_service, environnement_client=environnement_client,
            empty_on_400=True,
        ),
        "subgroup": lambda: _run_full_reload_job(
            name="subgroup", fetcher=fetcher, db=db,
            endpoint="/findAll/subgroups", table=f"{p}subgroup",
            method="POST", body={}, preprocess=preprocess_subgroup,
            transform=transform_subgroup, environnement_client=environnement_client,
        ),
        "vague": lambda: _run_full_reload_job(
            name="vague", fetcher=fetcher, db=db,
            endpoint="/findAll/vagues", table=f"{p}vague",
            method="GET", body=None, preprocess=None,
            transform=transform_vague, environnement_client=environnement_client,
        ),
        "session": lambda: _run_full_reload_job(
            name="session", fetcher=fetcher, db=db,
            endpoint="/findAll/sessions", table=f"{p}session",
            method="POST", body={"relations": {"vague": True}},
            preprocess=preprocess_session,
            transform=transform_session, environnement_client=environnement_client,
        ),
        # ── Incrémental ───────────────────────────────────────────────────────
        "passage": lambda: _run_incremental_job(
            name="passage", fetcher=fetcher, db=db,
            endpoint="/findAll/passages", table=f"{p}passage",
            method="GET",
            body_builder=lambda s, e: {
                "startDate": datetime.strptime(s, "%Y-%m-%d").strftime("%d/%m/%Y"),
                "endDate":   datetime.strptime(e, "%Y-%m-%d").strftime("%d/%m/%Y"),
            },
            preprocess=preprocess_passage, transform=transform_passage,
            environnement_client=environnement_client,
            column_updates="updated_at", primary_keys=["id_passage"], chunk_months=4,
        ),
        "passage_partner": lambda: _run_incremental_job(
            name="passage_partner", fetcher=fetcher, db=db,
            endpoint="/findAll/statistic_passage_partner", table=f"{p}passage_partner",
            method="POST",
            body_builder=lambda s, e: {"period": {"startDate": s, "endDate": e}},
            preprocess=None, transform=transform_passage_partner,
            environnement_client=environnement_client,
            column_updates="date", primary_keys=["id_partner"], chunk_months=4,
        ),
        "history": lambda: _run_incremental_job(
            name="history", fetcher=fetcher, db=db,
            endpoint="/findAll/history", table=f"{p}history",
            method="POST",
            body_builder=lambda s, e: {"updatedSince": s, "updatedBefore": e},
            preprocess=None, transform=transform_history,
            environnement_client=environnement_client,
            column_updates="updated_at", primary_keys=["id_reg_history"], chunk_months=4,
        ),
        "registration": lambda: _registration_runner(fetcher, db, p, environnement_client),
        "bankdetail": lambda: _run_incremental_job(
            name="bankdetail", fetcher=fetcher, db=db,
            endpoint="/findAll/bankDetails", table=f"{p}bankdetail",
            method="GET",
            body_builder=lambda s, e: {
                "updatedSince": s,
                "updatedBefore": e,
                "selects": "createdAt, updatedAt, deletedAt, bankDetailId, trancheId, choiceBankDetails, userId",
            },
            preprocess=preprocess_bankdetail, transform=transform_bankdetail,
            environnement_client=environnement_client,
            column_updates="updated_at", primary_keys=["bank_detail_id"], chunk_months=4,
        ),
        "user": lambda: _run_incremental_job(
            name="user", fetcher=fetcher, db=db,
            endpoint="/findAll/AllUsers", table=f"{p}user",
            method="POST",
            body_builder=lambda s, e: {"updatedSince": s, "updatedBefore": e},
            preprocess=None, transform=transform_user,
            environnement_client=environnement_client,
            column_updates="updated_at", primary_keys=["id_user"], chunk_months=4,
        ),
        "transaction": lambda: _load_csv_transaction(db, p, environnement_client),
        # ── SQL pur ───────────────────────────────────────────────────────────
        "tarification_1": lambda: _run_sql_job(
            name="tarification_1",
            db=db,
            sql=build_tarification_1_sql(p, environnement_client),
        ),
        "tarification_2": lambda: _run_sql_job(
            name="tarification_2",
            db=db,
            sql=build_tarification_2_sql(p, environnement_client),
        ),
        "tarification_3": lambda: _run_sql_job(
            name="tarification_3",
            db=db,
            sql=build_tarification_3_sql(p, environnement_client),
        ),
        "suivi_inscriptions": lambda: _run_sql_job(
            name="suivi_inscriptions",
            db=db,
            sql=build_suivi_inscriptions_sql(p, environnement_client),
        ),
        "suivi_validations": lambda: _run_sql_job(
            name="suivi_validations",
            db=db,
            sql=build_suivi_validations_sql(p, environnement_client),
        ),
        "constatation_reporting": lambda: _run_sql_job(
            name="constatation_reporting",
            db=db,
            sql=build_constatation_sql(p, environnement_client),
        ),
        "tarification_passages": lambda: _run_sql_job(
            name="tarification_passages",
            db=db,
            sql=build_tarification_passages_sql(p, environnement_client),
        ),
        "passages_detail": lambda: _run_sql_job(
            name="passages_detail",
            db=db,
            sql=build_passages_detail_sql(p, environnement_client),
        ),
        "tarification_filter": lambda: _run_multi_sql_job(
            name="tarification_filter",
            db=db,
            sql_list=build_tarification_filter_sql(p, environnement_client),
        ),
        "dernier_passage": lambda: _run_sql_job(
            name="dernier_passage",
            db=db,
            sql=build_dernier_passage_sql(p),
        ),
        # ── CSV statiques ─────────────────────────────────────────────────────
        "group":   lambda: _load_csv_group(db, p, environnement_client),
        "tranche": lambda: _load_csv_tranche(db, p, environnement_client),
        "booking": lambda: _load_csv_booking(db, p, environnement_client),
        # ── À lancer après run_cleanup_orphans_job ───────────────────────────
        "registration_enrich": lambda: _run_registration_enrich_job(db=db, p=p),
        "user_info_school_year": lambda: _run_user_info_school_year_job(
            db=db, p=p, environnement_client=environnement_client
        ),
    }


def _make_clients(
    base_url: str,
    secret_key_webresto: str,
    environnement_client: str,
) -> tuple[WebrestoFetcher, TrinoClient]:
    fetcher = WebrestoFetcher(base_url=base_url, api_key=secret_key_webresto)
    db = TrinoClient(
        environnement_client=environnement_client,
        ovh_api_key=os.environ["OVH_API_KEY"],
        ovh_secret_key=os.environ["OVH_SECRET_KEY"],
    )
    return fetcher, db


# ── Points d'entrée publics ───────────────────────────────────────────────────

def run_all_jobs(
    base_url: str,
    secret_key_webresto: str,
    environnement_client: str,
    prefix_table: str,
) -> dict[str, JobResult]:
    """Lance tous les jobs dans l'ordre et retourne leurs résultats."""
    fetcher, db = _make_clients(base_url, secret_key_webresto, environnement_client)
    runners = _make_runners(fetcher, db, prefix_table, environnement_client)
    results = {name: runner() for name, runner in runners.items()}

    successful = [n for n, r in results.items() if r.success]
    failed = [n for n, r in results.items() if not r.success]
    total_rows = sum(r.rows_upserted for r in results.values())
    logger.info(
        f"Synchronisation terminée : {len(successful)}/{len(results)} jobs réussis, "
        f"{total_rows} lignes traitées au total"
    )
    if failed:
        logger.warning(f"Jobs en échec : {failed}")
    return results


def run_job(
    job_name: str,
    base_url: str,
    secret_key_webresto: str,
    environnement_client: str,
    prefix_table: str,
) -> JobResult:
    """Lance un seul job par son nom et retourne son résultat."""
    fetcher, db = _make_clients(base_url, secret_key_webresto, environnement_client)
    runners = _make_runners(fetcher, db, prefix_table, environnement_client)
    if job_name not in runners:
        raise ValueError(
            f"Job inconnu : {job_name!r}. Jobs disponibles : {list(runners)}"
        )
    return runners[job_name]()


# ── Full reload Webresto ─────────────────────────────────────────────────────

def _csv_dir(environnement_client: str) -> Optional[str]:
    if "93" in environnement_client:
        return "data_process/temp_data/93"
    if "centre" in environnement_client:
        return "data_process/temp_data/centre"
    return None


def _load_csv_school_year(db: TrinoClient, p: str, environnement_client: str) -> JobResult:
    result = JobResult()
    t0 = time.time()
    name = "school_year"
    table = f"{p}school_year"
    d = _csv_dir(environnement_client)
    if d is None:
        result.errors.append(f"environnement_client non reconnu : {environnement_client!r}")
        result.status = "failed"
        result.duration_seconds = 0.0
        return result
    csv_path = f"{d}/schoolYear.csv"
    try:
        df = pd.read_csv(csv_path)
        df = df.rename(columns={"schoolYearId": "school_year_id", "startDate": "start_date", "endDate": "end_date"})
        df = df[["school_year_id", "label", "start_date", "end_date"]]
        db.truncate(table)
        rows = db.bulk_insert(table, df)
        result.rows_upserted = rows
        result.success = True
        result.status = "complete_success"
        logger.info(f"[{name}] OK — {rows} lignes insérées depuis {csv_path}")
    except Exception as e:
        logger.exception(f"[{name}] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"
    result.duration_seconds = round(time.time() - t0, 2)
    return result


def _load_csv_service(db: TrinoClient, p: str, environnement_client: str) -> JobResult:
    result = JobResult()
    t0 = time.time()
    name = "service"
    table = f"{p}service"
    d = _csv_dir(environnement_client)
    if d is None:
        result.errors.append(f"environnement_client non reconnu : {environnement_client!r}")
        result.status = "failed"
        result.duration_seconds = 0.0
        return result
    csv_path = f"{d}/service.csv"
    try:
        df = pd.read_csv(csv_path)
        df = df.rename(columns={"serviceId": "id_service", "organizationId": "id_organization"})
        df = df[["id_organization", "id_service", "label"]]
        db.truncate(table)
        rows = db.bulk_insert(table, df)
        result.rows_upserted = rows
        result.success = True
        result.status = "complete_success"
        logger.info(f"[{name}] OK — {rows} lignes insérées depuis {csv_path}")
    except Exception as e:
        logger.exception(f"[{name}] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"
    result.duration_seconds = round(time.time() - t0, 2)
    return result


def _load_csv_group(db: TrinoClient, p: str, environnement_client: str) -> JobResult:
    result = JobResult()
    t0 = time.time()
    name = "group"
    table = f"{p}group"
    d = _csv_dir(environnement_client)
    if d is None:
        result.errors.append(f"environnement_client non reconnu : {environnement_client!r}")
        result.status = "failed"
        result.duration_seconds = 0.0
        return result
    csv_path = f"{d}/group.csv"
    try:
        df = pd.read_csv(csv_path)
        df = df[df["deletedAt"].isna()]
        df = df.rename(columns={"groupId": "id_group", "organizationId": "id_organization"})
        df = df[["id_group", "label", "acronym", "id_organization"]]
        db.truncate(table)
        rows = db.bulk_insert(table, df)
        result.rows_upserted = rows
        result.success = True
        result.status = "complete_success"
        logger.info(f"[{name}] OK — {rows} lignes insérées depuis {csv_path}")
    except Exception as e:
        logger.exception(f"[{name}] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"
    result.duration_seconds = round(time.time() - t0, 2)
    return result


def _load_csv_group(db: TrinoClient, p: str, environnement_client: str) -> JobResult:
    result = JobResult()
    t0 = time.time()
    name = "group"
    table = f"{p}group"
    d = _csv_dir(environnement_client)
    if d is None:
        result.errors.append(f"environnement_client non reconnu : {environnement_client!r}")
        result.status = "failed"
        result.duration_seconds = 0.0
        return result
    csv_path = f"{d}/group.csv"
    try:
        df = pd.read_csv(csv_path)
        df = df[df["deletedAt"].isna()]
        df = df.rename(columns={"groupId": "id_group", "organizationId": "id_organization"})
        df = df[["id_group", "label", "acronym", "id_organization"]]
        db.truncate(table)
        rows = db.bulk_insert(table, df)
        result.rows_upserted = rows
        result.success = True
        result.status = "complete_success"
        logger.info(f"[{name}] OK — {rows} lignes insérées depuis {csv_path}")
    except Exception as e:
        logger.exception(f"[{name}] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"
    result.duration_seconds = round(time.time() - t0, 2)
    return result


def _load_csv_tranche(db: TrinoClient, p: str, environnement_client: str) -> JobResult:
    result = JobResult()
    t0 = time.time()
    name = "tranche"
    table = f"{p}tranche"
    d = _csv_dir(environnement_client)
    if d is None:
        result.errors.append(f"environnement_client non reconnu : {environnement_client!r}")
        result.status = "failed"
        result.duration_seconds = 0.0
        return result
    csv_path = f"{d}/tranche.csv"
    try:
        df = pd.read_csv(csv_path)
        df = df.rename(columns={"trancheId": "id_tranche", "organizationId": "id_organization"})
        df = df[["id_tranche", "label", "indice", "id_organization", "quotient"]]
        # object dtype → littéraux entiers non quotés (BIGINT) ; str → VARCHAR quoté
        df["id_tranche"] = df["id_tranche"].astype(object)
        df["indice"] = df["indice"].astype(object)
        df["label"] = df["label"].astype(str)
        db.truncate(table)
        rows = db.bulk_insert(table, df)
        result.rows_upserted = rows
        result.success = True
        result.status = "complete_success"
        logger.info(f"[{name}] OK — {rows} lignes insérées depuis {csv_path}")
    except Exception as e:
        logger.exception(f"[{name}] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"
    result.duration_seconds = round(time.time() - t0, 2)
    return result


def _load_csv_booking(db: TrinoClient, p: str, environnement_client: str) -> JobResult:
    result = JobResult()
    t0 = time.time()
    name = "booking"
    table = f"{p}booking"

    if "centre" not in environnement_client:
        result.warnings.append("Booking non disponible pour cet environnement — ignoré")
        result.success = True
        result.status = "complete_success"
        result.duration_seconds = round(time.time() - t0, 2)
        return result

    csv_path = "data_process/temp_data/centre/booking.csv"
    try:
        df = pd.read_csv(csv_path)
        df = df.rename(columns={
            "bookingId": "id_booking",
            "organizationId": "id_organization",
            "userId": "id_user",
            "serviceId": "id_service",
            "createdAt": "created_at",
            "updatedAt": "updated_at",
        })
        df = df[["id_booking", "id_organization", "id_user", "id_service",
                 "created_at", "updated_at", "origin", "date"]]
        df["created_at"] = pd.to_datetime(df["created_at"], format="mixed")
        df["updated_at"] = pd.to_datetime(df["updated_at"], format="mixed")
        df["date"] = pd.to_datetime(df["date"])
        db.truncate(table)
        rows = db.bulk_insert(table, df)
        result.rows_upserted = rows
        result.success = True
        result.status = "complete_success"
        logger.info(f"[{name}] OK — {rows} lignes insérées depuis {csv_path}")
    except Exception as e:
        logger.exception(f"[{name}] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"
    result.duration_seconds = round(time.time() - t0, 2)
    return result


def _load_csv_transaction(db: TrinoClient, p: str, environnement_client: str) -> JobResult:
    result = JobResult()
    t0 = time.time()
    name = "transaction"
    table = f"{p}transaction"
    d = _csv_dir(environnement_client)
    if d is None:
        result.errors.append(f"environnement_client non reconnu : {environnement_client!r}")
        result.status = "failed"
        result.duration_seconds = 0.0
        return result
    csv_path = f"{d}/transaction.csv"
    try:
        df = pd.read_csv(csv_path)
        df = transform_transaction(df, environnement_client)
        if df.empty:
            result.errors.append("CSV vide après transformation")
            result.status = "failed"
            result.duration_seconds = round(time.time() - t0, 2)
            return result
        db.truncate(table)
        rows = db.bulk_insert(table, df)
        result.rows_upserted = rows
        result.success = True
        result.status = "complete_success"
        logger.info(f"[{name}] OK — {rows} lignes insérées depuis {csv_path}")
    except Exception as e:
        logger.exception(f"[{name}] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"
    result.duration_seconds = round(time.time() - t0, 2)
    return result


def _load_csv_trimester(db: TrinoClient, p: str, environnement_client: str) -> JobResult:
    result = JobResult()
    t0 = time.time()
    name = "trimester"
    table = f"{p}trimester"
    d = _csv_dir(environnement_client)
    if d is None:
        result.errors.append(f"environnement_client non reconnu : {environnement_client!r}")
        result.status = "failed"
        result.duration_seconds = 0.0
        return result
    csv_path = f"{d}/trimester.csv"
    try:
        df = pd.read_csv(csv_path)
        df = df.rename(columns={
            "trimesterId":    "trimester_id",
            "organizationId": "organization_id",
            "startDate":      "start_date",
            "endDate":        "end_date",
            "schoolYear":     "school_year",
            "createdAt":      "created_at",
            "updatedAt":      "updated_at",
        })
        df = df[["created_at", "organization_id", "trimester_id", "updated_at",
                 "start_date", "end_date", "school_year", "index"]]
        db.truncate(table)
        rows = db.bulk_insert(table, df)
        result.rows_upserted = rows
        result.success = True
        result.status = "complete_success"
        logger.info(f"[{name}] OK — {rows} lignes insérées depuis {csv_path}")
    except Exception as e:
        logger.exception(f"[{name}] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"
    result.duration_seconds = round(time.time() - t0, 2)
    return result


def _load_csv_organization_enrollment(
    db: TrinoClient, p: str, environnement_client: str
) -> JobResult:
    result = JobResult()
    t0 = time.time()
    name = "organization_enrollment"
    table = f"{p}organization_enrollment"
    d = _csv_dir(environnement_client)
    if d is None:
        result.errors.append(f"environnement_client non reconnu : {environnement_client!r}")
        result.status = "failed"
        result.duration_seconds = 0.0
        return result
    csv_path = f"{d}/organizationEnrollment.csv"
    try:
        df = pd.read_csv(csv_path)
        df = df.rename(columns={
            "enrollmentId":             "enrollment_id",
            "organizationId":           "organization_id",
            "schoolYearId":             "school_year_id",
            "totalEnrollment":          "total_enrollment",
            "socialTarifBeneficiaries": "social_tarif_beneficiaries",
            "internCount":              "intern_count",
        })
        df = df[["enrollment_id", "organization_id", "school_year_id",
                 "total_enrollment", "intern_count", "social_tarif_beneficiaries"]]
        df = df.astype("int64")
        if df.empty:
            result.errors.append("CSV vide")
            result.status = "failed"
        else:
            db.truncate(table)
            rows = db.bulk_insert(table, df)
            result.rows_upserted = rows
            result.success = True
            result.status = "complete_success"
            logger.info(f"[{name}] OK — {rows} lignes insérées depuis {csv_path}")
    except Exception as e:
        logger.exception(f"[{name}] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"
    result.duration_seconds = round(time.time() - t0, 2)
    return result


def _load_csv_constatation(db: TrinoClient, p: str, environnement_client: str) -> JobResult:
    result = JobResult()
    t0 = time.time()
    name = "constatation"
    table = f"{p}constatation"

    if "centre" in environnement_client:
        result.warnings.append("Constatation non disponible pour l'environnement centre — ignoré")
        result.success = True
        result.status = "complete_success"
        result.duration_seconds = round(time.time() - t0, 2)
        return result

    csv_path = "data_process/temp_data/93/constatation.csv"
    try:
        df = pd.read_csv(csv_path)
        df = df[df["isSimulation"] == False]  # noqa: E712

        # Dernière constatation par trimestre
        idx = df.groupby("trimesterId")["constatationIndex"].idxmax()
        df = df.loc[idx]

        # Expansion des payloads JSON
        expanded_rows = []
        for row in df.itertuples():
            payload = json.loads(row.payload)
            if not payload:
                continue
            tmp = pd.DataFrame(payload)
            tmp["constatation_id"]    = row.constatationId
            tmp["trimestre_id"]       = row.trimesterId
            tmp["constatation_index"] = row.constatationIndex
            tmp["is_simulation"]      = False
            tmp["created_at"]         = row.createdAt
            tmp["updated_at"]         = row.updatedAt
            expanded_rows.append(tmp)

        if not expanded_rows:
            result.errors.append("Aucune ligne après expansion payload")
            result.status = "failed"
            result.duration_seconds = round(time.time() - t0, 2)
            return result

        df_final = pd.concat(expanded_rows, ignore_index=True)
        df_final["payload_index"] = range(len(df_final))

        # Suppression des colonnes indésirables
        _DROP = {
            "fullname", "firstName", "lastName", "birthDate", "classroom",
            "previousData", "regimeSiecle", "helpSetting", "feeAdjustments",
            "fee_adjustments", "ine",
        }
        df_final = df_final.drop(columns=[c for c in _DROP if c in df_final.columns])

        # Renommage camelCase → snake_case + group → label_group
        _RENAME = {
            "userId":              "user_id",
            "group":               "label_group",
            "trancheLabel":        "tranche_label",
            "startDate":           "start_date",
            "endDate":             "end_date",
            "trimesterDays":       "trimester_days",
            "billAmount":          "bill_amount",
            "aidAmount":           "aid_amount",
            "feeAdjustmentDays":   "fee_adjustment_days",
            "billingType":         "billing_type",
            "feeAdjustmentAmount": "fee_adjustment_amount",
        }
        df_final = df_final.rename(columns=_RENAME)

        # Ancien format : tranche (int) → tranche_label (str)
        _TRANCHE_MAP = {
            1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G",
            8: "H", 9: "I", 10: "J", 11: "K", 12: "L", 13: "M", 14: "Mbis",
        }
        if "tranche" in df_final.columns:
            if "tranche_label" not in df_final.columns:
                df_final["tranche_label"] = None
            mask = df_final["tranche_label"].isnull() & df_final["tranche"].notna()
            df_final.loc[mask, "tranche_label"] = (
                pd.to_numeric(df_final.loc[mask, "tranche"], errors="coerce")
                .map(_TRANCHE_MAP)
            )
            df_final = df_final.drop(columns=["tranche"])

        # Conversions de types
        _INT_COLS = [
            "user_id", "constatation_index", "trimester_days",
            "fee_adjustment_days", "constatation_id", "trimestre_id", "payload_index",
        ]
        for col in _INT_COLS:
            if col in df_final.columns:
                df_final[col] = pd.to_numeric(df_final[col], errors="coerce").astype("Int64")

        _FLOAT_COLS = ["bill_amount", "aid_amount", "fee_adjustment_amount"]
        for col in _FLOAT_COLS:
            if col in df_final.columns:
                df_final[col] = pd.to_numeric(df_final[col], errors="coerce")

        _DATE_COLS = ["start_date", "end_date", "created_at", "updated_at"]
        for col in _DATE_COLS:
            if col in df_final.columns:
                df_final[col] = pd.to_datetime(df_final[col], errors="coerce")

        # Sélection des colonnes du schéma Trino (reindex remplit les manquantes avec NaN)
        _SCHEMA_COLS = [
            "payload_index", "user_id", "constatation_index", "label_group", "subgroup",
            "tranche_label", "start_date", "end_date", "trimester_days", "bill_amount", "aid_amount",
            "fee_adjustment_days", "constatation_id", "billing_type", "trimestre_id",
            "fee_adjustment_amount", "is_simulation", "created_at", "updated_at",
        ]
        df_final = df_final.reindex(columns=_SCHEMA_COLS)

        db.truncate(table)
        rows_inserted = db.bulk_insert(table, df_final)
        result.rows_upserted = rows_inserted
        result.success = True
        result.status = "complete_success"
        logger.info(f"[{name}] OK — {rows_inserted} lignes insérées")

    except Exception as e:
        logger.exception(f"[{name}] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"

    result.duration_seconds = round(time.time() - t0, 2)
    return result


def run_full_reload_all_webresto_tables_job(
    base_url: str,
    secret_key_webresto: str,
    environnement_client: str,
    prefix_table: str,
    ovh_api_key: str,
    ovh_secret_key: str,
    start_date: str = "2020-01-01",
) -> dict[str, JobResult]:
    """
    Recharge toutes les tables Webresto depuis zéro (TRUNCATE + bulk_insert).

    Ordre d'exécution :
    school_year → organization → subgroup_mapping → service → group → tranche → subgroup
    → vague → session → user → bankdetail → passage → passage_partner → history
    → registration → trimester → organization_enrollment → constatation
    """
    fetcher = WebrestoFetcher(base_url=base_url, api_key=secret_key_webresto)
    db = TrinoClient(
        environnement_client=environnement_client,
        ovh_api_key=ovh_api_key,
        ovh_secret_key=ovh_secret_key,
    )
    p = prefix_table
    results: dict[str, JobResult] = {}
    _TABLES = [
        "school_year", "organization", "subgroup_mapping", "service", "group", "tranche",
        "subgroup", "vague", "session", "user", "bankdetail", "passage", "passage_partner",
        "history", "registration", "trimester", "organization_enrollment", "constatation",
        "booking",
    ]
    _total = len(_TABLES)

    print(f"\n{'='*60}")
    print(f"  FULL RELOAD Webresto — {environnement_client}")
    print(f"  {_total} tables à recharger depuis {start_date}")
    print(f"{'='*60}\n")

    def _step(step_name: str, fn: Callable[[], JobResult]) -> None:
        step_num = len(results) + 1
        print(f"[{step_num:2d}/{_total}] {step_name} ...", end=" ", flush=True)
        logger.info(f"[full_reload] Démarrage : {step_name}")
        r = fn()
        results[step_name] = r
        status_icon = "OK" if r.success else "ERREUR"
        errs = f"  ({len(r.errors)} erreur(s))" if r.errors else ""
        print(f"{status_icon}  {r.rows_upserted} lignes  {r.duration_seconds:.1f}s{errs}")
        logger.info(
            f"[full_reload] {step_name} → {r.status} "
            f"({r.rows_upserted} lignes, {r.duration_seconds:.1f}s)"
        )

    # 1. school_year (CSV — 93 et centre)
    _step("school_year", lambda: _load_csv_school_year(db, p, environnement_client))

    # 2. organization (API GET)
    _step("organization", lambda: _run_full_reload_job(
        name="organization", fetcher=fetcher, db=db,
        endpoint="/findAll/organizations", table=f"{p}organization",
        method="GET",
        body={
            "isHideDemo": "true",
            "select": "organizationId, name, rne, city, accessSoftware, type, ips, academy, vague, department",
        },
        preprocess=None, transform=transform_organization,
        environnement_client=environnement_client,
    ))

    # 3. subgroup_mapping (API GET)
    _step("subgroup_mapping", lambda: _run_full_reload_job(
        name="subgroup_mapping", fetcher=fetcher, db=db,
        endpoint="/getSubgroupMapping", table=f"{p}subgroup_mapping",
        method="GET", body=None, preprocess=None,
        transform=transform_subgroup_mapping, environnement_client=environnement_client,
    ))

    # 4. service (CSV — endpoint indisponible)
    _step("service", lambda: _load_csv_service(db, p, environnement_client))

    # 5. group (CSV — endpoint indisponible)
    _step("group", lambda: _load_csv_group(db, p, environnement_client))

    # 6. tranche (CSV — endpoint indisponible)
    _step("tranche", lambda: _load_csv_tranche(db, p, environnement_client))

    # 7. subgroup (API POST)
    _step("subgroup", lambda: _run_full_reload_job(
        name="subgroup", fetcher=fetcher, db=db,
        endpoint="/findAll/subgroups", table=f"{p}subgroup",
        method="POST", body={}, preprocess=preprocess_subgroup,
        transform=transform_subgroup, environnement_client=environnement_client,
    ))

    # 8. vague (API GET)
    _step("vague", lambda: _run_full_reload_job(
        name="vague", fetcher=fetcher, db=db,
        endpoint="/findAll/vagues", table=f"{p}vague",
        method="GET", body=None, preprocess=None,
        transform=transform_vague, environnement_client=environnement_client,
    ))

    # 9. session (API POST)
    _step("session", lambda: _run_full_reload_job(
        name="session", fetcher=fetcher, db=db,
        endpoint="/findAll/sessions", table=f"{p}session",
        method="POST", body={"relations": {"vague": True}},
        preprocess=preprocess_session, transform=transform_session,
        environnement_client=environnement_client,
    ))

    # 10. user (API POST body={} = tous les users)
    _step("user", lambda: _run_full_reload_job(
        name="user", fetcher=fetcher, db=db,
        endpoint="/findAll/AllUsers", table=f"{p}user",
        method="POST", body={}, preprocess=None,
        transform=transform_user, environnement_client=environnement_client,
    ))

    # 11. bankdetail (incrémental full reload par chunks de 4m)
    _step("bankdetail", lambda: _run_full_reload_incremental_job(
        name="bankdetail", fetcher=fetcher, db=db,
        endpoint="/findAll/bankDetails", table=f"{p}bankdetail",
        method="GET",
        body_builder=lambda s, e: {
            "updatedSince": s,
            "updatedBefore": e,
            "selects": "createdAt, updatedAt, deletedAt, bankDetailId, trancheId, choiceBankDetails, userId",
        },
        preprocess=preprocess_bankdetail, transform=transform_bankdetail,
        environnement_client=environnement_client, chunk_months=4, start_date=start_date,
    ))

    # 12. passage (incrémental full reload, format DD/MM/YYYY)
    _step("passage", lambda: _run_full_reload_incremental_job(
        name="passage", fetcher=fetcher, db=db,
        endpoint="/findAll/passages", table=f"{p}passage",
        method="GET",
        body_builder=lambda s, e: {
            "startDate": datetime.strptime(s, "%Y-%m-%d").strftime("%d/%m/%Y"),
            "endDate":   datetime.strptime(e, "%Y-%m-%d").strftime("%d/%m/%Y"),
        },
        preprocess=preprocess_passage, transform=transform_passage,
        environnement_client=environnement_client, chunk_months=4, start_date=start_date,
    ))

    # 13. passage_partner (incrémental full reload par chunks de 4m)
    _step("passage_partner", lambda: _run_full_reload_incremental_job(
        name="passage_partner", fetcher=fetcher, db=db,
        endpoint="/findAll/statistic_passage_partner", table=f"{p}passage_partner",
        method="POST",
        body_builder=lambda s, e: {"period": {"startDate": s, "endDate": e}},
        preprocess=None, transform=transform_passage_partner,
        environnement_client=environnement_client, chunk_months=4, start_date=start_date,
    ))

    # 14. history (incrémental full reload par chunks de 4m)
    _step("history", lambda: _run_full_reload_incremental_job(
        name="history", fetcher=fetcher, db=db,
        endpoint="/findAll/history", table=f"{p}history",
        method="POST",
        body_builder=lambda s, e: {"updatedSince": s, "updatedBefore": e},
        preprocess=None, transform=transform_history,
        environnement_client=environnement_client, chunk_months=4, start_date=start_date,
    ))

    # 15. registration (incrémental full reload par chunks de 2m)
    _step("registration", lambda: _run_full_reload_incremental_job(
        name="registration", fetcher=fetcher, db=db,
        endpoint="/findAll/registrations", table=f"{p}registration",
        method="POST",
        body_builder=lambda s, e: {"updatedSince": s, "updatedBefore": e},
        preprocess=preprocess_registration, transform=transform_registration,
        environnement_client=environnement_client, chunk_months=2, start_date=start_date,
    ))

    # 16. trimester (CSV)
    _step("trimester", lambda: _load_csv_trimester(db, p, environnement_client))

    # 17. organization_enrollment (CSV)
    _step("organization_enrollment", lambda: _load_csv_organization_enrollment(
        db, p, environnement_client
    ))

    # 18. constatation (CSV + preprocessing JSON, 93 uniquement)
    _step("constatation", lambda: _load_csv_constatation(db, p, environnement_client))

    # 19. booking (CSV — centre uniquement)
    _step("booking", lambda: _load_csv_booking(db, p, environnement_client))

    successful = [n for n, r in results.items() if r.success]
    failed = [n for n, r in results.items() if not r.success]
    total_rows = sum(r.rows_upserted for r in results.values())

    print(f"\n{'='*60}")
    print(f"  RÉSULTAT : {len(successful)}/{len(results)} tables OK — {total_rows} lignes total")
    if failed:
        print(f"  EN ÉCHEC  : {', '.join(failed)}")
    print(f"{'='*60}\n")

    logger.info(
        f"[full_reload] Terminé — {len(successful)}/{len(results)} tables rechargées, "
        f"{total_rows} lignes insérées au total"
    )
    if failed:
        logger.warning(f"[full_reload] Tables en échec : {failed}")

    return results


# ── Job établissement_detail (hors Webresto) ──────────────────────────────────

def _ensure_etablissement_detail_table(db: TrinoClient, table: str) -> None:
    """Crée la table si elle n'existe pas encore."""
    db.run_query(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            uai VARCHAR,
            school_year VARCHAR,
            type_etablissement VARCHAR,
            nom_etablissement VARCHAR,
            code_academie INTEGER,
            libelle_academie VARCHAR,
            code_departement INTEGER,
            libelle_departement VARCHAR,
            code_region INTEGER,
            libelle_region VARCHAR,
            libelle_nature VARCHAR,
            vacances_zone VARCHAR,
            ips DOUBLE
        )
    """)


def run_etablissement_detail_job(
    environnement_client: str,
    prefix_table: str,
    ovh_api_key: str,
    ovh_secret_key: str,
    prefix_webresto: str = None,
) -> JobResult:
    """
    Peuple etablissement_detail depuis les tables Trino source + CSV Éducation nationale.
    Un row est créé par (UAI × année scolaire trouvée dans les CSV IPS).

    prefix_table    : préfixe Webgerest (contient login).
    prefix_webresto : préfixe Webresto  (contient organization). Si None, utilise prefix_table.
    """
    from data_process.process.etablissement import (
        build_etablissement_df,
        get_uais_from_trino,
        load_annuaire,
        load_ips,
    )

    result = JobResult()
    t0 = time.time()
    table = "etablissement_detail"

    try:
        db = TrinoClient(environnement_client, ovh_api_key, ovh_secret_key)
        db_default = TrinoClient("default_dataset", ovh_api_key, ovh_secret_key)

        uais = get_uais_from_trino(db, prefix_table, environnement_client, prefix_webresto)
        logger.info(f"[etablissement_detail] {len(uais)} UAIs trouvés")

        if not uais:
            result.errors.append("Aucun UAI trouvé dans les tables source")
            result.status = "failed"
        else:
            annuaire = load_annuaire(uais)
            ips_df = load_ips(uais)

            if ips_df.empty:
                result.errors.append("Aucune donnée IPS trouvée pour ces UAIs")
                result.status = "failed"
            else:
                df = build_etablissement_df(annuaire, ips_df)
                _ensure_etablissement_detail_table(db_default, table)
                rows = db_default.upsert(table, ["uai", "school_year"], df)
                result.rows_upserted = rows
                result.success = True
                result.status = "complete_success"
                logger.info(f"[etablissement_detail] OK — {rows} lignes upsertées")

    except Exception as e:
        logger.exception("[etablissement_detail] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"

    result.duration_seconds = round(time.time() - t0, 2)
    return result


# ── Job ref_type_ips_corrections (hors Webresto) ─────────────────────────────

def _ensure_ref_type_ips_corrections_table(db: TrinoClient, table: str) -> None:
    """Crée la table si elle n'existe pas encore."""
    db.run_query(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            uai    VARCHAR,
            type   VARCHAR,
            ips    DOUBLE,
            source VARCHAR
        )
    """)


def run_ref_type_ips_corrections_job(
    ovh_api_key: str,
    ovh_secret_key: str,
) -> JobResult:
    """
    Peuple ref_type_ips_corrections depuis les CSV de corrections manuelles type/IPS.
    Full reload (TRUNCATE + INSERT) — pas de paramètre d'environnement.
    Utilisé par le forepass webresto pour enrichir la table organization sans accès fichier local.
    """
    from data_process.process.etablissement import load_ref_type_ips_corrections

    result = JobResult()
    t0 = time.time()
    table = "ref_type_ips_corrections"

    try:
        db = TrinoClient("default_dataset", ovh_api_key, ovh_secret_key)
        df = load_ref_type_ips_corrections()
        if df.empty:
            result.errors.append("Aucune donnée de corrections chargée depuis les CSV")
            result.status = "failed"
        else:
            _ensure_ref_type_ips_corrections_table(db, table)
            db.truncate(table)
            rows = db.bulk_insert(table, df)
            result.rows_upserted = rows
            result.success = True
            result.status = "complete_success"
            logger.info(f"[ref_type_ips_corrections] OK — {rows} lignes insérées")

    except Exception as e:
        logger.exception("[ref_type_ips_corrections] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"

    result.duration_seconds = round(time.time() - t0, 2)
    return result


# ── Job vacances_scolaires (hors Webresto) ────────────────────────────────────

def run_vacances_job(
    ovh_api_key: str,
    ovh_secret_key: str,
) -> JobResult:
    """
    Peuple vacance depuis les fichiers ICS des zones A, B et C.
    Stratégie : TRUNCATE puis bulk_insert (données statiques, full reload).
    """
    from data_process.process.vacances import load_vacances

    result = JobResult()
    t0 = time.time()
    table = "vacances"

    try:
        db = TrinoClient("default_dataset", ovh_api_key, ovh_secret_key)

        df = load_vacances()
        if df.empty:
            result.errors.append("Aucune période de vacances parsée depuis les fichiers ICS")
            result.status = "failed"
        else:
            _ensure_vacances_table(db, table)
            db.truncate(table)
            rows = db.bulk_insert(table, df)
            result.rows_upserted = rows
            result.success = True
            result.status = "complete_success"
            logger.info(f"[vacances_scolaires] OK — {rows} lignes insérées")

    except Exception as e:
        logger.exception("[vacances_scolaires] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"

    result.duration_seconds = round(time.time() - t0, 2)
    return result


def _ensure_vacances_table(db: TrinoClient, table: str) -> None:
    """Crée la table si elle n'existe pas encore."""
    db.run_query(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            zone VARCHAR,
            school_year VARCHAR,
            type_vacances VARCHAR,
            date_debut DATE,
            date_fin DATE
        )
    """)


# ── Job jours_feries (hors Webresto) ─────────────────────────────────────────

def run_jours_feries_job(
    ovh_api_key: str,
    ovh_secret_key: str,
) -> JobResult:
    """
    Peuple jours_feries depuis jours_feries_metropole.csv.
    Stratégie : TRUNCATE puis bulk_insert (données statiques, full reload).
    """
    from data_process.process.jours_feries import load_jours_feries

    result = JobResult()
    t0 = time.time()
    table = "jours_feries"

    try:
        db = TrinoClient("default_dataset", ovh_api_key, ovh_secret_key)

        df = load_jours_feries()
        if df.empty:
            result.errors.append("Aucun jour férié chargé depuis le CSV")
            result.status = "failed"
        else:
            _ensure_jours_feries_table(db, table)
            db.truncate(table)
            rows = db.bulk_insert(table, df)
            result.rows_upserted = rows
            result.success = True
            result.status = "complete_success"
            logger.info(f"[jours_feries] OK — {rows} lignes insérées")

    except Exception as e:
        logger.exception("[jours_feries] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"

    result.duration_seconds = round(time.time() - t0, 2)
    return result


# ── Job refresh datagouv (scraping) ───────────────────────────────────────────

def run_refresh_datagouv_job(dataset_key: str) -> JobResult:
    """
    Rafraîchit un fichier CSV statique de data_process/data/ en scrapant la
    page data.gouv.fr correspondante (cf. datagouv_scraper.DATASETS pour la
    liste des jeux de données suivis).

    Ne touche à aucune table Trino : met seulement à jour le fichier source
    local, qui est ensuite relu par les jobs qui en dépendent (ex.
    etablissement_detail, C3).
    """
    from data_process.process.datagouv_scraper import refresh_dataset

    result = JobResult()
    t0 = time.time()

    try:
        dest_path = refresh_dataset(dataset_key)
        with open(dest_path, encoding="utf-8", errors="ignore") as f:
            line_count = sum(1 for _ in f)
        result.rows_upserted = line_count
        result.success = True
        result.status = "complete_success"
        logger.info(f"[refresh_datagouv:{dataset_key}] OK — {line_count} lignes → {dest_path}")
    except Exception as e:
        logger.exception(f"[refresh_datagouv:{dataset_key}] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"

    result.duration_seconds = round(time.time() - t0, 2)
    return result


def _ensure_jours_feries_table(db: TrinoClient, table: str) -> None:
    """Crée la table si elle n'existe pas encore."""
    db.run_query(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            date DATE,
            annee INTEGER,
            zone VARCHAR,
            nom_jour_ferie VARCHAR
        )
    """)


# ── Job cleanup orphans (registrations + passages sans user) ─────────────────

def run_cleanup_orphans_job(
    environnement_client: str,
    prefix_table: str,
    ovh_api_key: str,
    ovh_secret_key: str,
) -> JobResult:
    """
    Supprime les lignes orphelines (id_user absent de la table user) :
      - registrations de la dernière vague (MAX id_vague)
      - passages avec date > 2025-08-01
    """
    result = JobResult()
    t0 = time.time()
    p = prefix_table

    try:
        db = TrinoClient(environnement_client, ovh_api_key, ovh_secret_key)

        rows_reg = db.run_query(f"""
            DELETE FROM {p}registration
            WHERE id_user NOT IN (SELECT id_user FROM {p}user)
              AND id_session IN (
                  SELECT s.id FROM {p}session s
                  WHERE s.id_vague = (SELECT MAX(id_vague) FROM {p}vague)
              )
        """)
        logger.info(f"[cleanup_orphans] registrations supprimées : {rows_reg}")

        rows_pas = db.run_query(f"""
            DELETE FROM {p}passage
            WHERE date > TIMESTAMP '2025-08-01 00:00:00'
              AND id_user NOT IN (SELECT id_user FROM {p}user)
        """)
        logger.info(f"[cleanup_orphans] passages supprimés : {rows_pas}")

        result.rows_upserted = rows_reg + rows_pas
        result.success = True
        result.status = "complete_success"

    except Exception as e:
        logger.exception("[cleanup_orphans] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"

    result.duration_seconds = round(time.time() - t0, 2)
    return result


# ── Job organization_enrollment (CSV temporaire) ─────────────────────────────

def run_organization_enrollment_job(
    environnement_client: str,
    ovh_api_key: str,
    ovh_secret_key: str,
    csv_path: str = "data_process/temp_data/centre/organizationEnrollment.csv",
) -> JobResult:
    """
    Charge le fichier CSV organizationEnrollment dans wr_prod_organization_enrollment.
    Stratégie : TRUNCATE puis bulk_insert (job temporaire, en attente de la route API).
    """
    result = JobResult()
    t0 = time.time()
    table = "wr_prod_organization_enrollment"

    try:
        db = TrinoClient(environnement_client, ovh_api_key, ovh_secret_key)

        df = pd.read_csv(csv_path)
        df = df.rename(columns={
            "enrollmentId":            "enrollment_id",
            "organizationId":          "organization_id",
            "schoolYearId":            "school_year_id",
            "totalEnrollment":         "total_enrollment",
            "socialTarifBeneficiaries":"social_tarif_beneficiaries",
            "internCount":             "intern_count",
        })
        df = df[["enrollment_id", "organization_id", "school_year_id",
                 "total_enrollment", "intern_count", "social_tarif_beneficiaries"]]
        df = df.astype("int64")

        if df.empty:
            result.errors.append("CSV vide")
            result.status = "failed"
        else:
            db.truncate(table)
            rows = db.bulk_insert(table, df)
            result.rows_upserted = rows
            result.success = True
            result.status = "complete_success"
            logger.info(f"[organization_enrollment] OK — {rows} lignes insérées")

    except Exception as e:
        logger.exception("[organization_enrollment] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"

    result.duration_seconds = round(time.time() - t0, 2)
    return result


# ── Job school_year (CSV temporaire) ─────────────────────────────────────────

def run_school_year_job(
    environnement_client: str,
    ovh_api_key: str,
    ovh_secret_key: str,
    csv_path: str = "data_process/temp_data/centre/schoolYear.csv",
) -> JobResult:
    """
    Charge le fichier CSV schoolYear dans wr_prod_school_year.
    Stratégie : TRUNCATE puis bulk_insert (job temporaire, en attente de la route API).
    """
    result = JobResult()
    t0 = time.time()
    table = "wr_prod_school_year"

    try:
        db = TrinoClient(environnement_client, ovh_api_key, ovh_secret_key)

        df = pd.read_csv(csv_path)
        df = df.rename(columns={
            "schoolYearId": "school_year_id",
            "label":        "label",
            "startDate":    "start_date",
            "endDate":      "end_date",
        })
        df = df[["school_year_id", "label", "start_date", "end_date"]]
        df["school_year_id"] = df["school_year_id"].astype("int64")
        df["start_date"] = pd.to_datetime(df["start_date"]).dt.date
        df["end_date"] = pd.to_datetime(df["end_date"]).dt.date

        if df.empty:
            result.errors.append("CSV vide")
            result.status = "failed"
        else:
            db.truncate(table)
            rows = db.bulk_insert(table, df)
            result.rows_upserted = rows
            result.success = True
            result.status = "complete_success"
            logger.info(f"[school_year] OK — {rows} lignes insérées")

    except Exception as e:
        logger.exception("[school_year] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"

    result.duration_seconds = round(time.time() - t0, 2)
    return result


# ── Job création des tables Webresto ─────────────────────────────────────────

def run_create_webresto_tables_job(
    dataset: str,
    prefix: str,
    ovh_api_key: str,
    ovh_secret_key: str,
) -> JobResult:
    """
    Crée toutes les tables Webresto sur un environnement donné (idempotent).

    Se connecte à catalog=db_mg6jk45h_{dataset}, schema={dataset} et exécute
    un CREATE TABLE IF NOT EXISTS pour chacune des 17 tables Webresto avec les
    propriétés Iceberg standard.

    Usage :
        result = run_create_webresto_tables_job(
            dataset="prodcentre",
            prefix="wr_prod_",
            ovh_api_key=os.environ["OVH_API_KEY"],
            ovh_secret_key=os.environ["OVH_SECRET_KEY"],
        )
    """
    from data_process.process.schemas_webresto import WEBRESTO_SCHEMAS

    result = JobResult()
    t0 = time.time()
    tables_ok = 0

    try:
        db = TrinoClient(dataset, ovh_api_key, ovh_secret_key)
        logger.info(f"[create_tables] Connexion à db_mg6jk45h_{dataset}.{dataset}")

        for table_suffix, schema in WEBRESTO_SCHEMAS.items():
            table_name = f"{prefix}{table_suffix}"
            try:
                db.run_query(schema.to_ddl(table_name))
                tables_ok += 1
                logger.info(f"[create_tables] OK — {table_name}")
            except Exception as e:
                logger.error(f"[create_tables] ERREUR sur {table_name} : {e}")
                result.errors.append(f"{table_name}: {e}")

        result.rows_upserted = tables_ok
        if not result.errors:
            result.success = True
            result.status = "complete_success"
            logger.info(f"[create_tables] {tables_ok}/{len(WEBRESTO_SCHEMAS)} tables créées")
        elif tables_ok > 0:
            result.success = True
            result.status = "partial_success"
            logger.warning(
                f"[create_tables] {tables_ok}/{len(WEBRESTO_SCHEMAS)} tables créées, "
                f"{len(result.errors)} erreur(s)"
            )
        else:
            result.status = "failed"

    except Exception as e:
        logger.exception("[create_tables] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"

    result.duration_seconds = round(time.time() - t0, 2)
    return result


# ── Job création des tables Webgerest ────────────────────────────────────────

def run_create_webgerest_tables_job(
    dataset: str,
    prefix: str,
    ovh_api_key: str,
    ovh_secret_key: str,
) -> JobResult:
    """
    Crée toutes les tables Webgerest sur un environnement donné (idempotent).

    Se connecte à catalog=db_mg6jk45h_{dataset}, schema={dataset} et exécute
    un CREATE TABLE IF NOT EXISTS pour chacune des 28 tables Webgerest avec les
    propriétés Iceberg standard.

    Usage :
        result = run_create_webgerest_tables_job(
            dataset="prodcentre",
            prefix="wg_",
            ovh_api_key=os.environ["OVH_API_KEY"],
            ovh_secret_key=os.environ["OVH_SECRET_KEY"],
        )
    """
    from data_process.process.schemas_webgerest import WEBGEREST_SCHEMAS

    result = JobResult()
    t0 = time.time()
    tables_ok = 0

    try:
        db = TrinoClient(dataset, ovh_api_key, ovh_secret_key)
        logger.info(f"[create_wg_tables] Connexion à db_mg6jk45h_{dataset}.{dataset}")

        for table_suffix, schema in WEBGEREST_SCHEMAS.items():
            table_name = f"{prefix}{table_suffix}"
            try:
                db.run_query(schema.to_ddl(table_name))
                tables_ok += 1
                logger.info(f"[create_wg_tables] OK — {table_name}")
            except Exception as e:
                logger.error(f"[create_wg_tables] ERREUR sur {table_name} : {e}")
                result.errors.append(f"{table_name}: {e}")

        result.rows_upserted = tables_ok
        if not result.errors:
            result.success = True
            result.status = "complete_success"
            logger.info(f"[create_wg_tables] {tables_ok}/{len(WEBGEREST_SCHEMAS)} tables créées")
        elif tables_ok > 0:
            result.success = True
            result.status = "partial_success"
            logger.warning(
                f"[create_wg_tables] {tables_ok}/{len(WEBGEREST_SCHEMAS)} tables créées, "
                f"{len(result.errors)} erreur(s)"
            )
        else:
            result.status = "failed"

    except Exception as e:
        logger.exception("[create_wg_tables] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"

    result.duration_seconds = round(time.time() - t0, 2)
    return result


def _wg_safe_id(s: str) -> str:
    """Sanitize un identifiant webgerest en composant SQL valide (minuscules, sans tirets)."""
    import re
    import unicodedata
    s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).lower().strip("_")
    return s


def run_create_webgerest_v2_tables_job(
    dataset: str,
    server_prefix: str,
    login_groups: list[str],
    base_url: str,
    client_key: str,
    client_secret: str,
    ovh_api_key: str,
    ovh_secret_key: str,
) -> JobResult:
    """
    Crée l'intégralité des tables Webgerest V2 et initialise login + descfic.

    Architecture V2 : une table physique par login_group ou login_site, sans
    partitionnement Iceberg. Contrairement à V1 (une grande table partitionnée),
    chaque table ne contient que les données d'un seul groupe ou site.

    Naming :
        {server_prefix}login                  → table login unique par serveur
        {server_prefix}{safe_group}_descfic   → une par login_group
        {server_prefix}{safe_group}_{table}   → données statut=1 (par groupe)
        {server_prefix}{safe_site}_{table}    → données statut=2 (par site)

    Phases :
        1. Crée + charge {server_prefix}login (fetch par groupe, upsert de tous)
        2. Crée + charge {server_prefix}{group}_descfic pour chaque login_group
        3. Lit la table login pour construire le mapping group → sites
        4. Lit chaque descfic et crée les tables de données correspondantes

    Args:
        dataset:        environnement Trino (ex: "prodcentre")
        server_prefix:  préfixe commun (ex: "centre_") → "centre_login", "centre_cd28_article"
        login_groups:   liste des groupes à traiter (ex: ["CD28", "CD18", "REG-CENT"])
        base_url:       URL de base de l'API Webgerest
        client_key:     clé client API Webgerest
        client_secret:  secret client API Webgerest
        ovh_api_key:    clé OVH pour Trino
        ovh_secret_key: secret OVH pour Trino
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from data_process.fetch.fetch_webgerest import WebgestFetcher
    from data_process.process.schemas_webgerest import WEBGEREST_SCHEMAS
    from data_process.process.schemas_webgerest_V2 import WEBGEREST_V2_SCHEMAS
    from data_process.process.transform_webgerest import transform_generic

    result = JobResult()
    t0 = time.time()
    fetcher = WebgestFetcher(base_url, client_key, client_secret)
    db = TrinoClient(dataset, ovh_api_key, ovh_secret_key)

    login_table = f"{server_prefix}login"

    # ── Phase 1 : créer + charger la table login (un fetch par groupe) ───────────
    logger.info(f"[wg_v2_init] Phase 1 — login ({login_table}), {len(login_groups)} groupe(s)")
    try:
        db.run_query(WEBGEREST_V2_SCHEMAS["login"].to_ddl(login_table))
    except Exception as e:
        logger.exception("[wg_v2_init] Erreur DDL login")
        result.errors.append(f"login_ddl: {e}")
        result.status = "failed"
        result.duration_seconds = round(time.time() - t0, 2)
        return result

    # Fetch en parallèle pour chaque groupe
    login_fetched: dict[str, pd.DataFrame | None] = {}
    with ThreadPoolExecutor(max_workers=len(login_groups) or 1) as executor:
        futures = {
            executor.submit(fetcher.fetch_table, "login", grp, None): grp
            for grp in login_groups
        }
        for future in as_completed(futures):
            grp = futures[future]
            try:
                login_fetched[grp] = future.result()
            except Exception as e:
                logger.error(f"[wg_v2_init] login fetch erreur pour {grp!r}: {e}")
                result.errors.append(f"login_fetch {grp}: {e}")

    total_login_rows = 0
    for grp in login_groups:
        df_raw = login_fetched.get(grp)
        if df_raw is None or df_raw.empty:
            logger.warning(f"[wg_v2_init] login {grp!r}: aucune donnée")
            continue
        try:
            df_login_grp = transform_generic(df_raw, WEBGEREST_SCHEMAS["login"], login_identifier=grp)
            rows = db.upsert(login_table, WEBGEREST_V2_SCHEMAS["login"].primary_keys, df_login_grp)
            result.rows_upserted += rows
            total_login_rows += rows
            logger.info(f"[wg_v2_init] login {grp!r}: {rows} lignes upsertées")
        except Exception as e:
            logger.error(f"[wg_v2_init] login écriture erreur pour {grp!r}: {e}")
            result.errors.append(f"login_write {grp}: {e}")

    if total_login_rows == 0 and not login_fetched:
        result.errors.append("login: aucune donnée pour aucun groupe")
        result.status = "failed"
        result.duration_seconds = round(time.time() - t0, 2)
        return result
    logger.info(f"[wg_v2_init] login total : {total_login_rows} lignes upsertées")

    # ── Phase 2 : créer + charger les tables descfic ──────────────────────────
    logger.info(f"[wg_v2_init] Phase 2 — descfic ({len(login_groups)} groupe(s))")

    # Fetch en parallèle
    descfic_fetched: dict[str, pd.DataFrame | None] = {}
    with ThreadPoolExecutor(max_workers=len(login_groups) or 1) as executor:
        futures = {
            executor.submit(fetcher.fetch_table, "descfic", grp, None): grp
            for grp in login_groups
        }
        for future in as_completed(futures):
            grp = futures[future]
            try:
                descfic_fetched[grp] = future.result()
            except Exception as e:
                logger.error(f"[wg_v2_init] descfic fetch erreur pour {grp!r}: {e}")
                result.errors.append(f"descfic_fetch {grp}: {e}")

    # Création DDL + chargement séquentiel
    descfic_schema_v2 = WEBGEREST_V2_SCHEMAS["descfic"]
    descfic_schema_v1 = WEBGEREST_SCHEMAS["descfic"]  # pour transform (injecte login_group puis le drop)
    for grp in login_groups:
        descfic_table = f"{server_prefix}{_wg_safe_id(grp)}_descfic"
        try:
            db.run_query(descfic_schema_v2.to_ddl(descfic_table))
        except Exception as e:
            logger.error(f"[wg_v2_init] DDL erreur pour {descfic_table}: {e}")
            result.errors.append(f"descfic_ddl {grp}: {e}")
            continue

        df_raw = descfic_fetched.get(grp)
        if df_raw is None or df_raw.empty:
            logger.info(f"[wg_v2_init] descfic {grp!r}: aucune donnée")
            continue
        try:
            # transform_generic injecte login_group (site_column du schéma V1) puis la sélection
            # finale du schéma V2 (sans login_group) le supprime automatiquement.
            df = transform_generic(df_raw, descfic_schema_v2, login_identifier=grp)
            if df.empty:
                continue
            db.run_query(f"DELETE FROM {descfic_table} WHERE nomfic IS NOT NULL")
            rows = db.bulk_insert(descfic_table, df)
            result.rows_upserted += rows
            logger.info(f"[wg_v2_init] descfic {grp!r}: {rows} lignes insérées")
        except Exception as e:
            logger.error(f"[wg_v2_init] descfic écriture erreur pour {grp!r}: {e}")
            result.errors.append(f"descfic_write {grp}: {e}")

    # ── Phase 3 : construire le mapping login_group → [login_sites] ───────────
    logger.info("[wg_v2_init] Phase 3 — lecture login_map depuis Trino")
    try:
        logins_df = db.query_as_dataframe(
            f"SELECT login, logingroupe FROM {login_table} "
            f"WHERE (nometabs IS NULL OR UPPER(nometabs) NOT LIKE '%DEMO]%')"
        )
        login_map: dict[str, list[str]] = {}
        for _, row in logins_df.iterrows():
            grp = row["logingroupe"]
            if grp not in login_map:
                login_map[grp] = []
            login_map[grp].append(row["login"])
    except Exception as e:
        logger.exception("[wg_v2_init] Impossible de lire login_map")
        result.errors.append(f"login_map: {e}")
        result.status = "partial_success" if result.rows_upserted > 0 else "failed"
        result.duration_seconds = round(time.time() - t0, 2)
        return result

    # ── Phase 4 : créer les tables de données selon statut descfic ────────────
    logger.info("[wg_v2_init] Phase 4 — création des tables de données")

    # Matching nomfic descfic → clé schema : utilise api_table_name si défini
    # (ex: "detail_article" a api_table_name="detailarticle" → nomfic="DETAILARTICLE")
    known_tables = {
        (schema.api_table_name or k).upper(): k
        for k, schema in WEBGEREST_V2_SCHEMAS.items()
        if k not in ("login", "descfic")
    }
    tables_created = 0

    for grp in login_groups:
        descfic_table = f"{server_prefix}{_wg_safe_id(grp)}_descfic"
        try:
            descfic_df = db.query_as_dataframe(f"SELECT nomfic, statut FROM {descfic_table}")
            descfic_map: dict[str, int] = {
                row["nomfic"].upper(): int(row["statut"])
                for _, row in descfic_df.iterrows()
                if row["nomfic"]
            }
        except Exception as e:
            logger.error(f"[wg_v2_init] Lecture {descfic_table} impossible: {e}")
            result.errors.append(f"descfic_read {grp}: {e}")
            continue

        sites = login_map.get(grp, [])

        for nomfic_upper, schema_key in known_tables.items():
            statut = descfic_map.get(nomfic_upper)
            if statut is None:
                continue

            schema = WEBGEREST_V2_SCHEMAS[schema_key]

            if statut == 1:
                table_name = f"{server_prefix}{_wg_safe_id(grp)}_{schema_key}"
                try:
                    db.run_query(schema.to_ddl(table_name))
                    tables_created += 1
                    logger.info(f"[wg_v2_init] OK — {table_name}")
                    time.sleep(0.3)
                except Exception as e:
                    logger.error(f"[wg_v2_init] DDL erreur {table_name}: {e}")
                    result.errors.append(f"ddl {table_name}: {e}")

            elif statut == 2:
                for site in sites:
                    table_name = f"{server_prefix}{_wg_safe_id(site)}_{schema_key}"
                    try:
                        db.run_query(schema.to_ddl(table_name))
                        tables_created += 1
                        logger.info(f"[wg_v2_init] OK — {table_name}")
                        time.sleep(0.3)
                    except Exception as e:
                        logger.error(f"[wg_v2_init] DDL erreur {table_name}: {e}")
                        result.errors.append(f"ddl {table_name}: {e}")

    logger.info(f"[wg_v2_init] Phase 4 terminée — {tables_created} table(s) créées")

    result.success = not result.errors or result.rows_upserted > 0
    result.status = "complete_success" if not result.errors else "partial_success"
    if result.rows_upserted == 0 and result.errors:
        result.status = "failed"
        result.success = False
    result.duration_seconds = round(time.time() - t0, 2)
    logger.info(f"[wg_v2_init] {result.summary()}")
    return result


_WG_V2_DEFAULT_START_DATE = "2016-01-01"
_WG_V2_WRITE_MAX_RETRIES = 3
_WG_V2_WRITE_RETRY_DELAYS = [30, 60, 120]


def _wg_v2_is_retriable(e: Exception) -> bool:
    s = str(e)
    return any(kw in s for kw in (
        "CommitFailedException", "branch main has changed",
        "ICEBERG_CATALOG_ERROR", "Failed to load view",
        "RESTError 503", "Response ended prematurely", "no healthy upstream",
        "Failed to create transaction",
    ))


def run_load_webgerest_v2_all_tables_job(
    dataset: str,
    server_prefix: str,
    login_groups: list[str],
    base_url: str,
    client_key: str,
    client_secret: str,
    ovh_api_key: str,
    ovh_secret_key: str,
    tables: list[str] | None = None,
    max_workers: int = 5,
) -> JobResult:
    """
    Charge toutes les tables de données Webgerest V2 (chargement daté/incrémental).

    Pour chaque table dans WEBGEREST_V2_SCHEMAS (hors login et descfic) :
      - statut descfic = 1 → 1 fetch par login_group → {prefix}{safe_group}_{table}
      - statut descfic = 2 → 1 fetch par login_site  → {prefix}{safe_site}_{table}

    Stratégie d'écriture par table cible :
      - Tables avec column_updates : from_date = MAX(column_updates) dans la table cible
        → DELETE WHERE column_updates >= min(données_fetchées) + bulk_insert
      - Tables sans column_updates : full replace → DELETE FROM + bulk_insert

    Le login_map et les descfic sont lus depuis Trino (créés par init_webgerest_v2_tables.py).

    Args:
        dataset:       environnement Trino (ex: "prodcentre")
        server_prefix: préfixe serveur (ex: "centre_")
        login_groups:  groupes à traiter (ex: ["CD28", "CD18"])
        tables:        liste de tables à charger (None = toutes sauf login/descfic)
        max_workers:   parallélisme des appels API
    """
    import random
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from data_process.fetch.fetch_webgerest import WebgestFetcher
    from data_process.process.schemas_webgerest_V2 import WEBGEREST_V2_SCHEMAS
    from data_process.process.transform_webgerest import transform_generic

    result = JobResult()
    t0 = time.time()
    fetcher = WebgestFetcher(base_url, client_key, client_secret)
    db = TrinoClient(dataset, ovh_api_key, ovh_secret_key)
    login_table = f"{server_prefix}login"

    # ── Lire login_map depuis Trino ───────────────────────────────────────────
    try:
        logins_df = db.query_as_dataframe(
            f"SELECT login, logingroupe FROM {login_table} "
            f"WHERE (nometabs IS NULL OR UPPER(nometabs) NOT LIKE '%DEMO]%')"
        )
        login_map: dict[str, list[str]] = {}
        for _, row in logins_df.iterrows():
            login_map.setdefault(row["logingroupe"], []).append(row["login"])
    except Exception as e:
        logger.exception("[wg_v2_load] Impossible de lire login_map")
        result.errors.append(f"login_map: {e}")
        result.status = "failed"
        result.duration_seconds = round(time.time() - t0, 2)
        return result

    # ── Lire descfic_map depuis Trino ─────────────────────────────────────────
    descfic_map: dict[str, dict[str, int]] = {}  # login_group → {NOMFIC_UPPER: statut}
    for grp in login_groups:
        descfic_table = f"{server_prefix}{_wg_safe_id(grp)}_descfic"
        try:
            df = db.query_as_dataframe(f"SELECT nomfic, statut FROM {descfic_table}")
            descfic_map[grp] = {
                str(row["nomfic"]).upper(): int(row["statut"])
                for _, row in df.iterrows() if row["nomfic"]
            }
        except Exception as e:
            logger.error(f"[wg_v2_load] Lecture {descfic_table} impossible: {e}")
            result.errors.append(f"descfic_read {grp}: {e}")

    # ── Sélection des tables ──────────────────────────────────────────────────
    data_schemas = {
        k: v for k, v in WEBGEREST_V2_SCHEMAS.items()
        if k not in ("login", "descfic") and (tables is None or k in tables)
    }
    logger.info(f"[wg_v2_load] {len(data_schemas)} table(s) × {len(login_groups)} groupe(s)")

    # ── Chargement table par table ────────────────────────────────────────────
    for table_name, schema in data_schemas.items():
        api_route = schema.api_table_name or table_name
        logger.info(f"[wg_v2_load] === {table_name} ===")

        # Construire les tâches (login_identifier, table_cible)
        fetch_tasks: list[tuple[str, str]] = []
        for grp in login_groups:
            statut = descfic_map.get(grp, {}).get(table_name.upper())
            if statut is None:
                logger.warning(f"[wg_v2_load] {table_name}: pas d'entrée descfic pour {grp!r}")
                continue
            if statut == 1:
                fetch_tasks.append((grp, f"{server_prefix}{_wg_safe_id(grp)}_{table_name}"))
            elif statut == 2:
                for site in login_map.get(grp, []):
                    fetch_tasks.append((site, f"{server_prefix}{_wg_safe_id(site)}_{table_name}"))

        if not fetch_tasks:
            logger.info(f"[wg_v2_load] {table_name}: aucune tâche, ignoré")
            continue

        logger.info(f"[wg_v2_load] {table_name}: {len(fetch_tasks)} appel(s) API")

        # ── Fetch en parallèle (full : pas de from_date) ──────────────────────
        fetched: dict[str, pd.DataFrame | None] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fetcher.fetch_table, api_route, login_id, None): login_id
                for login_id, _ in fetch_tasks
            }
            for future in as_completed(futures):
                login_id = futures[future]
                try:
                    fetched[login_id] = future.result()
                except Exception as e:
                    logger.error(f"[wg_v2_load] {table_name} fetch {login_id!r}: {e}")
                    result.errors.append(f"fetch {table_name}/{login_id}: {e}")

        # ── Écriture séquentielle avec retry Iceberg ──────────────────────────
        for login_id, target_table in fetch_tasks:
            df_raw = fetched.get(login_id)
            if df_raw is None or df_raw.empty:
                logger.info(f"[wg_v2_load] {target_table}: aucune donnée")
                continue
            try:
                df = transform_generic(df_raw, schema, login_identifier=login_id)
            except Exception as e:
                logger.error(f"[wg_v2_load] {target_table} transform: {e}")
                result.errors.append(f"transform {target_table}: {e}")
                continue
            if df.empty:
                continue

            for attempt in range(_WG_V2_WRITE_MAX_RETRIES + 1):
                try:
                    db.run_query(f"DELETE FROM {target_table}")
                    rows = db.bulk_insert(target_table, df)
                    result.rows_upserted += rows
                    logger.info(f"[wg_v2_load] {target_table}: {rows} lignes")
                    break
                except Exception as e:
                    if _wg_v2_is_retriable(e) and attempt < _WG_V2_WRITE_MAX_RETRIES:
                        base = _WG_V2_WRITE_RETRY_DELAYS[attempt]
                        delay = base + random.randint(0, base // 2)
                        logger.warning(
                            f"[wg_v2_load] {target_table} erreur transitoire "
                            f"(tentative {attempt+1}/{_WG_V2_WRITE_MAX_RETRIES}), "
                            f"retry dans {delay}s: {type(e).__name__}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"[wg_v2_load] {target_table} écriture: {e}")
                        result.errors.append(f"write {target_table}: {e}")
                        break

    result.success = not result.errors or result.rows_upserted > 0
    result.status = "complete_success" if not result.errors else "partial_success"
    if result.rows_upserted == 0 and result.errors:
        result.status = "failed"
        result.success = False
    result.duration_seconds = round(time.time() - t0, 2)
    logger.info(f"[wg_v2_load] {result.summary()}")
    return result


# ── Lecture des prédictions ──────────────────────────────────────────

# Ordre de priorité des modèles pour le fallback
_MODEL_PRIORITY = ["Ensemble", "XGBoost21", "XGBoost35", "Prophet21", "Prophet35", "ARIMA", "MovingAverage"]


def get_predictions_for_site(
    uai: str,
    min_date: str,
    max_date: str,
    ovh_api_key: str,
    ovh_secret_key: str,
    table_name: str = "passage_predict",
    include_all_models: bool = False,
) -> list[dict]:
    """
    Retourne les prédictions pour un site (UAI) sur une période donnée.

    Pour chaque (target_date, service), sélectionne la prédiction du modèle
    Ensemble avec le plus petit horizon. En cas d'absence d'Ensemble, fallback
    sur XGBoost → Prophet → ARIMA → MovingAverage.

    Args:
        include_all_models: si True, ajoute un champ `models` avec la meilleure
            prédiction (horizon le plus court) de chaque modèle disponible.

    Returns:
        Liste de { date, service, prediction[, models] }
    """
    _PASSAGE_PREFIX_TO_ENV = {
        "wg_test_": "prodcentre",
        "wg_93_": "prod93",
        "wg_rhone_": "prodrhone",
        "wg_13_": "prod13",
    }
    _env = next(
        (e for p, e in _PASSAGE_PREFIX_TO_ENV.items() if table_name.startswith(p)),
        "default_dataset",
    )
    db = TrinoClient(_env, ovh_api_key, ovh_secret_key)

    # Construire la clause de priorité des modèles via CASE
    case_parts = " ".join(
        f"WHEN '{m}' THEN {i}" for i, m in enumerate(_MODEL_PRIORITY)
    )
    model_priority_expr = f"CASE model {case_parts} ELSE {len(_MODEL_PRIORITY)} END"

    sql = f"""
        SELECT target_date, service, prediction
        FROM (
            SELECT
                target_date,
                service,
                prediction,
                ROW_NUMBER() OVER (
                    PARTITION BY target_date, service
                    ORDER BY {model_priority_expr}, horizon ASC
                ) AS rn
            FROM {table_name}
            WHERE uai = '{uai}'
              AND target_date >= DATE '{min_date}'
              AND target_date <= DATE '{max_date}'
              AND prediction IS NOT NULL
        )
        WHERE rn = 1
        ORDER BY target_date, service
    """

    df = db.query_as_dataframe(sql)

    if df.empty:
        return []

    if include_all_models:
        sql_all = f"""
            SELECT target_date, service, model, prediction
            FROM (
                SELECT
                    target_date,
                    service,
                    model,
                    prediction,
                    ROW_NUMBER() OVER (
                        PARTITION BY target_date, service, model
                        ORDER BY horizon ASC
                    ) AS rn
                FROM {table_name}
                WHERE uai = '{uai}'
                  AND target_date >= DATE '{min_date}'
                  AND target_date <= DATE '{max_date}'
                  AND prediction IS NOT NULL
            )
            WHERE rn = 1
            ORDER BY target_date, service, model
        """
        df_all = db.query_as_dataframe(sql_all)
        # Pivot : { (target_date, service) -> { model: prediction } }
        models_map: dict = {}
        for _, row in df_all.iterrows():
            key = (str(row["target_date"]), row["service"])
            models_map.setdefault(key, {})[row["model"]] = row["prediction"]
    else:
        models_map = {}

    return [
        {
            "date": str(row["target_date"]),
            "service": row["service"],
            "prediction": row["prediction"],
            **({"models": models_map.get((str(row["target_date"]), row["service"]), {})} if include_all_models else {}),
        }
        for _, row in df.iterrows()
    ]


# ── Stats dashboard webgerest ─────────────────────────────────────────

_WEBGEREST_SERVER_CONFIG: dict[str, dict] = {
    "centre": {
        "environnement_client": "prodcentre",
        "prefix": "wg_test_",
        "login_groups": ["CD28", "CD18", "CD19", "CD41", "REG-CENT"],
    },
}


def _query_stats_dashboard(
    environnement_client: str,
    prefix: str,
    login_group: str,
    annee: str,
    hors_taxe: bool,
    ovh_api_key: str,
    ovh_secret_key: str,
) -> pd.DataFrame:
    col = "montantht" if hors_taxe else "montant"
    db = TrinoClient(environnement_client, ovh_api_key, ovh_secret_key)
    return db.query_as_dataframe(f"""
        SELECT type_produit, local_label, bio_label, egalim_label,
               MAX(nb_site) AS nb_site,
               SUM({col}) AS montant
        FROM {prefix}stats_dashboard
        WHERE login_group = '{login_group}'
          AND annee = '{annee}'
        GROUP BY type_produit, local_label, bio_label, egalim_label
    """)


def _query_stats_dashboard_effect(
    environnement_client: str,
    prefix: str,
    login_group: str,
    annee: str,
    ovh_api_key: str,
    ovh_secret_key: str,
) -> pd.DataFrame:
    db = TrinoClient(environnement_client, ovh_api_key, ovh_secret_key)
    return db.query_as_dataframe(f"""
        SELECT service, repas_servis, repas_par_jour, nb_sites,
               prix_revient, repas_par_jour2
        FROM {prefix}stats_dashboard_effect
        WHERE logingroup = '{login_group}'
          AND annee = '{annee}'
    """)


def _sum_where(df: pd.DataFrame, filters: dict) -> float:
    mask = pd.Series([True] * len(df), index=df.index)
    for col, val in filters.items():
        if isinstance(val, list):
            mask &= df[col].isin(val)
        else:
            mask &= df[col] == val
    return float(df.loc[mask, "montant"].sum())


_EMPTY_SERVICE = {"prix_de_revient": 0.0, "repas_servis": 0, "repas_par_jour1": 0.0, "repas_par_jour2": 0.0}


def _build_service_dict(df: pd.DataFrame, service_name: str) -> dict:
    row = df[df["service"] == service_name]
    if row.empty:
        return dict(_EMPTY_SERVICE)
    r = row.iloc[0]
    return {
        "prix_de_revient": float(r.get("prix_revient") or 0.0),
        "repas_servis": int(r.get("repas_servis") or 0),
        "repas_par_jour1": float(r.get("repas_par_jour") or 0.0),
        "repas_par_jour2": float(r.get("repas_par_jour2") or 0.0),
    }


def get_stats_dashboard(df_dashboard: pd.DataFrame, df_effect: pd.DataFrame) -> dict:
    """Agrège les deux DataFrames en le JSON final du tableau de bord."""
    d = df_dashboard
    nb_sites = int(d["nb_site"].max()) if not d.empty else 0

    return {
        "montant_bio": _sum_where(d, {"bio_label": "bio"}),
        "montant_non_bio": _sum_where(d, {"bio_label": "non bio"}),
        "montant_local": _sum_where(d, {"local_label": "local"}),
        "montant_non_local": _sum_where(d, {"local_label": "non local"}),
        "montant_egalim": _sum_where(d, {"egalim_label": "EGALIM"}),
        "montant_non_egalim": _sum_where(d, {"egalim_label": "non EGALIM"}),
        "montant_viande_egalim": _sum_where(d, {"type_produit": "viande", "egalim_label": "EGALIM"}),
        "montant_viande_non_egalim": _sum_where(d, {"type_produit": "viande", "egalim_label": "non EGALIM"}),
        "montant_poisson_egalim": _sum_where(d, {"type_produit": "poisson", "egalim_label": "EGALIM"}),
        "montant_poisson_non_egalim": _sum_where(d, {"type_produit": "poisson", "egalim_label": "non EGALIM"}),
        "montant_viande_et_poisson_egalim": _sum_where(d, {"type_produit": ["viande", "poisson"], "egalim_label": "EGALIM"}),
        "montant_viande_et_poisson_non_egalim": _sum_where(d, {"type_produit": ["viande", "poisson"], "egalim_label": "non EGALIM"}),
        "nb_sites": nb_sites,
        "service_dejeuner": _build_service_dict(df_effect, "dejeuner"),
        "service_journee": _build_service_dict(df_effect, "journee"),
    }


# ── Jobs Webgerest (fetch API) ────────────────────────────────────────────────

_WEBGEREST_DEFAULT_FROM_DATE = "2016-01-01"


def _get_from_date_for_site(
    db: TrinoClient,
    table: str,
    column_updates: str,
    site_column: str,
    login_val: str,
) -> str:
    """Retourne MAX(column_updates) filtré par login, ou la date par défaut."""
    try:
        df = db.query_as_dataframe(
            f"SELECT MAX({column_updates}) AS last_update "
            f"FROM {table} WHERE {site_column} = '{login_val}'"
        )
        if not df.empty and df.iloc[0, 0] is not None:
            val = df.iloc[0, 0]
            return val if isinstance(val, str) else val.strftime("%Y-%m-%d")
    except Exception as e:
        logger.warning(
            f"Impossible de lire MAX({column_updates}) WHERE {site_column}={login_val!r} "
            f"sur {table}: {e}"
        )
    return _WEBGEREST_DEFAULT_FROM_DATE


def run_webgerest_login_job(
    dataset: str,
    prefix: str,
    login_group: str,
    base_url: str,
    client_key: str,
    client_secret: str,
    ovh_api_key: str,
    ovh_secret_key: str,
) -> JobResult:
    """
    Synchronise la table login Webgerest.

    Un seul appel API avec login_group suffit pour récupérer l'ensemble de la table login.
    Stratégie : upsert (primary_key=login), sans from_date.

    Args:
        dataset:      environnement Trino (ex: "prodcentre")
        prefix:       préfixe des tables (ex: "wg_")
        login_group:  login_group à utiliser pour l'appel API (ex: "REG-CENT")
        base_url:     URL de base de l'API Webgerest
        client_key:   clé client API Webgerest
        client_secret: secret client API Webgerest
        ovh_api_key:  clé OVH pour Trino
        ovh_secret_key: secret OVH pour Trino
    """
    from data_process.fetch.fetch_webgerest import WebgestFetcher
    from data_process.process.schemas_webgerest import WEBGEREST_SCHEMAS
    from data_process.process.transform_webgerest import transform_generic

    result = JobResult()
    t0 = time.time()
    schema = WEBGEREST_SCHEMAS["login"]
    table_full = f"{prefix}login"
    fetcher = WebgestFetcher(base_url, client_key, client_secret)
    db = TrinoClient(dataset, ovh_api_key, ovh_secret_key)

    logger.info(f"[login] Démarrage — login_group={login_group!r}")

    try:
        df_raw = fetcher.fetch_table("login", login_group, None)
        if df_raw is None:
            result.errors.append(f"Aucune donnée retournée pour login_group={login_group!r}")
            result.status = "failed"
        else:
            df = transform_generic(df_raw, schema, login_identifier=login_group)
            if df.empty:
                result.errors.append("DataFrame vide après transformation")
                result.status = "failed"
            else:
                rows = db.upsert(table_full, schema.primary_keys, df)
                result.rows_upserted = rows
                result.success = True
                result.status = "complete_success"
                logger.info(f"[login] {rows} lignes upsertées")
    except Exception as e:
        logger.exception("[login] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"

    result.duration_seconds = round(time.time() - t0, 2)
    logger.info(f"[login] {result.summary()}")
    return result


def run_webgerest_descfic_job(
    dataset: str,
    prefix: str,
    login_groups: list[str],
    base_url: str,
    client_key: str,
    client_secret: str,
    ovh_api_key: str,
    ovh_secret_key: str,
) -> JobResult:
    """
    Synchronise la table descfic Webgerest pour un ou plusieurs login_groups.

    Stratégie : bulk_replace — DELETE WHERE login_group = ? puis bulk_insert.

    Args:
        dataset:      environnement Trino (ex: "prodcentre")
        prefix:       préfixe des tables (ex: "wg_")
        login_groups: groupes à synchroniser
        base_url:     URL de base de l'API Webgerest
        client_key:   clé client API Webgerest
        client_secret: secret client API Webgerest
        ovh_api_key:  clé OVH pour Trino
        ovh_secret_key: secret OVH pour Trino
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from data_process.fetch.fetch_webgerest import WebgestFetcher
    from data_process.process.schemas_webgerest import WEBGEREST_SCHEMAS
    from data_process.process.transform_webgerest import transform_generic

    result = JobResult()
    t0 = time.time()
    schema = WEBGEREST_SCHEMAS["descfic"]
    table_full = f"{prefix}descfic"
    fetcher = WebgestFetcher(base_url, client_key, client_secret)
    db = TrinoClient(dataset, ovh_api_key, ovh_secret_key)

    logger.info(f"[descfic] Démarrage — {len(login_groups)} groupe(s)")

    # Phase 1 : fetch en parallèle
    fetched: dict[str, pd.DataFrame | None] = {}
    with ThreadPoolExecutor(max_workers=len(login_groups) or 1) as executor:
        futures = {
            executor.submit(fetcher.fetch_table, "descfic", grp, None): grp
            for grp in login_groups
        }
        for future in as_completed(futures):
            grp = futures[future]
            try:
                fetched[grp] = future.result()
            except Exception as e:
                logger.error(f"[descfic] Erreur fetch pour {grp!r}: {e}")
                result.errors.append(f"fetch {grp}: {e}")

    # Phase 2 : écriture séquentielle (DELETE + bulk_insert par groupe)
    for grp in login_groups:
        if grp not in fetched:
            continue
        df_raw = fetched[grp]
        if df_raw is None:
            logger.info(f"[descfic] Aucune donnée pour {grp!r}")
            continue
        try:
            df = transform_generic(df_raw, schema, login_identifier=grp)
            if df.empty:
                continue
            grp_esc = grp.replace("'", "''")
            db.run_query(f"DELETE FROM {table_full} WHERE login_group = '{grp_esc}'")
            rows = db.bulk_insert(table_full, df)
            result.rows_upserted += rows
            logger.info(f"[descfic] {grp!r}: {rows} lignes insérées")
        except Exception as e:
            logger.error(f"[descfic] Erreur écriture pour {grp!r}: {e}")
            result.errors.append(f"write {grp}: {e}")

    result.success = not result.errors or result.rows_upserted > 0
    result.status = "complete_success" if not result.errors else "partial_success"
    if result.rows_upserted == 0 and result.errors:
        result.status = "failed"
        result.success = False
    result.duration_seconds = round(time.time() - t0, 2)
    logger.info(f"[descfic] {result.summary()}")
    return result


def run_webgerest_table_job(
    table_name: str,
    dataset: str,
    prefix: str,
    login_groups: list[str],
    base_url: str,
    client_key: str,
    client_secret: str,
    ovh_api_key: str,
    ovh_secret_key: str,
    mode: str = "merge",
    max_workers: int = 5,
) -> JobResult:
    """
    Synchronise une table Webgerest pour un ou plusieurs login_groups.

    Routing DESCFIC :
        statut=1 → un seul fetch par login_group (données centralisées)
        statut=2 → un fetch par login_site du groupe (données par site)

    Modes d'écriture :
        merge        → upsert (nécessite primary_keys dans le schéma)
        bulk_append  → bulk_insert sans suppression préalable
        bulk_replace → DELETE WHERE site_column=? puis bulk_insert

    Fetch parallèle (ThreadPoolExecutor) / écriture séquentielle.

    Args:
        table_name:   clé dans WEBGEREST_SCHEMAS (ex: "article", "mvtart")
        dataset:      environnement Trino (ex: "prodcentre")
        prefix:       préfixe des tables (ex: "wg_")
        login_groups: groupes à synchroniser
        base_url:     URL de base de l'API Webgerest
        client_key:   clé client API Webgerest
        client_secret: secret client API Webgerest
        ovh_api_key:  clé OVH pour Trino
        ovh_secret_key: secret OVH pour Trino
        mode:         "merge" | "bulk_append" | "bulk_replace"
        max_workers:  parallélisme des appels HTTP
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from data_process.fetch.fetch_webgerest import WebgestFetcher
    from data_process.process.schemas_webgerest import WEBGEREST_SCHEMAS
    from data_process.process.transform_webgerest import transform_generic

    result = JobResult()
    t0 = time.time()
    schema = WEBGEREST_SCHEMAS[table_name]
    table_full = f"{prefix}{table_name}"
    api_route = schema.api_table_name or table_name
    fetcher = WebgestFetcher(base_url, client_key, client_secret)
    db = TrinoClient(dataset, ovh_api_key, ovh_secret_key)

    logger.info(f"[{table_name}] Démarrage — mode={mode}, {len(login_groups)} groupe(s)")

    # ── Étape 1 : charger DESCFIC depuis Trino ────────────────────────────────
    try:
        groups_clause = ", ".join(f"'{g}'" for g in login_groups)
        descfic_df = db.query_as_dataframe(
            f"SELECT nomfic, statut, login_group "
            f"FROM {prefix}descfic "
            f"WHERE UPPER(nomfic) = '{table_name.upper()}' AND login_group IN ({groups_clause})"
        )
        descfic_map: dict[str, int] = {
            row["login_group"]: int(row["statut"])
            for _, row in descfic_df.iterrows()
        }
    except Exception as e:
        logger.error(f"[{table_name}] Impossible de charger DESCFIC : {e}")
        result.errors.append(f"descfic_load: {e}")
        result.status = "failed"
        result.duration_seconds = round(time.time() - t0, 2)
        return result

    for grp in login_groups:
        if grp not in descfic_map:
            logger.warning(f"[{table_name}] Aucune entrée DESCFIC pour login_group={grp!r}, ignoré")

    # ── Étape 2 : charger les login_sites pour les groupes statut=2 ───────────
    statut2_groups = [g for g in login_groups if descfic_map.get(g) == 2]
    login_map: dict[str, list[tuple[str, object]]] = {}  # login_group → [(login_site, code_site)]
    if statut2_groups:
        try:
            s2_clause = ", ".join(f"'{g}'" for g in statut2_groups)
            logins_df = db.query_as_dataframe(
                f"SELECT login, code_site, logingroupe "
                f"FROM {prefix}login WHERE logingroupe IN ({s2_clause})"
                f" AND (nometabs IS NULL OR UPPER(nometabs) NOT LIKE '%DEMO]%')"
            )
            for _, row in logins_df.iterrows():
                grp = row["logingroupe"]
                if grp not in login_map:
                    login_map[grp] = []
                login_map[grp].append((row["login"], row["code_site"]))
        except Exception as e:
            logger.error(f"[{table_name}] Impossible de charger les login_sites : {e}")
            result.errors.append(f"login_load: {e}")
            result.status = "failed"
            result.duration_seconds = round(time.time() - t0, 2)
            return result

    # ── Étape 3 : construire la liste des tâches de fetch ─────────────────────
    # Chaque tâche : (login_identifier, code_site, from_date, descfic_statut)
    fetch_tasks: list[tuple[str, object, str, int | None]] = []

    for grp in login_groups:
        statut = descfic_map.get(grp)
        if statut is None:
            continue

        if statut == 1:
            # Un seul fetch pour tout le groupe
            from_date = (
                _get_from_date_for_site(db, table_full, schema.column_updates, schema.site_column, grp)
                if schema.column_updates
                else _WEBGEREST_DEFAULT_FROM_DATE
            )
            fetch_tasks.append((grp, None, from_date, 1))

        elif statut == 2:
            # Un fetch par login_site
            for login_site, code_site in login_map.get(grp, []):
                from_date = (
                    _get_from_date_for_site(
                        db, table_full, schema.column_updates, schema.site_column, login_site
                    )
                    if schema.column_updates
                    else _WEBGEREST_DEFAULT_FROM_DATE
                )
                fetch_tasks.append((login_site, code_site, from_date, 2))
        else:
            logger.warning(
                f"[{table_name}] statut DESCFIC={statut} inconnu pour {grp!r}, ignoré"
            )

    if not fetch_tasks:
        logger.warning(f"[{table_name}] Aucune tâche de fetch, job terminé sans données")
        result.status = "complete_success"
        result.success = True
        result.duration_seconds = round(time.time() - t0, 2)
        return result

    logger.info(f"[{table_name}] {len(fetch_tasks)} appel(s) API planifié(s)")

    # ── Étape 4 : fetch en parallèle ─────────────────────────────────────────
    fetched: dict[str, pd.DataFrame | None] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetcher.fetch_table, api_route, login_id, from_date): login_id
            for login_id, _cs, from_date, _st in fetch_tasks
        }
        for future in as_completed(futures):
            login_id = futures[future]
            try:
                fetched[login_id] = future.result()
            except Exception as e:
                logger.error(f"[{table_name}] Erreur fetch pour {login_id!r}: {e}")
                result.errors.append(f"fetch {login_id}: {e}")

    # ── Étape 5 : écriture séquentielle ───────────────────────────────────────
    for login_id, code_site_val, _, descfic_statut_val in fetch_tasks:
        if login_id not in fetched:
            continue
        df_raw = fetched[login_id]
        if df_raw is None:
            logger.info(f"[{table_name}] Aucune donnée pour {login_id!r}")
            continue
        try:
            df = transform_generic(
                df_raw, schema,
                login_identifier=login_id,
                code_site=code_site_val,
                descfic_statut=descfic_statut_val,
            )
            if df.empty:
                logger.info(f"[{table_name}] DataFrame vide après transform pour {login_id!r}")
                continue

            if mode == "merge":
                rows = db.upsert(table_full, schema.primary_keys, df)
            elif mode == "bulk_append":
                rows = db.bulk_insert(table_full, df)
            elif mode == "bulk_replace":
                login_esc = str(login_id).replace("'", "''")
                db.run_query(
                    f"DELETE FROM {table_full} "
                    f"WHERE {schema.site_column} = '{login_esc}'"
                )
                rows = db.bulk_insert(table_full, df)
            else:
                raise ValueError(f"Mode inconnu : {mode!r}")

            result.rows_upserted += rows
            logger.info(f"[{table_name}] {login_id!r}: {rows} lignes ({mode})")

        except Exception as e:
            logger.error(f"[{table_name}] Erreur écriture pour {login_id!r}: {e}")
            result.errors.append(f"write {login_id}: {e}")

    result.success = not result.errors or result.rows_upserted > 0
    result.status = "complete_success" if not result.errors else "partial_success"
    if result.rows_upserted == 0 and result.errors:
        result.status = "failed"
        result.success = False
    result.duration_seconds = round(time.time() - t0, 2)
    logger.info(f"[{table_name}] {result.summary()}")
    return result


# ── Reporting Webgerest (stats_fcj + stats_liv) ───────────────────────────────

def _run_step_with_retry(job_name: str, table: str, execute_fn, max_retries: int = 5, retry_delay: int = 60):
    """
    Exécute execute_fn() avec retry sur les erreurs transitoires Trino.
    Les erreurs utilisateur (COLUMN_NOT_FOUND, TypeError, etc.) font échouer immédiatement.
    Retourne (rows, error_str | None).
    """
    import trino.exceptions as _trino_exc

    _retryable = (_trino_exc.TrinoExternalError, _trino_exc.Http502Error)

    for attempt in range(1, max_retries + 2):
        try:
            rows = execute_fn()
            return rows, None
        except _retryable as e:
            if attempt <= max_retries:
                logger.warning(
                    f"[{job_name}] {table} : erreur transitoire (tentative {attempt}/{max_retries+1}),"
                    f" retry dans {retry_delay}s — {e}"
                )
                time.sleep(retry_delay)
            else:
                logger.error(f"[{job_name}] {table} : abandon après {max_retries} retries")
                return 0, str(e)
        except Exception as e:
            logger.exception(f"[{job_name}] {table} : erreur non-retryable")
            return 0, str(e)




def _parse_annee(annee: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not annee:
        return None, None
    annee = annee.strip()
    if not annee.isdigit() or len(annee) != 4:
        raise ValueError(f"ANNEE invalide : '{annee}'. Format attendu : '2024'")
    return f"{annee}-01-01", f"{int(annee) + 1}-01-01"


def run_webgerest_reporting_fcj_job(
    dataset: str,
    prefix: str,
    ovh_api_key: str,
    ovh_secret_key: str,
    annee: Optional[str] = None,
    zone_scolaire: str = 'B',
) -> JobResult:
    """
    Calcule et écrit les tables de reporting FCJ Webgerest :
      - {prefix}stats_fcj59
      - {prefix}stats_fcj59_detail
      - {prefix}stats_recap_site
      - {prefix}stat_effect_cred_1
      - {prefix}stats_dashboard_effect

    annee        : filtre optionnel au format "2024" (→ 2024-01-01 / 2025-01-01).
    zone_scolaire : zone pour le calcul des jours ouvrés (défaut: 'B').
    """
    from data_process.process.stats_fcj import (
        compute_stats_fcj59,
        compute_stats_fcj59_detail,
        compute_stats_recap_site,
        compute_stat_effect_cred_1,
        compute_stats_dashboard_effect,
    )

    result = JobResult()
    t0 = time.time()
    db = TrinoClient(dataset, ovh_api_key, ovh_secret_key)
    date_debut, date_fin = _parse_annee(annee)

    _date_where     = (f"datej    >= DATE '{date_debut}' AND datej    < DATE '{date_fin}'" if date_debut else None)
    _datestat_where = (f"datestat >= DATE '{date_debut}' AND datestat < DATE '{date_fin}'" if date_debut else None)
    _annee_where    = (f"annee = '{annee}'" if annee else None)

    steps = [
        ("stats_fcj59",            lambda: compute_stats_fcj59(db, prefix, date_debut, date_fin),           _date_where),
        ("stats_fcj59_detail",     lambda: compute_stats_fcj59_detail(db, prefix, date_debut, date_fin),     _date_where),
        ("stats_recap_site",       lambda: compute_stats_recap_site(db, prefix, date_debut, date_fin),       _datestat_where),
        ("stat_effect_cred_1",     lambda: compute_stat_effect_cred_1(db, prefix, date_debut, date_fin),     _date_where),
        ("stats_dashboard_effect", lambda: compute_stats_dashboard_effect(
            db, prefix, ovh_api_key, ovh_secret_key, date_debut, date_fin, zone_scolaire,
        ), _annee_where),
    ]

    for table_suffix, compute_fn, delete_where in steps:
        table = f"{prefix}{table_suffix}"
        logger.info(f"[reporting_fcj] Calcul {table}...")

        def _execute(t=table, fn=compute_fn, dw=delete_where):
            df = fn()
            if df.empty:
                logger.warning(f"[reporting_fcj] {t} : DataFrame vide, table inchangée")
                return 0
            db.truncate(t, where=dw)
            rows = db.bulk_insert(t, df)
            logger.info(f"[reporting_fcj] {t} : {rows} lignes insérées")
            return rows

        rows, err = _run_step_with_retry("reporting_fcj", table, _execute)
        if err:
            result.errors.append(f"{table}: {err}")
        else:
            result.rows_upserted += rows

    result.success = not result.errors or result.rows_upserted > 0
    result.status = "complete_success" if not result.errors else (
        "partial_success" if result.rows_upserted > 0 else "failed"
    )
    result.duration_seconds = round(time.time() - t0, 2)
    logger.info(f"[reporting_fcj] {result.summary()}")
    return result


def run_webgerest_reporting_liv_job(
    dataset: str,
    prefix: str,
    ovh_api_key: str,
    ovh_secret_key: str,
    annee: Optional[str] = None,
) -> JobResult:
    """
    Calcule et écrit les tables de reporting LIV Webgerest :
      - {prefix}stats_liv59
      - {prefix}stats_liv
      - {prefix}stats_liv_mois
      - {prefix}stats_liv_annee
      - {prefix}stats_liv_egalim
      - {prefix}stats_dashboard

    annee : filtre optionnel au format "2024" (→ 2024-01-01 / 2025-01-01).
    """
    from data_process.process.stats_liv import (
        compute_stats_liv59,
        compute_stats_liv,
        compute_stats_liv_mois,
        compute_stats_liv_annee,
        compute_stats_liv_egalim,
        compute_stats_dashboard,
    )

    result = JobResult()
    t0 = time.time()
    now = datetime.now()
    db = TrinoClient(dataset, ovh_api_key, ovh_secret_key)
    date_debut, date_fin = _parse_annee(annee)

    _annee_where = (f"annee = '{annee}'" if annee else None)

    timestamped_steps = [
        ("stats_liv59",     lambda: compute_stats_liv59(db, prefix, date_debut, date_fin)),
        ("stats_liv",       lambda: compute_stats_liv(db, prefix, date_debut, date_fin)),
        ("stats_liv_mois",  lambda: compute_stats_liv_mois(db, prefix, date_debut, date_fin)),
        ("stats_liv_annee", lambda: compute_stats_liv_annee(db, prefix, date_debut, date_fin)),
    ]
    other_steps = [
        ("stats_liv_egalim", lambda: compute_stats_liv_egalim(db, prefix)),
        ("stats_dashboard",  lambda: compute_stats_dashboard(db, prefix)),
    ]

    for table_suffix, compute_fn in timestamped_steps:
        table = f"{prefix}{table_suffix}"
        logger.info(f"[reporting_liv] Calcul {table}...")

        def _execute_ts(t=table, fn=compute_fn):
            df = fn()
            if df.empty:
                logger.warning(f"[reporting_liv] {t} : DataFrame vide, table inchangée")
                return 0
            df["date_import"] = now
            df["date_modif"]  = now
            db.truncate(t, where=_annee_where)
            rows = db.bulk_insert(t, df)
            logger.info(f"[reporting_liv] {t} : {rows} lignes insérées")
            return rows

        rows, err = _run_step_with_retry("reporting_liv", table, _execute_ts)
        if err:
            result.errors.append(f"{table}: {err}")
        else:
            result.rows_upserted += rows

    for table_suffix, compute_fn in other_steps:
        table = f"{prefix}{table_suffix}"
        logger.info(f"[reporting_liv] Calcul {table}...")

        def _execute_other(t=table, fn=compute_fn):
            df = fn()
            if df.empty:
                logger.warning(f"[reporting_liv] {t} : DataFrame vide, table inchangée")
                return 0
            db.truncate(t, where=_annee_where)
            rows = db.bulk_insert(t, df)
            logger.info(f"[reporting_liv] {t} : {rows} lignes insérées")
            return rows

        rows, err = _run_step_with_retry("reporting_liv", table, _execute_other)
        if err:
            result.errors.append(f"{table}: {err}")
        else:
            result.rows_upserted += rows

    result.success = not result.errors or result.rows_upserted > 0
    result.status = "complete_success" if not result.errors else (
        "partial_success" if result.rows_upserted > 0 else "failed"
    )
    result.duration_seconds = round(time.time() - t0, 2)
    logger.info(f"[reporting_liv] {result.summary()}")
    return result


# ── Recherche floue d'utilisateurs ───────────────────────────────────────────

_USER_CACHE: dict[tuple, tuple[float, pd.DataFrame]] = {}
_USER_CACHE_TTL = 14400  # 4h
_USER_CACHE_LOCK = Lock()


def _normalize_name(text: Optional[str]) -> str:
    """NFD + strip accents + uppercase, cohérent avec to_snake_case de transform_webgerest."""
    if not text:
        return ""
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode()
    return text.upper().strip()


def _get_users_df(
    environnement_client: str,
    prefix_table: str,
    id_organization: float,
    ovh_api_key: str,
    ovh_secret_key: str,
) -> pd.DataFrame:
    key = (environnement_client, prefix_table, int(id_organization))
    with _USER_CACHE_LOCK:
        if key in _USER_CACHE:
            ts, df = _USER_CACHE[key]
            if time.time() - ts < _USER_CACHE_TTL:
                return df

    db = TrinoClient(environnement_client, ovh_api_key, ovh_secret_key)
    df = db.query_as_dataframe(f"""
        SELECT id_user, first_name, last_name, date_birth
        FROM {prefix_table}user
        WHERE id_organization = {id_organization}
          AND first_name IS NOT NULL
          AND last_name IS NOT NULL
    """)

    if not df.empty:
        df["_first_norm"] = df["first_name"].apply(_normalize_name)
        df["_last_norm"]  = df["last_name"].apply(_normalize_name)

    with _USER_CACHE_LOCK:
        _USER_CACHE[key] = (time.time(), df)

    return df


def find_user_candidates(
    id_organization: float,
    last_name: str,
    first_name: str,
    date_birth: Optional[str],
    environnement_client: str,
    prefix_table: str,
    ovh_api_key: str,
    ovh_secret_key: str,
) -> list[dict]:
    """
    Retourne les 10 userId les plus proches d'un triplet (nom, prénom, date_naissance)
    au sein d'une organisation, via scoring Jaro-Winkler.

    Si date_birth est fourni : top 5 avec correspondance de date + top 5 sans.
    Sinon : top 10 tous candidats (date_valid=False).
    """
    from rapidfuzz.distance import JaroWinkler
    from rapidfuzz import process as rfprocess

    df = _get_users_df(environnement_client, prefix_table, id_organization, ovh_api_key, ovh_secret_key)

    if df.empty:
        return []

    df = df.copy()
    q_first = _normalize_name(first_name)
    q_last  = _normalize_name(last_name)

    scores_first = rfprocess.cdist([q_first], df["_first_norm"].tolist(),
                                    scorer=JaroWinkler.similarity, workers=-1)[0]
    scores_last  = rfprocess.cdist([q_last],  df["_last_norm"].tolist(),
                                    scorer=JaroWinkler.similarity, workers=-1)[0]
    df["score"] = (0.4 * scores_first + 0.6 * scores_last).round(4)

    df["date_valid"] = False
    if date_birth:
        df["date_valid"] = df["date_birth"].apply(
            lambda d: d is not None and str(d) == date_birth
        )

    def _to_row(row: pd.Series) -> dict:
        return {
            "userId": int(row["id_user"]),
            "score": row["score"],
            "nom": row["last_name"],
            "prenom": row["first_name"],
            "date_naissance": str(row["date_birth"]) if row["date_birth"] is not None else None,
            "date_valid": bool(row["date_valid"]),
        }

    if date_birth:
        matched   = df[df["date_valid"]].nlargest(5, "score")
        unmatched = df[~df["date_valid"]].nlargest(5, "score")
        top = pd.concat([matched, unmatched])
    else:
        top = df.nlargest(10, "score")

    return [_to_row(row) for _, row in top.iterrows()]

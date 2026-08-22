import asyncio
import io
import logging
import time
import dataclasses
import hashlib
import hmac
import json
import os
import secrets
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import Response, StreamingResponse
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from data_process.jobs import (
    JobName, run_all_jobs, run_etablissement_detail_job, run_job,
    run_jours_feries_job, run_vacances_job, run_ref_type_ips_corrections_job,
    get_predictions_for_site,
    get_stats_dashboard, _query_stats_dashboard, _query_stats_dashboard_effect,
    _WEBGEREST_SERVER_CONFIG,
    run_webgerest_login_job, run_webgerest_descfic_job, run_webgerest_table_job,
    run_cleanup_orphans_job, find_user_candidates,
    run_webgerest_reporting_fcj_job, run_webgerest_reporting_liv_job,
)
from data_process.process.stats_webresto import (
    resolve_school_year_id,
    query_kpis_and_par_tranche, query_general_tarif2, query_general_enrollment,
    query_recours_inscriptions, query_recours_validations,
    query_recours_kpis_par_tranche,
    query_passages_tarif3_cached,
    query_export_tarif1,
    query_available_filters,
    _cache_key, get_cached_response, set_cached_response,
    _df_to_records,
    assemble_general, assemble_recours, assemble_export, assemble_passages,
    EXPORT_COL_LABELS,
)
try:
    from prediction_passages.main import (
        run_arima_rolling,
        run_prophet_rolling,
        run_xgb_rolling,
        run_ma_rolling,
        run_ensemble_rolling,
        run_ensemble_rolling_retry,
        run_global_dep_xgb_rolling,
    )
    from prediction_passages.src.data_prep import DataPreparation as PredictionDataPreparation
    _PREDICTION_AVAILABLE = True
except ImportError as _import_err:
    _PREDICTION_AVAILABLE = False
    _import_err_msg = str(_import_err)

load_dotenv()

logger = logging.getLogger("uvicorn.error")

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

app = FastAPI(
    title="IAnord Data API",
    description="API pour faire le lien les services d'ianord au datalake.",
    version="2.3.21"
)

# ── Monitorage applicatif (Prometheus) ───────────────────────────────
# Expose /metrics et instrumente automatiquement http_requests_total /
# http_request_duration_seconds_bucket (label "handler" = template de route).
# should_group_status_codes=False : conserve le code HTTP exact (404, 500...)
# plutôt que la classe (4xx, 5xx), pour matcher les requêtes des dashboards
# Grafana existants (status=~"4..", status=~"5..").
Instrumentator(should_group_status_codes=False).instrument(app).expose(app)

PREDICTION_JOBS_RUNNING = Gauge(
    "databridge_prediction_jobs_running",
    "Nombre de jobs de prédiction actuellement en cours d'exécution",
)
PREDICTION_JOBS_COMPLETED = Counter(
    "databridge_prediction_jobs_completed_total",
    "Nombre de jobs de prédiction terminés avec succès",
    ["model"],
)
PREDICTION_JOBS_FAILED = Counter(
    "databridge_prediction_jobs_failed_total",
    "Nombre de jobs de prédiction terminés en échec",
    ["model"],
)
# Même raison que pour SYNC_JOB_RESULTS plus bas : pré-enregistrer les séries à 0
# pour que la toute première transition soit visible par increase()/rate().
for _model_name in ("arima", "prophet", "xgboost", "moving_average", "ensemble", "all", "global_dep_xgb"):
    PREDICTION_JOBS_COMPLETED.labels(model=_model_name)
    PREDICTION_JOBS_FAILED.labels(model=_model_name)

# Jobs de synchronisation Webresto -> Trino (/sync, /sync/{job_name}).
# Un job en échec partiel (ex: la gateway amont change de contrat -- cf. incident
# bankdetail) répond tout de même HTTP 200 (JobResult.status = "partial_success"),
# donc invisible dans http_requests_total : cette métrique couvre spécifiquement
# ce cas, invisible au niveau HTTP.
# Le label est nommé "sync_job" (pas "job") pour éviter la collision avec le
# label "job" que Prometheus ajoute automatiquement à chaque cible scrapée
# (job_name="databridge-api" dans prometheus.yml) -- honor_labels valant false
# par défaut, un label "job" exposé par l'appli aurait été silencieusement
# renommé en "exported_job" par Prometheus.
SYNC_JOB_RESULTS = Counter(
    "databridge_sync_job_results_total",
    "Résultats des jobs de synchronisation Webresto par job et par statut",
    ["sync_job", "status"],
)
# Le client Prometheus ne crée une série exposée qu'au premier .labels(...).inc() --
# sans ce pré-enregistrement, la toute première transition (série inexistante -> 1)
# n'est jamais observée par un scrape, et increase()/rate() ne peut donc jamais la
# détecter (aucune valeur "0" de référence n'a été échantillonnée avant coup).
for _job_name in JobName:
    for _status in ("complete_success", "partial_success", "failed"):
        SYNC_JOB_RESULTS.labels(sync_job=_job_name.value, status=_status)


def _record_sync_result(job_name: str, result) -> None:
    SYNC_JOB_RESULTS.labels(sync_job=job_name, status=result.status).inc()

# ── Rôles & credentials ──────────────────────────────────────────────


class Role(str, Enum):
    admin = "admin"
    webgerest_readonly = "webgerest_readonly"
    webresto_readonly = "webresto_readonly"


# Registry : rôle → (client_key, client_secret)
# Le rôle admin accepte aussi les anciennes variables CLIENT_KEY / CLIENT_SECRET
ROLE_CREDENTIALS: dict[Role, tuple[str | None, str | None]] = {
    Role.admin: (
        os.getenv("CLIENT_KEY_ADMIN") or os.getenv("CLIENT_KEY"),
        os.getenv("CLIENT_SECRET_ADMIN") or os.getenv("CLIENT_SECRET"),
    ),
    Role.webgerest_readonly: (
        os.getenv("CLIENT_KEY_WEBGEREST_RO"),
        os.getenv("CLIENT_SECRET_WEBGEREST_RO"),
    ),
    Role.webresto_readonly: (
        os.getenv("CLIENT_KEY_WEBRESTO_RO"),
        os.getenv("CLIENT_SECRET_WEBRESTO_RO"),
    ),
}

# Stockage simple du token en mémoire
active_tokens = {}

# Suivi des jobs de prédiction en arrière-plan
prediction_jobs: dict[str, dict] = {}



class PredictionModel(str, Enum):
    arima = "arima"
    prophet = "prophet"
    xgboost = "xgboost"
    moving_average = "moving_average"
    ensemble = "ensemble"


class PredictionRequest(BaseModel):
    horizon_days: Optional[int] = None
    step_days: Optional[int] = None
    min_train_weeks: Optional[int] = None
    variant: Optional[str] = None
    horizon_weeks: Optional[int] = None
    use_retry: bool = False
    wait: Optional[int] = None
    max_retries: Optional[int] = None
    prefix: Optional[str] = None
    run_ensemble: bool = True
    force_start_date: Optional[str] = None


def _run_prediction_job(job_id: str, func, kwargs: dict):
    """Exécute une fonction rolling et met à jour le statut du job."""
    model_name = prediction_jobs[job_id]["model_name"]
    PREDICTION_JOBS_RUNNING.inc()
    try:
        func(**kwargs)
        prediction_jobs[job_id]["status"] = "completed"
        PREDICTION_JOBS_COMPLETED.labels(model=model_name).inc()
    except Exception as e:
        import traceback
        prediction_jobs[job_id]["status"] = "failed"
        prediction_jobs[job_id]["error"] = f"{type(e).__name__}: {e}"
        print(f"[job {job_id}] ERREUR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        PREDICTION_JOBS_FAILED.labels(model=model_name).inc()
    finally:
        prediction_jobs[job_id]["finished_at"] = datetime.utcnow()
        PREDICTION_JOBS_RUNNING.dec()


def _check_prediction_available():
    """Lève une 503 si les modules de prédiction ne sont pas installés."""
    if not _PREDICTION_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Modules de prédiction non disponibles : {_import_err_msg}",
        )


def _build_kwargs(model: PredictionModel, body: PredictionRequest, env: Optional[str] = None) -> tuple:
    """Retourne (function, kwargs) pour un modèle donné."""
    if model == PredictionModel.arima:
        func = run_arima_rolling
        params = {"horizon_days": body.horizon_days, "step_days": body.step_days,
                  "min_train_weeks": body.min_train_weeks, "env": env, "prefix": body.prefix}
    elif model == PredictionModel.prophet:
        func = run_prophet_rolling
        params = {"variant": body.variant, "horizon_days": body.horizon_days,
                  "step_days": body.step_days, "min_train_weeks": body.min_train_weeks,
                  "env": env, "prefix": body.prefix}
    elif model == PredictionModel.xgboost:
        func = run_xgb_rolling
        params = {"variant": body.variant, "horizon_days": body.horizon_days,
                  "step_days": body.step_days, "min_train_weeks": body.min_train_weeks,
                  "env": env, "prefix": body.prefix}
    elif model == PredictionModel.moving_average:
        func = run_ma_rolling
        params = {"horizon_weeks": body.horizon_weeks, "step_days": body.step_days,
                  "min_train_weeks": body.min_train_weeks, "env": env, "prefix": body.prefix}
    elif model == PredictionModel.ensemble:
        if body.use_retry:
            func = run_ensemble_rolling_retry
            params = {"wait": body.wait, "max_retries": body.max_retries}
        else:
            func = run_ensemble_rolling
            params = {"env": env, "prefix": body.prefix}
    else:
        raise ValueError(f"Modèle inconnu : {model}")
    # Retirer les paramètres None pour utiliser les défauts des fonctions
    kwargs = {k: v for k, v in params.items() if v is not None}
    return func, kwargs


def generate_token() -> str:
    """Génère un token sécurisé"""
    return secrets.token_urlsafe(32)


def create_signature(client_key: str, client_secret: str, timestamp: str, nonce: str) -> str:
    """Crée une signature HMAC-SHA256 pour OAuth1"""
    message = f"{client_key}{timestamp}{nonce}"
    signature = hmac.new(
        client_secret.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    return signature


def _verify_token(x_api_token: str = Header(..., alias="X-Api-Token")) -> dict:
    """Vérifie la validité du token via le header X-Api-Token."""
    token = x_api_token

    if token not in active_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide ou expiré",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = active_tokens[token]
    if datetime.utcnow() > token_data["expires_at"]:
        del active_tokens[token]
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expiré",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token_data


def require_role(*allowed_roles: Role):
    """Factory de dépendance FastAPI : vérifie le token ET le rôle."""
    def dependency(token_data: dict = Depends(_verify_token)) -> dict:
        if token_data["role"] not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Accès refusé pour le rôle '{token_data['role']}'",
            )
        return token_data
    return dependency


def strict_query_params(*allowed: str):
    """Factory de dépendance : rejette tout query param absent de `allowed`."""
    allowed_set = frozenset(allowed)
    async def _check(request: Request):
        unknown = set(request.query_params.keys()) - allowed_set
        if unknown:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Paramètre(s) de requête inconnu(s) : {', '.join(sorted(unknown))}",
            )
    return Depends(_check)


class TokenRequest(BaseModel):
    client_key: str
    client_secret: str


class UserSearchRequest(BaseModel):
    id_organization: float
    last_name: str
    first_name: str
    date_birth: Optional[str] = None  # "YYYY-MM-DD"


class UserMatch(BaseModel):
    userId: int
    score: float
    nom: str
    prenom: str
    date_naissance: Optional[str]
    date_valid: bool


@app.post("/auth/token")
async def get_token(body: TokenRequest):
    """
    Génère un token d'authentification.

    Les credentials sont passés dans le body JSON (jamais dans l'URL).
    Le rôle est déterminé automatiquement à partir des credentials.

    Corps de la requête :
      - client_key    : Clé client
      - client_secret : Secret client

    Returns:
        Token d'accès, rôle attribué et date d'expiration
    """
    client_key = body.client_key
    client_secret = body.client_secret
    # Identifier le rôle à partir des credentials
    matched_role = None
    for role, (key, secret) in ROLE_CREDENTIALS.items():
        if key and secret and client_key == key and client_secret == secret:
            matched_role = role
            break

    if matched_role is None:
        logger.warning(f"[auth] Tentative de connexion échouée pour client_key={client_key!r}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiants invalides"
        )

    logger.info(f"[auth] Connecté via {client_key!r} — rôle : {matched_role.value}")
    token = generate_token()
    expires_at = datetime.utcnow() + timedelta(hours=24)

    active_tokens[token] = {
        "created_at": datetime.utcnow(),
        "expires_at": expires_at,
        "client_key": client_key,
        "role": matched_role,
    }

    return {
        "access_token": token,
        "token_type": "bearer",
        "role": matched_role.value,
        "expires_in": 86400,
        "expires_at": expires_at.isoformat()
    }


class SyncRequest(BaseModel):
    base_url: str
    prefix_table: str


class EtablissementDetailRequest(BaseModel):
    prefix_table: str
    prefix_webresto: Optional[str] = None


@app.post("/sync", dependencies=[Depends(require_role(Role.admin))])
async def sync_all_jobs(body: SyncRequest, env: str = Query(..., alias="environnement_client")):
    """
    Lance la synchronisation complète Webresto → Trino pour tous les jobs.

    Nécessite un Bearer token valide (obtenu via POST /auth/token).

    Corps de la requête :
      - base_url   : URL de base de l'API Webresto
      - prefix_table : Préfixe des tables cibles

    Query param :
      - environnement_client : Identifiant de l'environnement client

    Variables d'environnement côté serveur :
      - OVH_API_KEY / OVH_SECRET_KEY

    Returns:
        Dict { nom_job: { success, status, rows_upserted, errors, warnings, duration_seconds, ... } }
    """
    # Résolution de la clé API Webresto depuis le dict serveur
    secrets_map = _parse_secrets_env("WEBRESTO_SECRETS")

    secret_key = secrets_map.get(body.base_url)
    if not secret_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Aucune clé Webresto configurée pour base_url={body.base_url!r}",
        )

    # Credentials OVH (toujours côté serveur)
    ovh_api_key = os.getenv("OVH_API_KEY")
    ovh_secret_key = os.getenv("OVH_SECRET_KEY")
    if not ovh_api_key or not ovh_secret_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Variables OVH_API_KEY / OVH_SECRET_KEY manquantes",
        )

    try:
        results = await asyncio.to_thread(
            run_all_jobs,
            body.base_url,
            secret_key,
            env,
            body.prefix_table,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la synchronisation : {e}",
        )

    for name, result in results.items():
        _record_sync_result(name, result)

    return {name: dataclasses.asdict(result) for name, result in results.items()}


@app.post("/sync/{job_name}", dependencies=[Depends(require_role(Role.admin))])
async def sync_one_job(job_name: JobName, body: SyncRequest, env: str = Query(..., alias="environnement_client")):
    """
    Lance la synchronisation d'un seul job Webresto → Trino.

    Paramètre de chemin :
      - job_name : nom du job (organization, user, passage, ...)

    Corps de la requête :
      - base_url     : URL de base de l'API Webresto
      - prefix_table : Préfixe des tables cibles

    Query param :
      - environnement_client : Identifiant de l'environnement client

    Returns:
        { success, status, rows_upserted, errors, warnings, duration_seconds, ... }
    """
    secrets_map = _parse_secrets_env("WEBRESTO_SECRETS")

    secret_key = secrets_map.get(body.base_url)
    if not secret_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Aucune clé Webresto configurée pour base_url={body.base_url!r}",
        )

    ovh_api_key = os.getenv("OVH_API_KEY")
    ovh_secret_key = os.getenv("OVH_SECRET_KEY")
    if not ovh_api_key or not ovh_secret_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Variables OVH_API_KEY / OVH_SECRET_KEY manquantes",
        )

    try:
        result = await asyncio.to_thread(
            run_job,
            job_name.value,
            body.base_url,
            secret_key,
            env,
            body.prefix_table,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la synchronisation du job {job_name.value!r} : {e}",
        )

    _record_sync_result(job_name.value, result)

    return dataclasses.asdict(result)


@app.post("/etablissement_detail", dependencies=[Depends(require_role(Role.admin))])
async def sync_etablissement_detail(body: EtablissementDetailRequest, env: str = Query(..., alias="environnement_client")):
    """
    Peuple la table etablissement_detail depuis les tables Trino source + CSV Éducation nationale.

    Corps de la requête :
      - prefix_table : Préfixe des tables cibles (ex: "prod_")

    Query param :
      - environnement_client : Identifiant de l'environnement (ex: "prodcentre")

    Returns:
        { success, status, rows_upserted, errors, warnings, duration_seconds }
    """
    ovh_api_key = os.getenv("OVH_API_KEY")
    ovh_secret_key = os.getenv("OVH_SECRET_KEY")
    if not ovh_api_key or not ovh_secret_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Variables OVH_API_KEY / OVH_SECRET_KEY manquantes",
        )

    try:
        result = await asyncio.to_thread(
            run_etablissement_detail_job,
            env,
            body.prefix_table,
            ovh_api_key,
            ovh_secret_key,
            body.prefix_webresto,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du peuplement de etablissement_detail : {e}",
        )

    return dataclasses.asdict(result)


@app.post("/ref_type_ips_corrections", dependencies=[Depends(require_role(Role.admin))])
async def sync_ref_type_ips_corrections():
    """
    Peuple ref_type_ips_corrections depuis les CSV de corrections manuelles type/IPS.
    Full reload (TRUNCATE + INSERT) — pas de paramètre d'environnement.
    """
    ovh_api_key = os.getenv("OVH_API_KEY")
    ovh_secret_key = os.getenv("OVH_SECRET_KEY")
    if not ovh_api_key or not ovh_secret_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Variables OVH_API_KEY / OVH_SECRET_KEY manquantes",
        )

    try:
        result = await asyncio.to_thread(
            run_ref_type_ips_corrections_job,
            ovh_api_key,
            ovh_secret_key,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du peuplement de ref_type_ips_corrections : {e}",
        )

    return dataclasses.asdict(result)


@app.post("/jours_feries", dependencies=[Depends(require_role(Role.admin))])
async def sync_jours_feries():
    """
    Peuple la table jours_feries depuis jours_feries_metropole.csv.

    Full reload (TRUNCATE + INSERT) — données statiques, pas de paramètre d'environnement.

    Returns:
        { success, status, rows_upserted, errors, warnings, duration_seconds }
    """
    ovh_api_key = os.getenv("OVH_API_KEY")
    ovh_secret_key = os.getenv("OVH_SECRET_KEY")
    if not ovh_api_key or not ovh_secret_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Variables OVH_API_KEY / OVH_SECRET_KEY manquantes",
        )

    try:
        result = await asyncio.to_thread(
            run_jours_feries_job,
            ovh_api_key,
            ovh_secret_key,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du peuplement de jours_feries : {e}",
        )

    return dataclasses.asdict(result)


@app.post("/vacances", dependencies=[Depends(require_role(Role.admin))])
async def sync_vacances(body: None = None):
    """
    Peuple la table vacances_scolaires depuis les fichiers ICS des zones A, B et C.

    Full reload (TRUNCATE + INSERT) — données statiques, pas de paramètre d'environnement.

    Returns:
        { success, status, rows_upserted, errors, warnings, duration_seconds }
    """
    ovh_api_key = os.getenv("OVH_API_KEY")
    ovh_secret_key = os.getenv("OVH_SECRET_KEY")
    if not ovh_api_key or not ovh_secret_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Variables OVH_API_KEY / OVH_SECRET_KEY manquantes",
        )

    try:
        result = await asyncio.to_thread(
            run_vacances_job,
            ovh_api_key,
            ovh_secret_key,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du peuplement de vacances_scolaires : {e}",
        )

    return dataclasses.asdict(result)


@app.post("/cleanup/orphans", dependencies=[Depends(require_role(Role.admin))])
async def cleanup_orphans(body: EtablissementDetailRequest, env: str = Query(..., alias="environnement_client")):
    """
    Supprime les lignes orphelines (id_user absent de la table user) :
      - Registrations de la dernière vague (MAX id_vague)
      - Passages avec date > 2025-08-01

    Corps de la requête :
      - prefix_table : Préfixe des tables cibles (ex: "wr_93_")

    Query param :
      - environnement_client : Identifiant de l'environnement (ex: "prod93")

    Returns:
        { success, status, rows_upserted, errors, warnings, duration_seconds }
    """
    ovh_api_key = os.getenv("OVH_API_KEY")
    ovh_secret_key = os.getenv("OVH_SECRET_KEY")
    if not ovh_api_key or not ovh_secret_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Variables OVH_API_KEY / OVH_SECRET_KEY manquantes",
        )

    try:
        result = await asyncio.to_thread(
            run_cleanup_orphans_job,
            env,
            body.prefix_table,
            ovh_api_key,
            ovh_secret_key,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du nettoyage des orphelins : {e}",
        )

    return dataclasses.asdict(result)


@app.get("/predict/jobs", dependencies=[Depends(require_role(Role.admin, Role.webgerest_readonly, Role.webresto_readonly))])
async def list_prediction_jobs():
    """
    Liste tous les jobs de prédiction lancés.

    Returns:
        Liste de { job_id, model_name, status, started_at }
    """
    return [
        {
            "job_id": j["job_id"],
            "model_name": j["model_name"],
            "status": j["status"],
            "started_at": j["started_at"].isoformat(),
        }
        for j in prediction_jobs.values()
    ]


@app.get("/predict/status/{job_id}", dependencies=[Depends(require_role(Role.admin, Role.webgerest_readonly, Role.webresto_readonly))])
async def get_prediction_status(job_id: str):
    """
    Retourne le statut détaillé d'un job de prédiction.

    Returns:
        { job_id, model_name, status, started_at, finished_at, duration_seconds, error, params }
    """
    if job_id not in prediction_jobs:
        raise HTTPException(status_code=404, detail="Job non trouvé")
    job = prediction_jobs[job_id]
    return {
        **job,
        "started_at": job["started_at"].isoformat(),
        "finished_at": job["finished_at"].isoformat() if job["finished_at"] else None,
        "duration_seconds": (
            (job["finished_at"] or datetime.utcnow()) - job["started_at"]
        ).total_seconds(),
    }


@app.get(
    "/predict/results/{uai}",
    dependencies=[Depends(require_role(Role.admin, Role.webgerest_readonly, Role.webresto_readonly))],
)
async def get_predictions(
    uai: str,
    min_date: str,
    max_date: str,
    env: Optional[str] = Query(None, alias="environnement_client"),
    include_all_models: bool = Query(False),
):
    """
    Retourne les prédictions pour un site (UAI) sur une période donnée.

    Pour chaque (target_date, service), retourne la meilleure prédiction disponible :
    Ensemble (horizon le plus court) → XGBoost → Prophet → ARIMA → MovingAverage.

    Paramètres :
      - uai                  : identifiant UAI du site (path)
      - min_date             : date de début YYYY-MM-DD (query)
      - max_date             : date de fin YYYY-MM-DD (query)
      - environnement_client : ex: 'prodcentre', 'prod13' (query, optionnel)

    Returns:
        Liste de { date, service, prediction }
    """
    from prediction_passages.src.data_prep import _ENV_TO_PREFIX
    from prediction_passages.src.trino_store import passage_predict_table

    ovh_api_key = os.getenv("OVH_API_KEY_READ_ONLY") or os.getenv("OVH_API_KEY")
    ovh_secret_key = os.getenv("OVH_SECRET_KEY_READ_ONLY") or os.getenv("OVH_SECRET_KEY")
    if not ovh_api_key or not ovh_secret_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Variables OVH manquantes",
        )

    if env:
        prefix = _ENV_TO_PREFIX.get(env)
        if prefix is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"environnement_client inconnu : {env!r}. Valeurs : {list(_ENV_TO_PREFIX)}",
            )
        table_name = passage_predict_table(prefix)
    else:
        table_name = "passage_predict"

    try:
        results = await asyncio.to_thread(
            get_predictions_for_site, uai, min_date, max_date,
            ovh_api_key, ovh_secret_key, table_name, include_all_models,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la lecture des prédictions : {e}",
        )

    return results


@app.post(
    "/user/search",
    response_model=list[UserMatch],
    dependencies=[Depends(require_role(Role.admin, Role.webresto_readonly))],
)
async def search_user(
    body: UserSearchRequest,
    env: str = Query(..., alias="environnement_client"),
):
    """
    Retourne les 10 userId les plus probables pour un triplet
    (nom, prénom, date_naissance) dans une organisation donnée.

    Scoring Jaro-Winkler sur nom/prénom + correspondance exacte sur date_naissance.

    Corps de la requête :
      - id_organization : identifiant de l'organisation
      - last_name       : nom de famille (ENT)
      - first_name      : prénom (ENT)
      - date_birth      : date de naissance YYYY-MM-DD (optionnel)

    Query param :
      - environnement_client : identifiant de l'environnement (ex: "prodcentre")
        Le préfixe des tables cibles (ex: "wr_centre_") est déduit automatiquement
        de l'environnement via _WEBRESTO_SERVER_CONFIG.

    Returns:
        Top 10 { userId, score, nom, prenom, date_naissance, date_valid }
        Si date_birth fourni : 5 avec date_valid=true + 5 avec date_valid=false.
        Sinon : 10 meilleurs scores, tous date_valid=false.
    """
    ovh_api_key = os.getenv("OVH_API_KEY_READ_ONLY") or os.getenv("OVH_API_KEY")
    ovh_secret_key = os.getenv("OVH_SECRET_KEY_READ_ONLY") or os.getenv("OVH_SECRET_KEY")
    if not ovh_api_key or not ovh_secret_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Variables OVH manquantes",
        )

    prefix_table = next(
        (cfg["table_prefix"] for cfg in _WEBRESTO_SERVER_CONFIG.values() if cfg["environnement_client"] == env),
        None,
    )
    if prefix_table is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"environnement_client inconnu : {env!r}. "
                f"Valeurs : {[cfg['environnement_client'] for cfg in _WEBRESTO_SERVER_CONFIG.values()]}"
            ),
        )

    try:
        results = await asyncio.to_thread(
            find_user_candidates,
            body.id_organization,
            body.last_name,
            body.first_name,
            body.date_birth,
            env,
            prefix_table,
            ovh_api_key,
            ovh_secret_key,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la recherche utilisateur : {e}",
        )

    return results


@app.post("/predict/all", dependencies=[Depends(require_role(Role.admin))])
async def launch_all_predictions(
    body: PredictionRequest = PredictionRequest(),
    env: Optional[str] = Query(None, alias="environnement_client"),
):
    """
    Lance tous les modèles rolling séquentiellement en arrière-plan
    (ARIMA → Prophet → XGBoost → MovingAverage → Ensemble).

    Retourne immédiatement un job_id pour suivre l'avancement via GET /predict/status/{job_id}.

    Corps de la requête (optionnel) : paramètres partagés par les modèles.

    Query param :
      - environnement_client : Identifiant de l'environnement client (ex: "prodcentre")
    """
    _check_prediction_available()
    if not os.getenv("OVH_API_KEY") or not os.getenv("OVH_SECRET_KEY"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Variables OVH_API_KEY / OVH_SECRET_KEY manquantes",
        )

    # Vérifier qu'aucun job "all" n'est déjà en cours
    for job in prediction_jobs.values():
        if job["model_name"] == "all" and job["status"] == "running":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Un job 'all' est déjà en cours (job_id={job['job_id']})",
            )

    job_id = str(uuid.uuid4())
    prediction_jobs[job_id] = {
        "job_id": job_id,
        "model_name": "all",
        "status": "running",
        "started_at": datetime.utcnow(),
        "finished_at": None,
        "error": None,
        "params": body.model_dump(exclude_none=True),
    }

    async def _run_all():
        PREDICTION_JOBS_RUNNING.inc()
        try:
            # Dataprep unique partagé entre tous les modèles
            shared_data_prep = PredictionDataPreparation(
                use_manual_entry=True,
                env=env,
                prefix=body.prefix or None,
            )
            shared_df = await asyncio.to_thread(shared_data_prep.load_and_prepare)

            shared = {'df': shared_df, 'data_prep': shared_data_prep}
            base = {k: v for k, v in {
                'horizon_days': body.horizon_days,
                'step_days': body.step_days,
                'min_train_weeks': body.min_train_weeks,
            }.items() if v is not None}
            ma_kw = {k: v for k, v in {
                'horizon_weeks': body.horizon_weeks,
                'step_days': body.step_days,
                'min_train_weeks': body.min_train_weeks,
            }.items() if v is not None}

            fsd = {'force_start_date': body.force_start_date} if body.force_start_date else {}
            steps = [
                (run_arima_rolling,    {**base, **shared, **fsd}),
                (run_prophet_rolling,  {**base, **shared, **fsd, 'variant': '21'}),
                (run_prophet_rolling,  {**base, **shared, **fsd, 'variant': '35'}),
                (run_xgb_rolling,      {**base, **shared, **fsd, 'variant': '21'}),
                (run_xgb_rolling,      {**base, **shared, **fsd, 'variant': '35'}),
                (run_ma_rolling,       {**ma_kw, **shared, **fsd}),
            ]
            if body.run_ensemble:
                ens_kw = {'prefix': shared_data_prep._prefix}
                if body.force_start_date:
                    ens_kw['force_start_date'] = body.force_start_date
                steps.append((run_ensemble_rolling, ens_kw))

            for func, kwargs in steps:
                await asyncio.to_thread(func, **kwargs)
            prediction_jobs[job_id]["status"] = "completed"
            PREDICTION_JOBS_COMPLETED.labels(model="all").inc()
        except Exception as e:
            prediction_jobs[job_id]["status"] = "failed"
            prediction_jobs[job_id]["error"] = f"{type(e).__name__}: {e}"
            PREDICTION_JOBS_FAILED.labels(model="all").inc()
        finally:
            prediction_jobs[job_id]["finished_at"] = datetime.utcnow()
            PREDICTION_JOBS_RUNNING.dec()

    asyncio.create_task(_run_all())

    return {
        "job_id": job_id,
        "model_name": "all",
        "status": "running",
        "message": "Tous les modèles rolling lancés en arrière-plan.",
    }


@app.post("/predict/global_dep_xgb", dependencies=[Depends(require_role(Role.admin))])
async def launch_global_dep_xgb(
    body: PredictionRequest = PredictionRequest(),
):
    """
    Lance le job GlobalDepXGB : XGBoost entraîné sur tous les établissements départementaux
    (CD13, CD18, CD41, CD89, CD28, CD19, CD21…) de plusieurs environnements simultanément.

    Les prédictions sont stockées dans la table passage_predict de chaque env.
    Variant fixé à '35' — pas de paramètre environnement_client.

    Corps de la requête (optionnel) :
      - horizon_days     : fenêtre de prédiction (max 35)
      - step_days        : pas d'avancement en jours
      - min_train_weeks  : semaines minimum d'historique
      - force_start_date : forcer le redémarrage depuis cette date
    """
    _check_prediction_available()
    if not os.getenv("OVH_API_KEY") or not os.getenv("OVH_SECRET_KEY"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Variables OVH_API_KEY / OVH_SECRET_KEY manquantes",
        )

    job_name = "global_dep_xgb"
    for job in prediction_jobs.values():
        if job["model_name"] == job_name and job["status"] == "running":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Un job '{job_name}' est déjà en cours (job_id={job['job_id']})",
            )

    kwargs: dict = {}
    if body.horizon_days is not None:
        kwargs["horizon_days"] = body.horizon_days
    if body.step_days is not None:
        kwargs["step_days"] = body.step_days
    if body.min_train_weeks is not None:
        kwargs["min_train_weeks"] = body.min_train_weeks
    if body.force_start_date is not None:
        kwargs["force_start_date"] = body.force_start_date

    job_id = str(uuid.uuid4())
    prediction_jobs[job_id] = {
        "job_id": job_id,
        "model_name": job_name,
        "status": "running",
        "started_at": datetime.utcnow(),
        "finished_at": None,
        "error": None,
        "params": kwargs,
    }

    asyncio.create_task(
        asyncio.to_thread(_run_prediction_job, job_id, run_global_dep_xgb_rolling, kwargs)
    )

    return {
        "job_id": job_id,
        "model_name": job_name,
        "status": "running",
        "message": "Job 'GlobalDepXGB' lancé en arrière-plan.",
    }


@app.post("/predict/{model_name}", dependencies=[Depends(require_role(Role.admin))])
async def launch_prediction(
    model_name: PredictionModel,
    body: PredictionRequest = PredictionRequest(),
    env: Optional[str] = Query(None, alias="environnement_client"),
):
    """
    Lance un job de prédiction rolling en arrière-plan.

    Modèles disponibles : arima, prophet, xgboost, moving_average, ensemble.

    Retourne immédiatement un job_id pour suivre l'avancement via GET /predict/status/{job_id}.

    Query param :
      - environnement_client : Identifiant de l'environnement client (ex: "prodcentre")

    Corps de la requête (optionnel) :
      - horizon_days      : fenêtre de prédiction (ARIMA, Prophet, XGBoost)
      - step_days          : pas d'avancement en jours
      - min_train_weeks    : semaines minimum d'historique
      - variant            : '21' ou '35' (Prophet, XGBoost)
      - horizon_weeks      : semaines de prédiction (MovingAverage)
      - use_retry           : utiliser le mode retry pour Ensemble
      - wait / max_retries : paramètres retry (Ensemble)
    """
    _check_prediction_available()
    if not os.getenv("OVH_API_KEY") or not os.getenv("OVH_SECRET_KEY"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Variables OVH_API_KEY / OVH_SECRET_KEY manquantes",
        )

    # Vérifier qu'aucun job du même modèle n'est déjà en cours
    for job in prediction_jobs.values():
        if job["model_name"] == model_name.value and job["status"] == "running":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Un job '{model_name.value}' est déjà en cours (job_id={job['job_id']})",
            )

    func, kwargs = _build_kwargs(model_name, body, env)

    job_id = str(uuid.uuid4())
    prediction_jobs[job_id] = {
        "job_id": job_id,
        "model_name": model_name.value,
        "status": "running",
        "started_at": datetime.utcnow(),
        "finished_at": None,
        "error": None,
        "params": kwargs,
    }

    asyncio.create_task(
        asyncio.to_thread(_run_prediction_job, job_id, func, kwargs)
    )

    return {
        "job_id": job_id,
        "model_name": model_name.value,
        "status": "running",
        "message": f"Job '{model_name.value}' lancé en arrière-plan.",
    }


# ── Helpers secrets ──────────────────────────────────────────────────────────


def _parse_secrets_env(var_name: str) -> dict:
    """
    Lit une variable d'environnement contenant un JSON et la parse.
    Tolère les valeurs wrappées dans des guillemets doubles
    (ex: WEBGEREST_SECRETS="{...}") telles que certaines plateformes les injectent.
    """
    raw = os.getenv(var_name, "{}").strip()
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Variable {var_name} invalide (JSON malformé)",
        )


# ── Helpers Webgerest ────────────────────────────────────────────────────────


def _resolve_webgerest_secrets(base_url: str) -> tuple[str, str]:
    """Lit WEBGEREST_SECRETS et retourne (client_key, client_secret) pour base_url."""
    secrets_map = _parse_secrets_env("WEBGEREST_SECRETS")
    entry = secrets_map.get(base_url)
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Aucun secret Webgerest configuré pour base_url={base_url!r}",
        )
    if isinstance(entry, dict):
        client_key = entry.get("client_key", "")
        client_secret = entry.get("client_secret", "")
    else:
        client_key, client_secret = entry[0], entry[1]
    if not client_key or not client_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Credentials Webgerest incomplets pour base_url={base_url!r}",
        )
    return client_key, client_secret


class WebgestLoginSyncRequest(BaseModel):
    base_url: str
    dataset: str
    prefix: str
    login_group: str   # un seul appel suffit pour récupérer toute la table login


class WebgestSyncRequest(BaseModel):
    base_url: str
    dataset: str
    prefix: str
    login_groups: list[str] | None = None


class WebgestTableSyncRequest(WebgestSyncRequest):
    mode: str = "merge"
    max_workers: int = 5


def _resolve_login_groups(login_groups: list[str] | None, dataset: str, prefix: str, ovh_api_key: str, ovh_secret_key: str) -> list[str]:
    """Retourne login_groups tel quel, ou le lit depuis SELECT DISTINCT login_group FROM {prefix}login."""
    if login_groups:
        return login_groups
    db = TrinoClient(dataset, ovh_api_key, ovh_secret_key)
    df = db.query_as_dataframe(f"SELECT DISTINCT login_group FROM {prefix}login WHERE login_group IS NOT NULL")
    groups = df["login_group"].tolist()
    if not groups:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Aucun login_group trouvé dans {prefix}login — sync login requis au préalable",
        )
    logger.info(f"[login_groups] {len(groups)} groupe(s) lus depuis {prefix}login : {groups}")
    return groups


# ── Routes Webgerest sync ─────────────────────────────────────────────────────


@app.post("/webgerest/sync/login", dependencies=[Depends(require_role(Role.admin))])
async def webgerest_sync_login(body: WebgestLoginSyncRequest):
    """
    Synchronise la table login Webgerest (upsert par login).

    Un seul appel API avec login_group suffit pour récupérer l'ensemble de la table.

    Corps de la requête :
      - base_url    : URL de base de l'API Webgerest (clé dans WEBGEREST_SECRETS)
      - dataset     : environnement Trino (ex: "prodcentre")
      - prefix      : préfixe des tables (ex: "wg_")
      - login_group : groupe utilisé pour l'appel API (ex: "REG-CENT")

    Returns:
        { success, status, rows_upserted, errors, warnings, duration_seconds }
    """
    client_key, client_secret = _resolve_webgerest_secrets(body.base_url)
    ovh_api_key = os.getenv("OVH_API_KEY")
    ovh_secret_key = os.getenv("OVH_SECRET_KEY")
    if not ovh_api_key or not ovh_secret_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Variables OVH_API_KEY / OVH_SECRET_KEY manquantes",
        )
    try:
        result = await asyncio.to_thread(
            run_webgerest_login_job,
            body.dataset, body.prefix, body.login_group,
            body.base_url, client_key, client_secret,
            ovh_api_key, ovh_secret_key,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur login sync : {e}",
        )
    return dataclasses.asdict(result)


@app.post("/webgerest/sync/descfic", dependencies=[Depends(require_role(Role.admin))])
async def webgerest_sync_descfic(body: WebgestSyncRequest):
    """
    Synchronise la table descfic Webgerest (bulk_replace par login_group).

    Corps de la requête :
      - base_url     : URL de base de l'API Webgerest
      - dataset      : environnement Trino
      - prefix       : préfixe des tables
      - login_groups : groupes à synchroniser

    Returns:
        { success, status, rows_upserted, errors, warnings, duration_seconds }
    """
    client_key, client_secret = _resolve_webgerest_secrets(body.base_url)
    ovh_api_key = os.getenv("OVH_API_KEY")
    ovh_secret_key = os.getenv("OVH_SECRET_KEY")
    if not ovh_api_key or not ovh_secret_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Variables OVH_API_KEY / OVH_SECRET_KEY manquantes",
        )
    try:
        login_groups = _resolve_login_groups(body.login_groups, body.dataset, body.prefix, ovh_api_key, ovh_secret_key)
        result = await asyncio.to_thread(
            run_webgerest_descfic_job,
            body.dataset, body.prefix, login_groups,
            body.base_url, client_key, client_secret,
            ovh_api_key, ovh_secret_key,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur descfic sync : {e}",
        )
    return dataclasses.asdict(result)


@app.post("/webgerest/sync/table/{table_name}", dependencies=[Depends(require_role(Role.admin))])
async def webgerest_sync_table(table_name: str, body: WebgestTableSyncRequest):
    """
    Synchronise une table Webgerest avec routing DESCFIC.

    Routing : statut=1 → un fetch par login_group / statut=2 → un fetch par login_site.
    Fetch parallèle (ThreadPoolExecutor), écriture séquentielle.

    Paramètre de chemin :
      - table_name : clé dans WEBGEREST_SCHEMAS (ex: "article", "mvtart")

    Corps de la requête :
      - base_url     : URL de base de l'API Webgerest
      - dataset      : environnement Trino
      - prefix       : préfixe des tables
      - login_groups : groupes à synchroniser
      - mode         : "merge" | "bulk_append" | "bulk_replace" (défaut: "merge")
      - max_workers  : parallélisme des appels HTTP (défaut: 5)

    Returns:
        { success, status, rows_upserted, errors, warnings, duration_seconds }
    """
    from data_process.process.schemas_webgerest import WEBGEREST_SCHEMAS
    if table_name not in WEBGEREST_SCHEMAS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Table inconnue : {table_name!r}. Tables disponibles : {list(WEBGEREST_SCHEMAS)}",
        )
    if body.mode not in ("merge", "bulk_append", "bulk_replace"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Mode invalide : {body.mode!r}. Valeurs : merge | bulk_append | bulk_replace",
        )
    client_key, client_secret = _resolve_webgerest_secrets(body.base_url)
    ovh_api_key = os.getenv("OVH_API_KEY")
    ovh_secret_key = os.getenv("OVH_SECRET_KEY")
    if not ovh_api_key or not ovh_secret_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Variables OVH_API_KEY / OVH_SECRET_KEY manquantes",
        )
    try:
        login_groups = _resolve_login_groups(body.login_groups, body.dataset, body.prefix, ovh_api_key, ovh_secret_key)
        result = await asyncio.to_thread(
            run_webgerest_table_job,
            table_name, body.dataset, body.prefix, login_groups,
            body.base_url, client_key, client_secret,
            ovh_api_key, ovh_secret_key,
            body.mode, body.max_workers,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur sync table {table_name!r} : {e}",
        )
    return dataclasses.asdict(result)


@app.get(
    "/webgerest/stats/dashboard",
    dependencies=[Depends(require_role(Role.admin, Role.webgerest_readonly))],
)
async def webgerest_stats_dashboard(
    serveur: str,
    annee: str,
    login_group: str,
    hors_taxe: bool = False,
):
    """
    Retourne les statistiques du tableau de bord EGAlim / bio / local pour un groupe et une année.

    Paramètres :
      - serveur     : identifiant du serveur (ex: "centre")
      - annee       : année au format YYYY (ex: "2024")
      - login_group : groupe de collectivités (dépend du serveur)
      - hors_taxe   : utiliser les montants HT (défaut: False → TTC)

    Returns:
        JSON avec montants agrégés par catégorie, nb_sites, et stats par service
        (service_dejeuner, service_journee).
    """
    # Validation des paramètres d'entrée
    if serveur not in _WEBGEREST_SERVER_CONFIG:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Serveur inconnu. Valeurs valides : {list(_WEBGEREST_SERVER_CONFIG.keys())}",
        )
    config = _WEBGEREST_SERVER_CONFIG[serveur]
    if login_group not in config["login_groups"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"login_group inconnu pour serveur='{serveur}'. Valeurs valides : {config['login_groups']}",
        )

    ovh_api_key = os.getenv("OVH_API_KEY_READ_ONLY") or os.getenv("OVH_API_KEY")
    ovh_secret_key = os.getenv("OVH_SECRET_KEY_READ_ONLY") or os.getenv("OVH_SECRET_KEY")
    if not ovh_api_key or not ovh_secret_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Variables OVH manquantes",
        )

    ec = config["environnement_client"]
    prefix = config["prefix"]

    try:
        df_dashboard, df_effect = await asyncio.gather(
            asyncio.to_thread(
                _query_stats_dashboard, ec, prefix, login_group, annee, hors_taxe,
                ovh_api_key, ovh_secret_key,
            ),
            asyncio.to_thread(
                _query_stats_dashboard_effect, ec, prefix, login_group, annee,
                ovh_api_key, ovh_secret_key,
            ),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération des données : {e}",
        )

    return get_stats_dashboard(df_dashboard, df_effect)


class WebgestReportingFcjRequest(BaseModel):
    dataset: str
    prefix: str
    annee: Optional[str] = None
    zone_scolaire: str = "B"


class WebgestReportingLivRequest(BaseModel):
    dataset: str
    prefix: str
    annee: Optional[str] = None


@app.post("/webgerest/reporting/fcj", dependencies=[Depends(require_role(Role.admin))])
async def webgerest_reporting_fcj(body: WebgestReportingFcjRequest):
    """
    Calcule et écrit les tables de reporting FCJ Webgerest (stats_fcj59, stats_fcj59_detail,
    stats_recap_site, stat_effect_cred_1, stats_dashboard_effect).

    Corps de la requête :
      - dataset        : environnement Trino (ex: "prodcentre")
      - prefix         : préfixe des tables (ex: "wg_")
      - annee        : filtre optionnel, format "2024" (année civile)
      - zone_scolaire  : zone pour les jours ouvrés (défaut: "B")

    Returns:
        { success, status, rows_upserted, errors, warnings, duration_seconds }
    """
    ovh_api_key = os.getenv("OVH_API_KEY")
    ovh_secret_key = os.getenv("OVH_SECRET_KEY")
    if not ovh_api_key or not ovh_secret_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Variables OVH_API_KEY / OVH_SECRET_KEY manquantes",
        )
    try:
        result = await asyncio.to_thread(
            run_webgerest_reporting_fcj_job,
            body.dataset, body.prefix, ovh_api_key, ovh_secret_key,
            body.annee, body.zone_scolaire,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur reporting FCJ : {e}",
        )
    return dataclasses.asdict(result)


@app.post("/webgerest/reporting/liv", dependencies=[Depends(require_role(Role.admin))])
async def webgerest_reporting_liv(body: WebgestReportingLivRequest):
    """
    Calcule et écrit les tables de reporting LIV Webgerest (stats_liv59, stats_liv,
    stats_liv_mois, stats_liv_annee, stats_liv_egalim, stats_dashboard).

    Corps de la requête :
      - dataset : environnement Trino (ex: "prodcentre")
      - prefix  : préfixe des tables (ex: "wg_")
      - annee   : filtre optionnel, format "2024" (année civile)

    Returns:
        { success, status, rows_upserted, errors, warnings, duration_seconds }
    """
    ovh_api_key = os.getenv("OVH_API_KEY")
    ovh_secret_key = os.getenv("OVH_SECRET_KEY")
    if not ovh_api_key or not ovh_secret_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Variables OVH_API_KEY / OVH_SECRET_KEY manquantes",
        )
    try:
        result = await asyncio.to_thread(
            run_webgerest_reporting_liv_job,
            body.dataset, body.prefix, ovh_api_key, ovh_secret_key, body.annee,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur reporting LIV : {e}",
        )
    return dataclasses.asdict(result)


# ── Stats WeResto ─────────────────────────────────────────────────────────────

_WEBRESTO_SERVER_CONFIG: dict[str, dict] = {
    "centre": {
        "environnement_client": "prodcentre",
        "table_prefix": "wr_centre_",
    },
}


def _webresto_keys() -> tuple[str, str]:
    api_key = os.getenv("OVH_API_KEY_READ_ONLY") or os.getenv("OVH_API_KEY")
    sec_key = os.getenv("OVH_SECRET_KEY_READ_ONLY") or os.getenv("OVH_SECRET_KEY")
    if not api_key or not sec_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Variables OVH manquantes",
        )
    return api_key, sec_key


def _webresto_config(environnement: str) -> tuple[str, str]:
    if environnement not in _WEBRESTO_SERVER_CONFIG:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Environnement inconnu. Valeurs valides : {list(_WEBRESTO_SERVER_CONFIG.keys())}",
        )
    cfg = _WEBRESTO_SERVER_CONFIG[environnement]
    return cfg["environnement_client"], cfg["table_prefix"]


def _export_filename(
    school_year: str,
    ext: str,
    department: List[str] | None,
    type_orga: List[str] | None,
    facturation_type: List[str] | None,
    ips_min: float | None,
    ips_max: float | None,
    id_organization: List[int] | None = None,
    access_software: List[str] | None = None,
) -> str:
    date_str = datetime.utcnow().strftime("%Y%m%d")
    parts = [f"export_webresto_{school_year}_{date_str}"]
    if id_organization:
        parts.append("org-" + "-".join(str(i) for i in id_organization))
    if department:
        parts.append("dept-" + "-".join(department))
    if type_orga:
        parts.append("type-" + "-".join(type_orga))
    if facturation_type:
        parts.append("ftype-" + "-".join(facturation_type))
    if access_software:
        parts.append("soft-" + "-".join(access_software))
    if ips_min is not None or ips_max is not None:
        a = str(ips_min) if ips_min is not None else ""
        b = str(ips_max) if ips_max is not None else ""
        parts.append(f"ips-{a}-{b}")
    return "_".join(parts) + f".{ext}"


@app.get(
    "/webresto/stats/general",
    dependencies=[
        Depends(require_role(Role.admin, Role.webresto_readonly)),
        strict_query_params(
            "environnement", "school_year",
            "nom_etablissement", "facturation_type", "department", "type",
            "access_software", "ips_min", "ips_max", "id_organization",
        ),
    ],
)
async def webresto_stats_general(
    environnement: str,
    school_year: str,
    nom_etablissement: Optional[List[str]] = Query(default=None),
    facturation_type: Optional[List[str]] = Query(default=None),
    department: Optional[List[str]] = Query(default=None),
    type_orga: Optional[List[str]] = Query(default=None, alias="type"),
    access_software: Optional[List[str]] = Query(default=None),
    ips_min: Optional[float] = None,
    ips_max: Optional[float] = None,
    id_organization: Optional[List[int]] = Query(default=None),
):
    """
    Statistiques générales WeResto (onglet Général).

    Retourne :
      - kpis : totaux (dossiers_deposes, dossiers_valides, etc.)
      - par_tranche : dossiers_valides par (tranche, facturation_type)
      - par_categorie : stats par (tranche, sous-groupe, mode_transmission)
      - effectif_par_etablissement : effectif réel depuis organization_enrollment
    """
    ec, prefix = _webresto_config(environnement)
    api_key, sec_key = _webresto_keys()

    ck = _cache_key(
        environnement, school_year,
        nom_etablissement, facturation_type, department, type_orga, access_software,
        ips_min, ips_max, id_organization,
    )
    cached = get_cached_response(ck)
    if cached is not None:
        logger.info("[general] cache hit")
        return cached

    t_total = time.perf_counter()
    try:
        t0 = time.perf_counter()
        school_year_id = await asyncio.to_thread(
            resolve_school_year_id, ec, prefix, api_key, sec_key, school_year,
        )
        logger.info(f"[general] resolve_school_year_id: {time.perf_counter() - t0:.3f}s (id={school_year_id})")

        t1 = time.perf_counter()
        (kpis, par_tranche), par_categorie, df_enrollment, df_suivi = await asyncio.gather(
            asyncio.to_thread(
                query_kpis_and_par_tranche,
                ec, prefix, api_key, sec_key, school_year_id,
                nom_etablissement, facturation_type, department, type_orga, access_software,
                ips_min, ips_max, id_organization,
            ),
            asyncio.to_thread(
                query_general_tarif2,
                ec, prefix, api_key, sec_key, school_year,
                nom_etablissement, department, type_orga, access_software,
                school_year_id, id_organization,
            ),
            asyncio.to_thread(
                query_general_enrollment,
                ec, prefix, api_key, sec_key, school_year,
                nom_etablissement, department, type_orga, access_software,
                ips_min, ips_max, school_year_id, id_organization,
            ),
            asyncio.to_thread(
                query_recours_inscriptions,
                ec, prefix, api_key, sec_key, school_year,
                nom_etablissement, department, type_orga, access_software,
                ips_min, ips_max, school_year_id, id_organization,
            ),
        )
        logger.info(f"[general] gather (4 threads): {time.perf_counter() - t1:.3f}s")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération des données : {e}",
        )

    result = assemble_general(kpis, par_tranche, par_categorie, df_enrollment, df_suivi)
    set_cached_response(ck, result)
    logger.info(f"[general] total: {time.perf_counter() - t_total:.3f}s")
    return result


@app.get(
    "/webresto/stats/recours",
    dependencies=[
        Depends(require_role(Role.admin, Role.webresto_readonly)),
        strict_query_params(
            "environnement", "school_year",
            "nom_etablissement", "department", "type",
            "facturation_type", "access_software", "ips_min", "ips_max", "id_organization",
        ),
    ],
)
async def webresto_stats_recours(
    environnement: str,
    school_year: str,
    nom_etablissement: Optional[List[str]] = Query(default=None),
    department: Optional[List[str]] = Query(default=None),
    type_orga: Optional[List[str]] = Query(default=None, alias="type"),
    facturation_type: Optional[List[str]] = Query(default=None),
    access_software: Optional[List[str]] = Query(default=None),
    ips_min: Optional[float] = None,
    ips_max: Optional[float] = None,
    id_organization: Optional[List[int]] = Query(default=None),
):
    """
    Onglet Recours — données restreintes à facturation_type IN ('ticket', 'interne').
    Non disponible pour les environnements 93.
    """
    ec, prefix = _webresto_config(environnement)
    if "93" in ec:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="La route /recours n'est pas disponible pour les environnements 93.",
        )
    api_key, sec_key = _webresto_keys()

    ck = _cache_key(
        "recours", environnement, school_year,
        nom_etablissement, department, type_orga, facturation_type, access_software,
        ips_min, ips_max, id_organization,
    )
    cached = get_cached_response(ck)
    if cached is not None:
        logger.info("[recours] cache hit")
        return cached

    try:
        school_year_id = await asyncio.to_thread(
            resolve_school_year_id, ec, prefix, api_key, sec_key, school_year,
        )
        (kpis, par_tranche), df_enrollment, df_validations = await asyncio.gather(
            asyncio.to_thread(
                query_recours_kpis_par_tranche,
                ec, prefix, api_key, sec_key, school_year_id,
                nom_etablissement, department, type_orga, access_software,
                ips_min, ips_max, id_organization, facturation_type,
            ),
            asyncio.to_thread(
                query_general_enrollment,
                ec, prefix, api_key, sec_key, school_year,
                nom_etablissement, department, type_orga, access_software,
                ips_min, ips_max, school_year_id, id_organization,
            ),
            asyncio.to_thread(
                query_recours_validations,
                ec, prefix, api_key, sec_key, school_year,
                nom_etablissement, department, type_orga, access_software,
                ips_min, ips_max, school_year_id, id_organization,
            ),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération des données : {e}",
        )

    result = assemble_recours(kpis, par_tranche, df_enrollment, df_validations)
    set_cached_response(ck, result)
    return result


@app.get(
    "/webresto/filters",
    dependencies=[
        Depends(require_role(Role.admin, Role.webresto_readonly)),
        strict_query_params("environnement", "school_year", "page"),
    ],
)
async def webresto_filters(
    environnement: str,
    school_year: str,
    page: str,
):
    """
    Valeurs disponibles pour chaque filtre de la page donnée.
    Pour page='inscription' : filtre sur tarification_1_sc{school_year_id}.
    """
    ec, prefix = _webresto_config(environnement)
    api_key, sec_key = _webresto_keys()

    try:
        school_year_id = await asyncio.to_thread(
            resolve_school_year_id, ec, prefix, api_key, sec_key, school_year,
        )
        result = await asyncio.to_thread(
            query_available_filters,
            ec, prefix, api_key, sec_key, school_year_id, page,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération des filtres : {e}",
        )

    return result


@app.get(
    "/webresto/stats/passages",
    dependencies=[
        Depends(require_role(Role.admin, Role.webresto_readonly)),
        strict_query_params(
            "environnement", "school_year",
            "nom_etablissement", "id_organization",
            "facturation_type", "access_software", "ips_min", "ips_max",
            "service", "date_debut", "date_fin",
        ),
    ],
)
async def webresto_stats_passages(
    environnement: str,
    school_year: str,
    nom_etablissement: Optional[List[str]] = Query(default=None),
    id_organization: Optional[List[int]] = Query(default=None),
    facturation_type: Optional[List[str]] = Query(default=None),
    access_software: Optional[List[str]] = Query(default=None),
    ips_min: Optional[float] = None,
    ips_max: Optional[float] = None,
    service: Optional[List[str]] = Query(default=None),
    date_debut: Optional[str] = None,
    date_fin: Optional[str] = None,
):
    ec, prefix = _webresto_config(environnement)
    if "93" in ec:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Route /passages non disponible pour les environnements 93.",
        )
    api_key, sec_key = _webresto_keys()

    cache_k = _cache_key(
        "passages", ec, prefix, school_year,
        nom_etablissement, id_organization, facturation_type,
        access_software, ips_min, ips_max,
        service, date_debut, date_fin,
    )
    if (cached := get_cached_response(cache_k)) is not None:
        return cached

    school_year_id = await asyncio.to_thread(
        resolve_school_year_id, ec, prefix, api_key, sec_key, school_year
    )

    df_passages, df_enrollment = await asyncio.gather(
        asyncio.to_thread(
            query_passages_tarif3_cached,
            ec, prefix, api_key, sec_key, school_year_id,
            nom_etablissement, id_organization, facturation_type,
            access_software, ips_min, ips_max,
            service, date_debut, date_fin,
        ),
        asyncio.to_thread(
            query_general_enrollment,
            ec, prefix, api_key, sec_key, school_year,
            nom_etablissement, None, None,
            access_software, ips_min, ips_max,
            school_year_id, id_organization,
        ),
    )

    result = assemble_passages(df_passages, df_enrollment)
    set_cached_response(cache_k, result)
    return result


@app.get(
    "/webresto/stats/export",
    dependencies=[
        Depends(require_role(Role.admin, Role.webresto_readonly)),
        strict_query_params(
            "environnement", "school_year", "format",
            "nom_etablissement", "facturation_type", "department", "type",
            "access_software", "ips_min", "ips_max", "id_organization",
        ),
    ],
)
async def webresto_stats_export(
    environnement: str,
    school_year: str,
    format: Optional[str] = None,
    nom_etablissement: Optional[List[str]] = Query(default=None),
    facturation_type: Optional[List[str]] = Query(default=None),
    department: Optional[List[str]] = Query(default=None),
    type_orga: Optional[List[str]] = Query(default=None, alias="type"),
    access_software: Optional[List[str]] = Query(default=None),
    ips_min: Optional[float] = None,
    ips_max: Optional[float] = None,
    id_organization: Optional[List[int]] = Query(default=None),
):
    """
    Export agrégé par établissement (onglet Export).

    format omis ou "json" → liste JSON.
    format "csv" ou "excel" → fichier téléchargeable.
    """
    if format not in (None, "json", "csv", "excel"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Paramètre format invalide. Valeurs valides : json, csv, excel",
        )

    ec, prefix = _webresto_config(environnement)
    api_key, sec_key = _webresto_keys()

    try:
        df_raw = await asyncio.to_thread(
            query_export_tarif1,
            ec, prefix, api_key, sec_key, school_year,
            nom_etablissement, facturation_type, department, type_orga, access_software,
            ips_min, ips_max, id_organization,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération des données : {e}",
        )

    df = assemble_export(df_raw)

    if format in (None, "json"):
        return _df_to_records(df)

    # Pour le filename, résoudre les IDs depuis le DataFrame si seul nom_etablissement a été passé
    org_ids_for_filename = id_organization
    if not org_ids_for_filename and nom_etablissement and "id_organization" in df_raw.columns:
        org_ids_for_filename = sorted(
            int(x) for x in df_raw["id_organization"].dropna().unique()
        )

    df_labeled = df.rename(columns=EXPORT_COL_LABELS)

    if format == "excel":
        filename = _export_filename(
            school_year, "xlsx", department, type_orga, facturation_type, ips_min, ips_max,
            org_ids_for_filename, access_software,
        )
        buf = io.BytesIO()
        df_labeled.to_excel(buf, index=False, engine="openpyxl")
        buf.seek(0)
        return Response(
            content=buf.read(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # csv
    filename = _export_filename(
        school_year, "csv", department, type_orga, facturation_type, ips_min, ips_max,
        org_ids_for_filename, access_software,
    )
    csv_data = df_labeled.to_csv(index=False)
    return StreamingResponse(
        iter([csv_data]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.on_event("startup")
async def _configure_log_format():
    fmt = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", ""):
        for handler in logging.getLogger(name).handlers:
            handler.setFormatter(fmt)

    # Le root logger n'a pas de handler quand uvicorn est lancé via CMD — les loggers
    # data_process.* propagent vers lui mais leurs messages sont silencieusement ignorés.
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        root.addHandler(handler)


@app.get("/")
async def root():
    """Point d'entrée de l'API"""
    return {
        "message": "Webgerest Data API",
        "version": "1.0.5",
        "endpoints": {
            "auth": "/auth/token",
            "health": "/health",
            "sync": "/sync",
            "predict": "/predict/{model_name}",
        }
    }



# ── Superset Embedding ────────────────────────────────────────────────────────


def _superset_url(environnement: str) -> str:
    urls = _parse_secrets_env("SUPERSET_URLS")
    url = urls.get(environnement)
    if not url:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Environnement Superset inconnu. Valeurs valides : {list(urls.keys())}",
        )
    return url


@app.get(
    "/superset/guest-token",
    dependencies=[Depends(require_role(Role.admin, Role.webgerest_readonly, Role.webresto_readonly))],
)
async def superset_guest_token(dashboard_id: str, environnement_client: str):
    """
    Retourne un guest token Superset pour embarquer un dashboard en mode embedded SDK.

    Le front doit appeler cette route depuis fetchGuestToken — ne jamais appeler
    /api/v1/security/guest_token/ directement depuis le client.

    Paramètres :
      - dashboard_id       : UUID du dashboard (visible dans Dashboard → Embed)
      - environnement_client : "centre" ou "93"

    Variables d'environnement :
      - SUPERSET_URLS      : JSON map {"centre": "https://...", "93": "https://..."}
      - SUPERSET_USERNAME  : compte de service Superset (partagé entre envs)
      - SUPERSET_PASSWORD  : mot de passe du compte de service
    """
    import requests as req

    superset_url = _superset_url(environnement_client)
    username = os.getenv("SUPERSET_USERNAME")
    password = os.getenv("SUPERSET_PASSWORD")
    if not username or not password:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Variables SUPERSET_USERNAME / SUPERSET_PASSWORD manquantes",
        )

    def _fetch_token() -> str:
        login = req.post(f"{superset_url}/api/v1/security/login", json={
            "username": username,
            "password": password,
            "provider": "db",
            "refresh": False,
        }, timeout=10)
        login.raise_for_status()
        access_token = login.json()["access_token"]

        guest = req.post(
            f"{superset_url}/api/v1/security/guest_token/",
            headers={"Authorization": f"Bearer {access_token}"},
            json={
                "user": {"username": "guest"},
                "resources": [{"type": "dashboard", "id": dashboard_id}],
                "rls": [],
            },
            timeout=10,
        )
        guest.raise_for_status()
        return guest.json()["token"]

    try:
        token = await asyncio.to_thread(_fetch_token)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Erreur Superset : {e}",
        )
    return {"token": token}


@app.get("/health")
async def health_check():
    """Vérification de l'état de l'API (pas d'authentification requise)"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    import copy
    import uvicorn.config
    log_config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    for formatter in log_config.get("formatters", {}).values():
        formatter["fmt"] = _LOG_FORMAT
        formatter["datefmt"] = _LOG_DATE_FORMAT
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=log_config,
    )
"""
Chargement incrémental de registration depuis la nouvelle gateway Webresto (data-lake).

Job forepass indépendant : ne dépend pas du package data_process / de l'API
IAnord. Objectif : pouvoir être déployé en un clic sur la data platform pour
basculer le chargement de registration sur la nouvelle gateway, sans attendre
la mise en production de ce repo.

La gateway a éclaté l'ancienne route en deux (/findAll/registrations et
/findAll/registrationForm), mais registrationForm reste toujours imbriqué
dans la réponse de /findAll/registrations (vérifié empiriquement, y compris
avec `selects` restreint) — on n'a donc besoin que d'une seule route pour
reconstituer la table `registration` telle qu'elle existe aujourd'hui.
La route registrationForm séparée n'apporte rien de plus : son filtre de
dates est en réalité celui de registration (pas de créé/modifié propre côté
webresto), donc pas d'intérêt à l'appeler indépendamment.

Autre différence avec l'ancienne API : la gateway ne renvoie plus l'objet
`session` imbriqué (donc plus de vagueId inline) — seulement `sessionId` à
plat. id_school_year se recalcule donc via sessionId → table session déjà
synchronisée (id_session→id_vague) → table vague (id_vague→id_school_year),
au lieu de dépendre d'un vagueId embarqué dans la réponse registration.

Logique :
  - SELECT MAX(updated_at) sur la table cible → point de reprise
  - Fetch par tranches de CHUNK_DAYS jours via GET /findAll/registrations
    (updatedSince / updatedBefore / selects)
  - Items sans sessionId : ignorés (log), même règle que l'ancienne API
  - Items avec deletedAt renseigné (soft delete côté webresto) :
      1. jamais insérés dans Trino
      2. purgés de Trino s'ils y étaient déjà
  - Filtrage centre : sessions/users de test ou anonymisés exclus (environnement "centre")
  - Items actifs : purge (même id) puis bulk_insert
  - Réconciliation hard-delete (filet de sécurité) : refetch systématique des
    RECONCILE_DAYS derniers jours, purge de ce que l'API ne renvoie plus.

PARAMS requis :
    BASE_URL              : host de la gateway (ex: "https://gateway.int.region-centre.ianord.fr"),
                            avec ou sans le suffixe "/data-lake" — les deux formes sont acceptées
    ENVIRONNEMENT_CLIENT  : ex "prodcentre"
    PREFIX_TABLE          : ex "wr_centre_"
    SECRET_KEY_WEBRESTO   : clé x-api-key
"""

import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from forepaas.core.settings import PARAMS
from forepaas.dwh import bulk_insert, connect

logger = logging.getLogger(__name__)

JOB_NAME = "registration"

WEBRESTO_BASE_URL    = PARAMS["BASE_URL"]
ENVIRONNEMENT_CLIENT = PARAMS["ENVIRONNEMENT_CLIENT"]
PREFIX_TABLE         = PARAMS["PREFIX_TABLE"]
SECRET_KEY           = PARAMS["SECRET_KEY_WEBRESTO"]

dataset_cible = f"dwh/db_mg6jk45h_{ENVIRONNEMENT_CLIENT}/"
p = PREFIX_TABLE
TABLE = f"{p}registration"

# La nouvelle gateway route sous /data-lake — accepte BASE_URL avec ou sans
# ce segment pour ne pas dépendre de la convention retenue au déploiement.
_BASE = WEBRESTO_BASE_URL.rstrip("/")
if _BASE.endswith("/data-lake"):
    _BASE = _BASE[: -len("/data-lake")]
_REGISTRATION_URL = f"{_BASE}/data-lake/findAll/registrations"

DEFAULT_START_DATE = "2022-08-01"
CHUNK_DAYS = 60
RECONCILE_DAYS = 7

SELECTS = "registrationId,status,sessionId,userId,createdAt,updatedAt"

_HEADERS = {"x-api-key": SECRET_KEY, "accept": "application/json"}
_RETRY_STATUSES = {502, 503, 504}
_RETRY_DELAYS = [5, 15, 30]

# ── Constantes métier (environnement "centre" uniquement) ─────────────────────

_SESSIONS_CENTRE_TO_DEL = [
    1, 3, 4, 5, 6, 7, 8, 66, 67, 68, 69, 70, 71, 72, 255, 256, 257, 258, 259, 260, 318
]

_USERS_CENTRE_DELETED = [
    804, 418, 979, 990, 876, 1283, 3160, 3227, 3229, 3373, 3400, 3468, 3520, 3633,
    3790, 3860, 4019, 8, 17547, 18003, 17680, 9620, 23910, 30936, 24005, 26944, 18275,
    11819, 31045, 24428, 31091, 27015, 11548, 26675, 17501, 19430, 17134, 10545, 26029,
    11856, 17732, 19494, 9645, 7215, 7256, 19126, 29463, 7205, 7332, 7441, 7360, 7172,
    15334, 32777, 7371, 7376, 10628, 19479, 26343, 26512, 7224, 7288, 7151, 26126,
    17777, 15609, 18048, 10222, 14260, 19200, 19420, 17562, 11048, 10036, 7269, 24864,
    10395, 9761, 11365, 17643, 18018, 33174, 17774, 26068, 6994, 19213, 24534, 29084,
    10519, 17495, 16859, 11305, 19293, 26138, 7176, 15711, 19219, 33408, 19418, 7131,
    24776, 15569, 19189, 19235, 29173, 20371, 19778, 20058, 33650, 29852, 26971, 33653,
    19567, 33688, 31411, 16713, 17629, 33734, 11258, 10797, 10818, 33744, 19197, 7322,
    11686, 16721, 9611, 11976, 9777, 33774, 33776, 17486, 16800, 7282, 26638, 14617,
    11050, 31226, 26664, 19557, 19816, 17402, 10431, 19554, 11273, 6857, 33803, 11292,
    7318, 12240, 24152, 10503, 15432, 7275, 24106, 15741, 33830, 26443, 26395, 7307,
    10243, 10988, 24508, 18001, 10566, 11955, 24858, 34082, 6724, 34635, 34686, 34801,
    24489, 11671, 17724, 24318, 34506, 11297, 10796, 16557, 24947, 7144, 12021, 19313,
    10267, 29641, 24383, 24697, 24210, 26569, 629, 7165, 25870, 24212, 11930, 11624,
    10250, 18115, 10891, 10790, 16584, 26657, 29623, 26903, 10497, 19534, 16638, 7310,
    19695, 19693, 15274, 10573, 10297, 35158, 19251, 10915, 24778, 30253, 29859, 29388,
    29054, 30176, 29789, 9729, 24634, 36017, 7225, 29134, 29434, 10256, 15853, 29466,
    7260, 9968, 7415, 26285, 29240, 24912, 29261, 7117, 7208, 26051, 29065, 7171,
    25864, 35848, 29634, 29104, 29372, 29806, 23926, 29726, 29730, 30084, 29235, 29568,
    7126, 7383, 29813, 29205, 24832, 24377, 29209, 9760, 17772, 7194, 7297, 29098,
    7265, 29786, 7328, 26991, 24780, 16179, 36435, 18897, 7451, 7344, 30191, 7422,
    7218, 24529, 17804, 29224, 7162, 24440, 30177, 24897, 26456, 29788, 30117, 29250,
    26380, 10534, 10543, 26151, 6728, 36584, 37405, 55342, 54660, 55599, 55255, 45957,
    49164, 41138, 54571, 55421, 55344, 41264, 40943, 49078, 39659, 55295, 62154, 62269,
    54951, 41651, 55481, 54737, 54682, 70827, 70828, 41775, 62512, 41553, 55426, 55239,
    7673, 55002, 41022, 63099, 55386, 55288, 62901, 41114, 49423, 46181, 41095, 63337,
    41332, 40851, 41411, 45399, 55235, 55544, 54898, 55203, 41492, 55362, 63041, 41604,
    55025, 55475, 41219, 40275, 55127, 55357, 54765, 55338, 54810, 41526, 55065, 49128,
    55613, 55190, 55931, 54876, 41772, 49360, 41484, 41035, 40918, 41113, 40956, 61720,
    55639, 55424, 55341, 41338, 55663, 40844, 40877, 49777, 55305, 55229, 49356, 54920,
    41033, 49403, 41700, 49821, 55959, 41884, 40958, 60433, 49042, 60104, 55412, 54857,
    55433, 55585, 55608, 49112, 55200, 55075, 76656, 60780, 41236, 49055, 49566, 60979,
    55286, 50411, 59139, 55657, 54806, 55456, 48981, 49570, 49490, 55490, 55160, 49549,
    55553, 74496, 55623, 54676, 60201, 49564, 7287, 54683, 55467, 55604, 28816, 103467,
    5182, 108707,
]


# ── Client gateway ───────────────────────────────────────────────────────────

def _fetch_api(updated_since: str, updated_before: str) -> list:
    params = {"updatedSince": updated_since, "updatedBefore": updated_before, "selects": SELECTS}
    last_error: Exception = RuntimeError("Aucune tentative effectuée")
    total_attempts = len(_RETRY_DELAYS) + 1

    for attempt, delay in enumerate([0] + _RETRY_DELAYS, start=1):
        if delay:
            logger.warning(f"[{JOB_NAME}] retry {attempt}/{total_attempts} dans {delay}s")
            time.sleep(delay)
        try:
            resp = requests.get(_REGISTRATION_URL, headers=_HEADERS, params=params, timeout=120)
            if resp.status_code in _RETRY_STATUSES:
                last_error = RuntimeError(f"HTTP {resp.status_code}")
                continue
            resp.raise_for_status()
            try:
                return resp.json() or []
            except ValueError as json_err:
                last_error = RuntimeError(
                    f"Réponse non-JSON (HTTP {resp.status_code}, "
                    f"body={resp.text[:200]!r}) : {json_err}"
                )
        except requests.exceptions.RequestException as e:
            last_error = e

    raise last_error


def _get_last_updated(source) -> str:
    try:
        result = source.query(f"SELECT MAX(updated_at) AS last_updated FROM {TABLE}")
        if isinstance(result, pd.DataFrame) and not result.empty:
            val = result.iloc[0]["last_updated"]
            if pd.notna(val):
                return val if isinstance(val, str) else val.strftime("%Y-%m-%d")
    except Exception as e:
        logger.warning(f"[{JOB_NAME}] impossible de lire MAX(updated_at) : {e}")
    return DEFAULT_START_DATE


def _load_maps(source) -> tuple:
    """id_school_year se recalcule via sessionId → session_map → vague_map
    (la gateway ne renvoie plus vagueId imbriqué dans registration)."""
    vague_df = source.query(f"SELECT id_vague, id_school_year FROM {p}vague")
    vague_map = (
        vague_df.set_index("id_vague")["id_school_year"].to_dict()
        if isinstance(vague_df, pd.DataFrame) and not vague_df.empty else {}
    )
    session_df = source.query(f"SELECT id, id_vague FROM {p}session")
    session_map = (
        session_df.set_index("id")["id_vague"].to_dict()
        if isinstance(session_df, pd.DataFrame) and not session_df.empty else {}
    )
    return vague_map, session_map


# ── Transform ──────────────────────────────────────────────────────────────

def _transform(items: list, vague_map: dict, session_map: dict) -> tuple[pd.DataFrame, list]:
    """Retourne (df_active, ids_to_purge)."""
    kept = []
    for item in items:
        if item.get("sessionId") is None:
            reg_id = item.get("id") or item.get("registrationId")
            logger.debug(f"[{JOB_NAME}] ignoré (session null) id={reg_id}")
            continue
        if "id" not in item and "registrationId" in item:
            item["id"] = item["registrationId"]
        rf = item.get("registrationForm") or {}
        item["rfChoiceBankDetail"] = rf.get("choiceBankDetail")
        item["rfTrancheId"]        = rf.get("trancheId")
        item["rfSubgroupId"]       = rf.get("subgroupId")
        kept.append(item)

    if not kept:
        return pd.DataFrame(), []

    df = pd.DataFrame(kept)
    ids_to_purge = df["id"].tolist()

    deleted_mask = df["deletedAt"].notna()
    n_deleted = int(deleted_mask.sum())
    if n_deleted:
        logger.info(f"[{JOB_NAME}] {n_deleted} soft-deleted détecté(s) — purge sans réinsertion")
    df = df[~deleted_mask].copy()

    if "centre" in ENVIRONNEMENT_CLIENT and not df.empty:
        df = df[~df["sessionId"].isin(_SESSIONS_CENTRE_TO_DEL)]
        df = df[~df["userId"].isin(_USERS_CENTRE_DELETED)]

    if df.empty:
        return df, ids_to_purge

    df["sessionId"] = df["sessionId"].astype("int64")
    df["createdAt"] = pd.to_datetime(df["createdAt"], format="mixed")
    df["updatedAt"] = pd.to_datetime(df["updatedAt"], format="mixed")

    cols = ["id", "status", "sessionId", "userId",
            "rfChoiceBankDetail", "rfTrancheId", "rfSubgroupId", "createdAt", "updatedAt"]
    df = df[cols]
    df = df.rename(columns={
        "sessionId": "id_session",
        "userId": "id_user",
        "rfChoiceBankDetail": "choice_bank_detail",
        "rfTrancheId": "tranche_id",
        "rfSubgroupId": "subgroup_id",
        "createdAt": "created_at",
        "updatedAt": "updated_at",
    })

    if vague_map and session_map:
        df["id_school_year"] = df["id_session"].map(session_map).map(vague_map)
    else:
        df["id_school_year"] = None

    df["status"] = df["status"].astype(str).str.strip()
    df["choice_bank_detail"] = df["choice_bank_detail"].apply(
        lambda x: str(x).strip() if pd.notna(x) else None
    )
    df["tranche_id"] = df["tranche_id"].apply(lambda x: int(x) if pd.notna(x) else None)

    return df, ids_to_purge


def _purge(source, ids: list) -> None:
    if not ids:
        return
    ids_sql = ",".join(str(int(i)) for i in ids)
    source.query(f"DELETE FROM {TABLE} WHERE id IN ({ids_sql})")


# ── Réconciliation hard-delete ───────────────────────────────────────────────

def _reconcile_hard_deletes(source, days: int = RECONCILE_DAYS) -> None:
    """
    Filet de sécurité : un hard delete côté webresto ne se voit ni via
    deletedAt ni via updatedAt, la ligne disparaît simplement de la réponse.
    On refetch la fenêtre des `days` derniers jours et on supprime de Trino
    les id de cette fenêtre que l'API ne renvoie plus.
    """
    reconcile_start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    reconcile_end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(f"[{JOB_NAME}] Réconciliation hard-delete : {reconcile_start} → {reconcile_end}")
    items = _fetch_api(reconcile_start, reconcile_end)
    if not items:
        logger.warning(f"[{JOB_NAME}] Réconciliation : réponse API vide, aucune suppression")
        return

    api_ids = {item.get("id") or item.get("registrationId") for item in items}

    result = source.query(
        f"SELECT id FROM {TABLE} "
        f"WHERE updated_at >= TIMESTAMP '{reconcile_start} 00:00:00'"
    )
    if not isinstance(result, pd.DataFrame) or result.empty:
        logger.info(f"[{JOB_NAME}] Réconciliation : aucune ligne Trino sur la fenêtre")
        return

    trino_ids = set(result["id"].dropna().astype(int).tolist())
    stale_ids = trino_ids - api_ids
    if not stale_ids:
        logger.info(f"[{JOB_NAME}] Réconciliation : aucune ligne obsolète")
        return

    logger.info(
        f"[{JOB_NAME}] Réconciliation : {len(stale_ids)} ligne(s) absente(s) de l'API "
        f"(hard delete) — suppression"
    )
    _purge(source, list(stale_ids))


# ── Entry point ────────────────────────────────────────────────────────────

def customfunc(event):
    logger.info(f"Démarrage du job '{JOB_NAME}'")
    t0 = time.time()

    source = connect(dataset_cible)
    vague_map, session_map = _load_maps(source)

    last_updated = _get_last_updated(source)
    updated_before_end = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    current_start = datetime.strptime(last_updated, "%Y-%m-%d")
    end_date = datetime.strptime(updated_before_end, "%Y-%m-%d")

    total = 0
    try:
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=CHUNK_DAYS), end_date)
            s = current_start.strftime("%Y-%m-%d")
            e = current_end.strftime("%Y-%m-%d")
            logger.info(f"[{JOB_NAME}] Période {s} → {e}")

            items = _fetch_api(s, e)
            if items:
                logger.info(f"[{JOB_NAME}] {len(items)} lignes brutes récupérées")
                df, ids_to_purge = _transform(items, vague_map, session_map)
                _purge(source, ids_to_purge)
                if not df.empty:
                    bulk_insert(source, TABLE, df)
                    total += len(df)
                    logger.info(f"[{JOB_NAME}] {len(df)} lignes chargées ({s} → {e})")
            else:
                logger.info(f"[{JOB_NAME}] Aucune donnée pour cette période")

            current_start = current_end

        duration = round(time.time() - t0, 2)
        logger.info(f"[{JOB_NAME}] OK — {total} lignes chargées — {duration}s")
    except Exception as e:
        logger.error(f"[{JOB_NAME}] Erreur fatale : {type(e).__name__}: {e}")
        raise

    try:
        _reconcile_hard_deletes(source)
    except Exception as e:
        logger.error(f"[{JOB_NAME}] Réconciliation hard-delete échouée : {type(e).__name__}: {e}")

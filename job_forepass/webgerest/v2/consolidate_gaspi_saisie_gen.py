"""
Consolidation des tables gaspi_saisie_gen V2 vers wg_test_centre_gaspi_saisie_gen (schéma V1).

Stratégie : DROP TABLE + CREATE TABLE + un seul INSERT INTO … SELECT … UNION ALL
→ exactement 1 snapshot Iceberg, quel que soit le nombre de tables source.

PARAMS requis :
    BASE_URL, ENVIRONNEMENT_CLIENT, PREFIX_TABLE,
    CLIENT_WEBGEREST, SECRET_KEY_WEBGEREST, WEBGEREST_LOGIN_GROUPS
"""

import json
import logging
import math
import re
import unicodedata

import pandas as pd
from forepaas.core.settings import PARAMS
from forepaas.dwh import connect

logger = logging.getLogger(__name__)

# ── Paramètres ─────────────────────────────────────────────────────────────────

DATASET_CIBLE  = f"dwh/{PARAMS['ENVIRONNEMENT_CLIENT']}/"
SERVER_PREFIX  = PARAMS["PREFIX_TABLE"]
LOGIN_GROUPS   = json.loads(PARAMS["WEBGEREST_LOGIN_GROUPS"])
TARGET_TABLE   = "wg_test_centre_gaspi_saisie_gen"

CHUNK_SIZE = None

# ── DDL V1 cible ───────────────────────────────────────────────────────────────

_DDL = f"""
CREATE TABLE {TARGET_TABLE} (
    pk VARCHAR,
    login_site VARCHAR,
    id_gaspi_saisie_gen BIGINT,
    datej DATE,
    codss1 VARCHAR,
    codss2 VARCHAR,
    qte_gen DOUBLE,
    cle_jps VARCHAR,
    eff_prev BIGINT,
    pas_de_tri BOOLEAN,
    commentaire VARCHAR,
    eff_prod BIGINT,
    eff_reel_service BIGINT,
    descfic_statut BIGINT
)
WITH (
    partitioning = ARRAY['login_site'],
    extra_properties = MAP(
        ARRAY[
            'write.target-file-size-bytes',
            'write.metadata.delete-after-commit.enabled',
            'write.metadata.previous-versions-max'
        ],
        ARRAY['268435456', 'true', '50']
    )
)
""".strip()

_V2_COLS = (
    "id_gaspi_saisie_gen, datej, codss1, codss2, qte_gen, cle_jps, "
    "eff_prev, pas_de_tri, commentaire, eff_prod, eff_reel_service"
)

# ── Utilitaires ────────────────────────────────────────────────────────────────

def _safe_id(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).lower().strip("_")


def _sub_select(identifier: str, source_table: str, statut: int) -> str:
    return (
        f"SELECT\n"
        f"    CAST('{identifier}' || '_' || CAST(id_gaspi_saisie_gen AS VARCHAR) AS VARCHAR) AS pk,\n"
        f"    '{identifier}' AS login_site,\n"
        f"    {_V2_COLS},\n"
        f"    CAST({statut} AS BIGINT) AS descfic_statut\n"
        f"FROM {source_table}"
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def customfunc(event):
    logger.info("job : consolidate_gaspi_saisie_gen V2 → V1")
    source = connect(DATASET_CIBLE)
    login_table = f"{SERVER_PREFIX}login"

    # 1. login_map depuis centre_login
    df_login = source.select(login_table)
    df_login = df_login[df_login["profil"] == 2]
    if "fictif" in df_login.columns:
        df_login = df_login[df_login["fictif"] != True]
    if "nometabs" in df_login.columns:
        df_login = df_login[~df_login["nometabs"].str.upper().str.contains("DEMO]", na=False)]

    login_map: dict[str, list[str]] = {}
    for _, row in df_login.iterrows():
        login_map.setdefault(row["logingroupe"], []).append(row["login"])

    # 2. statut gaspi_saisie_gen par groupe
    statut_by_group: dict[str, int] = {}
    for grp in LOGIN_GROUPS:
        descfic_table = f"{SERVER_PREFIX}{_safe_id(grp)}_descfic"
        try:
            df_d = source.query(
                f"SELECT statut FROM {descfic_table} WHERE UPPER(nomfic) = 'GASPI_SAISIE_GEN'"
            )
            if isinstance(df_d, pd.DataFrame) and not df_d.empty:
                statut_by_group[grp] = int(df_d.iloc[0]["statut"])
        except Exception as e:
            logger.warning(f"  [{grp}] impossible de lire descfic gaspi_saisie_gen: {e}")

    # 3. Construire la liste (identifier, source_table, statut)
    entries: list[tuple[str, str, int]] = []
    for grp in LOGIN_GROUPS:
        statut = statut_by_group.get(grp)
        if statut is None:
            logger.warning(f"[{grp}] pas d'entrée descfic pour gaspi_saisie_gen, ignoré")
            continue
        if statut == 2:
            for site in login_map.get(grp, []):
                entries.append((site, f"{SERVER_PREFIX}{_safe_id(site)}_gaspi_saisie_gen", statut))
        else:
            entries.append((grp, f"{SERVER_PREFIX}{_safe_id(grp)}_gaspi_saisie_gen", statut))

    if not entries:
        logger.error("Aucune table source trouvée — abandon")
        return

    logger.info(f"{len(entries)} table(s) source à consolider")

    # 4. DROP TABLE + CREATE TABLE
    logger.info(f"DROP TABLE IF EXISTS {TARGET_TABLE}")
    source.query(f"DROP TABLE IF EXISTS {TARGET_TABLE}")
    logger.info(f"CREATE TABLE {TARGET_TABLE}")
    source.query(_DDL)

    # 5. INSERT INTO … SELECT … UNION ALL
    chunk_size = CHUNK_SIZE or len(entries)
    n_chunks = math.ceil(len(entries) / chunk_size)
    logger.info(
        f"INSERT en {n_chunks} requête(s) "
        f"({chunk_size} table(s)/requête → {n_chunks} snapshot(s))"
    )

    for i in range(n_chunks):
        chunk = entries[i * chunk_size : (i + 1) * chunk_size]
        union_sql = "\nUNION ALL\n".join(
            _sub_select(identifier, src_table, statut)
            for identifier, src_table, statut in chunk
        )
        sql = f"INSERT INTO {TARGET_TABLE}\n{union_sql}"
        logger.info(f"  Chunk {i+1}/{n_chunks} — {len(chunk)} table(s)...")
        source.query(sql)
        logger.info(f"  Chunk {i+1}/{n_chunks} OK")

    logger.info(f"consolidate_gaspi_saisie_gen terminé — {n_chunks} snapshot(s) créé(s)")

import logging
import time
from datetime import datetime

from forepaas.dwh import connect
from forepaas.core.settings import PARAMS

logger = logging.getLogger(__name__)

JOB_NAME = "user_info_school_year"

prefix_table = PARAMS["PREFIX_TABLE"]
environnement_client = PARAMS["ENVIRONNEMENT_CLIENT"]
dataset_cible = f"dwh/db_mg6jk45h_{environnement_client}/"

p = prefix_table

# Année scolaire courante — à mettre à jour manuellement lors d'une transition
CURRENT_SCHOOL_YEAR_ID = 3


def _check_school_year_transition(source, current_sy_id: int):
    df = source.query(f"SELECT MAX(id_school_year) AS max_sy FROM {p}vague")
    if df is None or df.empty or df["max_sy"].iloc[0] is None:
        return None
    max_id = int(df["max_sy"].iloc[0])
    return max_id if max_id > current_sy_id else None


def _backup_user_info_schoolyear(source, school_year_id: int) -> None:
    env_suffix = "dep93" if "93" in environnement_client else "centre"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_table = (
        f"db_mg6jk45h_backup.backup"
        f".{env_suffix}_user_info_schoolyear_sy{school_year_id}_{ts}"
    )
    source.query(
        f"CREATE TABLE {backup_table} AS"
        f" SELECT * FROM {p}user_info_school_year"
        f" WHERE school_year_id = {school_year_id}"
    )
    logger.info(f"[{JOB_NAME}] Backup sy={school_year_id} → {backup_table}")


def customfunc(event):
    logger.info(f"Démarrage du job '{JOB_NAME}'")
    t0 = time.time()

    source = connect(dataset_cible)

    current_sy_id = CURRENT_SCHOOL_YEAR_ID

    new_sy_id = _check_school_year_transition(source, current_sy_id)
    if new_sy_id:
        _backup_user_info_schoolyear(source, current_sy_id)
        logger.warning(
            f"[{JOB_NAME}] Transition détectée : school_year {current_sy_id} → {new_sy_id}."
            f" Mettre à jour CURRENT_SCHOOL_YEAR_ID dans ce fichier."
        )
        current_sy_id = new_sy_id

    sql = f"""
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
            LEFT JOIN {p}subgroup  sg  ON u.id_subgroup  = sg.id_subgroup
            LEFT JOIN {p}group     grp ON sg.id_group    = grp.id_group
            LEFT JOIN {p}tranche   t   ON b.id_tranche   = t.id_tranche
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
    """

    try:
        source.query(sql)
        duration = round(time.time() - t0, 2)
        logger.info(f"[{JOB_NAME}] OK — {duration}s")
    except Exception as e:
        logger.error(f"[{JOB_NAME}] Erreur fatale : {type(e).__name__}: {e}")
        raise

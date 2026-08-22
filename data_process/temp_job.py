"""
Jobs temporaires à exécuter une seule fois.
À supprimer une fois exécutés.
"""

# ── Backfill user first_name / last_name / date_birth depuis CSV ──────────────

def run_user_identity_csv_job(
    environnement_client: str,
    prefix_table: str,
    ovh_api_key: str,
    ovh_secret_key: str,
) -> "JobResult":
    """
    Backfill des colonnes first_name, last_name, date_birth sur la table user
    à partir des CSV contactDetail et userContactDetail (lien link='me').

    À exécuter une seule fois en attente des routes API dédiées.
    """
    import time
    from data_process.jobs import JobResult

    result = JobResult()
    t0 = time.time()
    table = f"{prefix_table}user"

    if "centre" in environnement_client:
        base = "data_process/temp_data/centre"
    elif "93" in environnement_client:
        base = "data_process/temp_data/93"
    else:
        result.errors.append(f"environnement_client non reconnu : {environnement_client!r}")
        result.status = "failed"
        result.duration_seconds = 0.0
        return result

    try:
        db = TrinoClient(environnement_client, ovh_api_key, ovh_secret_key)

        print("[user_identity_csv] Lecture des CSV...")
        ucd = pd.read_csv(f"{base}/userContactDetail.csv", low_memory=False)
        cd  = pd.read_csv(f"{base}/contactDetail.csv",     low_memory=False)
        sd  = pd.read_csv(f"{base}/studentdetail.csv",     low_memory=False)
        print(f"[user_identity_csv] userContactDetail : {len(ucd)} | contactDetail : {len(cd)} | studentdetail : {len(sd)}")

        # Garder uniquement les liens actifs link='me'
        ucd = ucd[ucd["deletedAt"].isna() & (ucd["link"] == "me")][["userId", "contactDetailId"]]
        cd  = cd[cd["deletedAt"].isna()][["contactDetailId", "firstname", "lastname", "dateBirth"]]
        # id_organization depuis registerToOrganization — un userId peut avoir plusieurs lignes,
        # on garde la plus récente (updatedAt max)
        sd = sd[sd["deletedAt"].isna()][["userId", "registerToOrganization", "updatedAt"]]
        sd = sd.sort_values("updatedAt").drop_duplicates("userId", keep="last")
        sd = sd[["userId", "registerToOrganization"]]
        print(f"[user_identity_csv] Après filtrage : {len(ucd)} liens 'me' | {len(cd)} contacts | {len(sd)} org mappings")

        df = ucd.merge(cd, on="contactDetailId", how="inner")
        df = df.merge(sd, on="userId", how="left")
        df["date_birth"] = pd.to_datetime(df["dateBirth"], errors="coerce").dt.date
        df = df.rename(columns={
            "userId":                 "id_user",
            "firstname":              "first_name",
            "lastname":               "last_name",
            "registerToOrganization": "id_organization",
        })
        df = df[["id_user", "first_name", "last_name", "date_birth", "id_organization"]].drop_duplicates("id_user")
        df["id_user"] = df["id_user"].astype(float)
        df["id_organization"] = pd.to_numeric(df["id_organization"], errors="coerce")

        total = len(df)
        print(f"[user_identity_csv] {total} users à backfiller")

        staging = f"{prefix_table}user_identity_stg"
        db.run_query(f"DROP TABLE IF EXISTS {staging}")
        db.run_query(f"""
            CREATE TABLE {staging} (
                id_user         DOUBLE,
                first_name      VARCHAR,
                last_name       VARCHAR,
                date_birth      DATE,
                id_organization DOUBLE
            )
        """)
        print(f"[user_identity_csv] Table staging '{staging}' créée")

        rows_inserted = db.bulk_insert(staging, df)
        print(f"[user_identity_csv] {rows_inserted} lignes insérées dans le staging")

        print(f"[user_identity_csv] MERGE en cours vers {table}...")
        db.run_query(f"""
            MERGE INTO {table} t
            USING {staging} s ON t.id_user = s.id_user
            WHEN MATCHED THEN UPDATE SET
                first_name      = s.first_name,
                last_name       = s.last_name,
                date_birth      = s.date_birth,
                id_organization = s.id_organization
        """)
        db.run_query(f"DROP TABLE {staging}")

        result.rows_upserted = rows_inserted
        result.success = True
        result.status = "complete_success"
        print(f"[user_identity_csv] OK — {rows_inserted} lignes mises à jour dans {table}")

    except Exception as e:
        logger.exception("[user_identity_csv] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"

    result.duration_seconds = round(time.time() - t0, 2)
    return result

import logging
import os
import time
import pandas as pd

from data_process.db.trino_client import TrinoClient
from data_process.jobs import JobResult
from data_process.process.registrations import transform_registration

logger = logging.getLogger(__name__)


# ── Chargement initial user_info_school_year ──────────────────────────────────

def run_user_info_school_year_initial_job(
    environnement_client: str,
    prefix_table: str,
    ovh_api_key: str,
    ovh_secret_key: str,
) -> JobResult:
    """
    Chargement initial de user_info_school_year pour school_year_id 1 et 2.

    Lit les tables user et bankdetail depuis Trino, effectue une jointure gauche
    sur id_user, puis duplique chaque ligne pour school_year_id=1 et school_year_id=2.
    La PK est int(str(school_year_id) + str(int(id_user))).

    À exécuter une seule fois avant d'activer le job incrémental.
    """
    result = JobResult()
    t0 = time.time()
    p = prefix_table
    table = f"{p}user_info_school_year"

    try:
        from data_process.process.schemas_webresto import WEBRESTO_SCHEMAS

        db = TrinoClient(environnement_client, ovh_api_key, ovh_secret_key)

        ddl = WEBRESTO_SCHEMAS["user_info_school_year"].to_ddl(table)
        db.run_query(ddl)
        logger.info(f"[user_info_school_year_initial] Table {table} créée (ou déjà existante)")

        df_user = db.query_as_dataframe(
            f"SELECT id_user, id_subgroup, created_at, updated_at FROM {p}user"
        )
        df_bd = db.query_as_dataframe(
            f"SELECT bank_detail_id, id_user, choice_bank_details, id_tranche, updated_at "
            f"FROM {p}bankdetail"
        )
        df_sg = db.query_as_dataframe(
            f"SELECT id_subgroup, label AS label_subgroup, id_group FROM {p}subgroup"
        )
        df_grp = db.query_as_dataframe(
            f"SELECT id_group, label AS label_group FROM {p}group"
        )
        df_tr = db.query_as_dataframe(
            f"SELECT id_tranche, label AS label_tranche FROM {p}tranche"
        )

        if df_user.empty:
            result.errors.append("Table user vide — impossible de charger user_info_school_year")
            result.status = "failed"
            result.duration_seconds = round(time.time() - t0, 2)
            return result

        df = df_user.merge(df_bd, on="id_user", how="left", suffixes=("", "_bd"))

        # updated_at = max(user.updated_at, bankdetail.updated_at)
        df["updated_at"] = df[["updated_at", "updated_at_bd"]].apply(
            lambda row: max(
                (v for v in row if pd.notna(v)),
                default=row["updated_at"],
            ),
            axis=1,
        )
        df.drop(columns=["updated_at_bd"], inplace=True)

        # Jointures pour les libellés
        df_sg_grp = df_sg.merge(df_grp, on="id_group", how="left")
        df = df.merge(df_sg_grp[["id_subgroup", "label_subgroup", "label_group"]], on="id_subgroup", how="left")
        df = df.merge(df_tr[["id_tranche", "label_tranche"]], on="id_tranche", how="left")

        dfs = []
        for sy_id in (1, 2):
            chunk = df.copy()
            chunk["school_year_id"] = sy_id
            chunk["id"] = chunk["id_user"].apply(
                lambda u: int(str(sy_id) + str(int(u)))
            )
            dfs.append(chunk)

        df_final = pd.concat(dfs, ignore_index=True)
        cols = ["id", "school_year_id", "id_user", "id_subgroup",
                "bank_detail_id", "choice_bank_details", "id_tranche",
                "label_tranche", "label_subgroup", "label_group",
                "created_at", "updated_at"]
        df_final = df_final[cols]

        rows = db.bulk_insert(table, df_final)
        result.rows_upserted = rows
        result.success = True
        result.status = "complete_success"
        logger.info(
            f"[user_info_school_year_initial] OK — {rows} lignes upsertées "
            f"({len(df_user)} users × 2 school years)"
        )

    except Exception as e:
        logger.exception("[user_info_school_year_initial] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"

    result.duration_seconds = round(time.time() - t0, 2)
    return result


# ── Chargement initial registration depuis CSV ────────────────────────────────

def run_registration_csv_job(
    environnement_client: str,
    prefix_table: str,
    ovh_api_key: str,
    ovh_secret_key: str,
) -> JobResult:
    """
    Chargement initial de registration depuis CSV.
    Stratégie : TRUNCATE puis bulk_insert.
    CSV source : data_process/temp_data/{93|centre}/registration.csv
    """
    result = JobResult()
    t0 = time.time()
    table = f"{prefix_table}registration"

    if "93" in environnement_client:
        csv_path = "data_process/temp_data/93/registration.csv"
    elif "centre" in environnement_client:
        csv_path = "data_process/temp_data/centre/registration.csv"
    else:
        result.errors.append(f"environnement_client non reconnu : {environnement_client!r}")
        result.status = "failed"
        result.duration_seconds = 0.0
        return result

    try:
        db = TrinoClient(environnement_client, ovh_api_key, ovh_secret_key)

        df = pd.read_csv(csv_path, low_memory=False)
        # Exclure les soft-deleted
        df = df[df["deletedAt"].isna()].copy()
        # Alignement avec le format attendu par transform_registration
        df = df.rename(columns={"registrationId": "id"})
        for col in ("rfChoiceBankDetail", "rfTrancheId", "rfSubgroupId"):
            if col not in df.columns:
                df[col] = None

        df = transform_registration(df, environnement_client)

        if df.empty:
            result.errors.append("CSV vide après transform")
            result.status = "failed"
        else:
            db.truncate(table)
            rows = db.bulk_insert(table, df)
            result.rows_upserted = rows
            result.success = True
            result.status = "complete_success"
            logger.info(
                f"[registration_csv] OK — {rows} lignes insérées depuis {csv_path}"
            )

    except Exception as e:
        logger.exception("[registration_csv] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"

    result.duration_seconds = round(time.time() - t0, 2)
    return result


# ── Chargement initial organization_enrollment depuis CSV ─────────────────────

def run_organization_enrollment_csv_job(
    environnement_client: str,
    prefix_table: str,
    ovh_api_key: str,
    ovh_secret_key: str,
) -> JobResult:
    """
    Chargement initial de organization_enrollment depuis CSV.
    Stratégie : TRUNCATE puis bulk_insert.
    CSV source : data_process/temp_data/{93|centre}/organizationEnrollment.csv

    À exécuter une seule fois en attente de la route API dédiée.
    """
    result = JobResult()
    t0 = time.time()
    table = f"{prefix_table}organization_enrollment"

    if "93" in environnement_client:
        csv_path = "data_process/temp_data/93/organizationEnrollment.csv"
    elif "centre" in environnement_client:
        csv_path = "data_process/temp_data/centre/organizationEnrollment.csv"
    else:
        result.errors.append(f"environnement_client non reconnu : {environnement_client!r}")
        result.status = "failed"
        result.duration_seconds = 0.0
        return result

    try:
        db = TrinoClient(environnement_client, ovh_api_key, ovh_secret_key)

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
            logger.info(
                f"[organization_enrollment_csv] OK — {rows} lignes insérées depuis {csv_path}"
            )

    except Exception as e:
        logger.exception("[organization_enrollment_csv] Erreur fatale")
        result.errors.append(str(e))
        result.status = "failed"

    result.duration_seconds = round(time.time() - t0, 2)
    return result

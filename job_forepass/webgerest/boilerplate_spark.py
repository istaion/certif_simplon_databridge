"""
BOILERPLATE: Data Platform Custom PySpark Action
================================================
A minimal, safe, and reusable template for Spark jobs in DPE.

Features:
- Proper logging with aligned sample output
- Lakehouse connection via `forepaas.dwh.connect`
- Configurable dataset/table via main()
- Optional table rebuild (drop + bulk_insert)
- Graceful Spark session handling
- Local testing via `__main__`

Docs:
- https://docs.dataplatform.ovh.net/#/en/product/dpe/actions/custom-pyspark/
- NYC Taxi end-to-end: https://docs.dataplatform.ovh.net/#/en/getting-further/pyspark/index
"""

# --------------------------------------------------------------------- #
# 0. Imports & Logging
# --------------------------------------------------------------------- #
import logging
from logging import getLogger

from forepaas.core.settings import CONFIG
from forepaas.dwh import bulk_insert, connect
from pyspark import StorageLevel
from pyspark.sql import SparkSession

logger = getLogger(__name__)


# --------------------------------------------------------------------- #
# 1. Helpers
# --------------------------------------------------------------------- #
def log_sample(df, n: int = 10, w: int = 24) -> None:
    """Pretty-print a small sample to logs with padded columns."""
    cols = df.columns
    fmt = " | ".join(f"{{:{w}}}" for _ in cols)
    logger.info("Sample rows:")
    logger.info(fmt.format(*cols))
    for r in df.limit(n).toLocalIterator():
        logger.info(fmt.format(*(("" if r[c] is None else str(r[c]))[:w] for c in cols)))
    logger.info("-" * max(60, len(fmt)))


# --------------------------------------------------------------------- #
# 2. Main Job
# --------------------------------------------------------------------- #
def run_spark_job(dataset: str, table: str, rebuild_table: bool = False) -> None:
    """
    Connect to Lakehouse, select table, log stats + sample, optionally rebuild the table.
    Uses cache + localCheckpoint (materialized) to take an in-session snapshot before delete.
    """
    logger.info("START - run_spark_job")

    # ----------------------------------------------------------------- #
    # 2.1 Spark Session
    # ----------------------------------------------------------------- #
    spark = SparkSession.builder.appName("Custom_PySpark_Action").getOrCreate()
    logger.info(f"Spark Version: {spark.version}")

    # ----------------------------------------------------------------- #
    # 2.2 Connect & Read
    # ----------------------------------------------------------------- #
    dataplant_id = CONFIG.get("dataplant_id")
    cn = connect(f"dwh/{dataset}/")
    db_table = f"db_{dataplant_id}_{dataset}.{dataset}.{table}"

    df = cn.select(db_table)
    row_count = df.count()
    logger.info(f"Table '{table}': {row_count} rows")

    # ----------------------------------------------------------------- #
    # 2.3 Your Transformations Go Here!
    # ----------------------------------------------------------------- #
    # Example:
    # df = df.filter("station_name = 'Cicero-Lake'")
    # logger.info(f"Only Cicero-Lake: {df.count()} rows")

    # ----------------------------------------------------------------- #
    # 2.4 Show Sample
    # ----------------------------------------------------------------- #
    log_sample(df, n=10, w=24)

    # ----------------------------------------------------------------- #
    # 2.5 Optional: Drop & Rebuild Table from Current DF
    # ----------------------------------------------------------------- #
    if rebuild_table:
        # Make a session-scoped snapshot independent of the external source.
        df_cached = df.persist(StorageLevel.MEMORY_AND_DISK)
        _ = df_cached.count()

        df_snap = df_cached.localCheckpoint(eager=True)
        df_cached.unpersist()

        logger.info(f"Dropping & rebuilding '{table}' in dataset '{dataset}'")
        cn.delete(table, {})  # drop existing table
        bulk_insert(cn, table, df_snap)  # reinsert snapshot

        # Sanity check
        reloaded = cn.select(db_table)
        logger.info(f"Table '{table}' - reinserted rows: {reloaded.count()}")

    # ----------------------------------------------------------------- #
    # 2.6 Cleanup
    # ----------------------------------------------------------------- #
    del cn
    logger.info("END - run_spark_job")


# --------------------------------------------------------------------- #
# 3. Entry Point
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    # Edit here only: keep the function reusable.
    DATASET = "default_dataset"
    TABLE = "stations_rides"
    REBUILD = False  # set to False unless you really want to drop+rebuild

    run_spark_job(dataset=DATASET, table=TABLE, rebuild_table=REBUILD)

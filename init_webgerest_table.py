from data_process.jobs import run_create_webgerest_tables_job
from dotenv import load_dotenv

load_dotenv()

import os

dataset="prod13"
prefix="wg_13_"
ovh_api_key=os.getenv("OVH_API_KEY")
ovh_secret_key=os.getenv("OVH_SECRET_KEY")

if __name__ == "__main__":
    run_create_webgerest_tables_job(dataset,prefix,ovh_api_key,ovh_secret_key)
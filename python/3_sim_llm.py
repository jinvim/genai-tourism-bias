# script to generate travel suggestions using llm
# this script has somewhat basic error handling, quota management, and progress tracking
# %%
import pandas as pd
import geopandas as gpd
import numpy as np
import sys
import instructor
import pytz
import os
import logging
import time
from datetime import datetime, timedelta
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import List
from pydantic import BaseModel


PROMPT_PATH = "prompt.md"
SHP_PATH = "../data/shp/tl_2023_us_state/tl_2023_us_state.shp"
DATA_PATH = "../data/sample.parquet"
OUTPUT_PATH = "../data/sim-llm-raw"
GOOGLE_API_KEY = "AIzaSyCjZ_1GjWqbTUvOgCxJr072U20GEL4m0A8"
DAILY_REQUEST_LIMIT = 1_000  # maximum requests per day
BATCH_SIZE = 100  # number of users to process in each batch
TIMEZONE = "US/Pacific"  # timezone for the daily reset

LOG = logging.getLogger(__name__)
# MODEL = "google/gemini-2.0-flash-lite"
# MODEL = "google/gemini-2.5-flash"
# MODEL = "google/gemini-2.5-flash-preview-04-17"
MODEL = "google/gemini-2.5-flash-lite-preview-06-17"

OUTPUT_DIR = OUTPUT_PATH + "/" + MODEL.split("/")[-1]

# %%
class Suggestion(BaseModel):
    userid: int
    location: str
    state: str
    rationale: str
    recommended_month: int
    duration_days: int
    total_budget_usd: int
    transportation_budget_usd: int
    accommodation_budget_usd: int
    fnb_budget_usd: int
    activities_budget_usd: int
    travel_distance_miles: int
    transportation_mode: str


class Suggestions(BaseModel):
    response: List[Suggestion]


# %%
# read instruction prompt
def read_instruction(batch_size):
    with open(PROMPT_PATH, "r") as f:
        instruction = f.read()
        # Replace the placeholder with the actual batch size
        return instruction.format(batch_size=batch_size)


def read_data():
    state_dict = get_state_dict()
    df = pd.read_parquet(DATA_PATH)
    df["state_name"] = df["state"].map(state_dict)
    return df


def get_state_dict():
    states = gpd.read_file(SHP_PATH)
    states["GEOID"] = states["GEOID"].astype(int)
    return states.set_index("GEOID")["NAME"].to_dict()


# %%
def gen_input_str(id, state, sex, agegrp, incgrp):
    return f"USERID:{id} – {agegrp} {sex} from {state} with annual household income of {incgrp}"


def gen_input(sub_df):
    rows = sub_df.apply(
        lambda x: gen_input_str(
            x["sampleid"], x["state_name"], x["sex"], x["agegrp"], x["incgrp"]
        ),
        axis=1,
    )
    return "\n".join(rows)


# %%
def send_request(model, instruction, content):
    client = instructor.from_provider(model)

    output, completion = client.chat.completions.create_with_completion(
        response_model=Suggestions,
        messages=[
            {
                "role": "system",
                "content": instruction,
            },
            {
                "role": "user",
                "content": content,
            },
        ],
        # disables thinking
        generation_config={"thinkingConfig": {"thinkingBudget": 0}},
    )
    return output, completion

# %%
def get_suggestions(sub_df, i, filename, instruction, qm, check_len=True):
    rows = gen_input(sub_df)
    # check if file already exists
    if os.path.exists(filename):
        LOG.info(f"file {filename} already exists. skipping.")
        return

    retry_counter = 0
    while True:
        try:
            output, completion = send_request(MODEL, instruction, rows)
            qm.add_request_count()
            res_df = pd.DataFrame([o.model_dump() for o in output.response])
            LOG.info(f"input token: {completion.usage_metadata.prompt_token_count}, output token: {completion.usage_metadata.candidates_token_count}")
            n_missing = BATCH_SIZE - res_df.shape[0]
            if n_missing != 0 & check_len:
                qm.missing_count += 1
                LOG.warning(
                    f"missing {n_missing} responses for {filename} retrying... (retry count: {qm.missing_count})"
                )
                continue
            res_df["iter"] = i
            res_df.to_parquet(filename, index=False)
            retry_counter = 0
            break

        except KeyboardInterrupt:
            LOG.info("Script interrupted by user")
            sys.exit(0)

        except Exception as e:
            LOG.error(f"encountered error: {e}")
            if "503" in str(e):
                LOG.info("waiting 2 minutes due to 503 error")
                time.sleep(60*2)
            elif "429" in str(e):
                LOG.info("waiting until midnight due exceeding quota")
                qm.sleep_until_reset()
                retry_counter = 0
            else:
                if retry_counter < 3:
                    LOG.info("waiting 5 minutes due to unspecified error")
                    time.sleep(60*5)
                    retry_counter += 1
                else:
                    LOG.info("3 retries failed. waiting until midnight due to unspecified error")
                    qm.sleep_until_reset()
                    retry_counter = 0
            continue

# %%
def get_missings(df):
    results_df = pd.read_parquet(OUTPUT_DIR).rename(columns={"userid": "sampleid"})
    results_df = results_df.drop_duplicates(subset=["iter", "sampleid"])
    merged = pd.merge(df, results_df, on=["iter", "sampleid"], how="left").rename(columns={"state_x": "state"})
    merged = merged[merged["location"].isna()]
    return merged[df.columns]


# %%
def handle_missing(df, instruction, qm):
    # check if there are any missing responses
    missing_df = get_missings(df)
    while not missing_df.empty:
        LOG.warning(f"Found {missing_df.shape[0]} missing responses. Reprocessing...")

        # reprocess missing data
        for i in tqdm(missing_df["iter"].unique(), desc="reprocessing iterations", position=0):
            iter_df = missing_df[missing_df["iter"] == i]

            # process all if iter_df is small enough
            if iter_df.shape[0] < BATCH_SIZE:
                id_begin = iter_df["sampleid"].min()
                id_end = iter_df["sampleid"].max()
                filename = f"{OUTPUT_DIR}/nan-iter{i:03d}-{id_begin:03d}-{id_end:03d}.parquet"
                get_suggestions(iter_df, i, filename, instruction, qm, check_len=False)
            else:
                # cut iter_df by batch size
                n_sections = len(iter_df) // BATCH_SIZE
                for sub_idx in tqdm(np.array_split(iter_df.index, n_sections), desc=f"batch", position=1):
                    sub_df = iter_df.loc[sub_idx]
                    id_begin = sub_df["sampleid"].min()
                    id_end = sub_df["sampleid"].max()
                    filename = f"{OUTPUT_DIR}/nan-iter{i:03d}-{id_begin:03d}-{id_end:03d}.parquet"

                    get_suggestions(sub_df, i, filename, instruction, qm, check_len=False)
        missing_df = get_missings(df)

# %%
class QuotaManager:
    def __init__(self):
        self.timezone = pytz.timezone(TIMEZONE)
        self.current_requests = 0
        self.last_reset_date = self.get_current_date()
        self.missing_count = 0

    def get_current_time(self) -> datetime:
        return datetime.now(self.timezone)

    def get_current_date(self) -> str:
        return self.get_current_time().strftime("%Y-%m-%d")

    def get_next_midnight(self) -> datetime:
        now = self.get_current_time()
        next_midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return next_midnight

    def seconds_until_midnight(self) -> int:
        now = self.get_current_time()
        next_midnight = self.get_next_midnight()
        return int((next_midnight - now).total_seconds())

    def check_quota_reset(self):
        """check if we need to reset the quota (new day)"""
        current_date = self.get_current_date()
        if current_date != self.last_reset_date:
            LOG.info(
                f"new day detected. resetting quota. previous: {self.last_reset_date}, current: {current_date}"
            )
            self.current_requests = 0
            self.last_reset_date = current_date

    def check_quota(self):
        """check if we need to wait for the next reset"""
        if self.current_requests >= DAILY_REQUEST_LIMIT:
            LOG.warning(
                f"Daily request limit reached: {self.current_requests}. Waiting for reset."
            )
            self.sleep_until_reset()
        self.check_quota_reset()

    def sleep_until_reset(self):
        seconds_to_wait = (
            self.seconds_until_midnight() + 30
        )  # add a buffer of 30 seconds
        LOG.info(f"Sleeping for {seconds_to_wait} seconds until next reset.")
        time.sleep(seconds_to_wait)
        self.check_quota_reset()

    def add_request_count(self, count: int = 1):
        self.current_requests += count
        self.check_quota()

# %%
def main():
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    # initialize quota manager
    qm = QuotaManager()
    # create output dir
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # ramdomly sampled demographic data
    df = read_data()
    # read instruction prompt
    instruction = read_instruction(BATCH_SIZE)

    for i in tqdm(range(0, 1_000), desc="iterations", position=0):
        iter_df = df[df["iter"] == i]

        # cut iter_df by batch size
        n_sections = len(iter_df) // BATCH_SIZE
        for sub_idx in tqdm(np.array_split(iter_df.index, n_sections), desc=f"batch", position=1):
            sub_df = iter_df.loc[sub_idx]
            # it is safe to assume that data is sorted by sampleid (because it is generated that way)
            id_begin = sub_df["sampleid"].min()
            id_end = sub_df["sampleid"].max()
            filename = f"{OUTPUT_DIR}/iter{i:03d}-{id_begin:03d}-{id_end:03d}.parquet"

            get_suggestions(sub_df, i, filename, instruction, qm)

    # handle missing responses
    handle_missing(df, instruction, qm)
    LOG.info("script completed successfully.")

# %%
if __name__ == "__main__":
    # configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    with logging_redirect_tqdm():
        main()


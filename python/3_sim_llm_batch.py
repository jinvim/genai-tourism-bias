# script to generate travel suggestions using llm
# uses batch prrocessing to handle large number of users
# %%
import pandas as pd
import geopandas as gpd
import numpy as np
import sys
from instructor.batch import BatchJob
import os
import logging
import time
from typing import List
from pydantic import BaseModel
from openai import OpenAI


PROMPT_PATH = "prompt-batch.md"
SHP_PATH = "../data/shp/tl_2023_us_state/tl_2023_us_state.shp"
DATA_PATH = "../data/sample.parquet"
OUTPUT_PATH = "../data/sim-llm-raw"

# API keys for either OpenAI or Google
GOOGLE_API_KEY = "KEY"
OPENAI_API_KEY = "KEY"

MIN_ITER = 0  # minimum number of iterations to process
MAX_ITER = 100  # maximum number of iterations to process
BATCH_SIZE = 20  # number of users to process in one batch
MAX_TOKENS = 7_500  # maximum tokens for the response
BATCH_PER_PART = 1_250 # number of batches to process in one part

# %%
class Suggestion(BaseModel):
    userid: str
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
def gen_input_str(iter, id, state, sex, agegrp, incgrp):
    return f"USERID:{iter:03d}-{id:03d} – {agegrp} {sex} from {state} with annual household income of {incgrp}"


def gen_input(sub_df):
    rows = sub_df.apply(
        lambda x: gen_input_str(
            x["iter"], x["sampleid"], x["state_name"], x["sex"], x["agegrp"], x["incgrp"]
        ),
        axis=1,
    )
    return "\n".join(rows)

# %%
def get_messages(dfs):
    instruction = read_instruction(BATCH_SIZE)
    for df in dfs:
        content = gen_input(df)
        yield [
            {
                "role": "system",
                "content": instruction,
            },
            {
                "role": "user",
                "content": content,
            },
        ]
# %%
class BatchManager:
    def __init__(self, model="gpt-4.1-nano-2025-04-14", temp=1.0):
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        self.client = OpenAI()
        self.model = model
        self.temp = temp
        self.filename = f"{self.model}-temp{self.temp}"
        self.output_dir = f"{OUTPUT_PATH}/{self.filename}"
        self.df = read_data()

        self.df = self.df[self.df["iter"] >= MIN_ITER]
        self.df = self.df[self.df["iter"] < MAX_ITER]

        self.create_output_dir()
        self.batch_files = []
        self.uploaded_files = []
        self.file_ids_file = f"{self.output_dir}/file-ids.csv"
        self.curr_batch_file = f"{self.output_dir}/current-batch.txt"

        self.curr_batch_id = None
        self.curr_input_file = None

        self.read_file_ids()
        if len(self.uploaded_files) == 0:
            logging.info("No uploaded batch files found. Initializing new batch files.")
            self.init_batchfiles()
            self.upload_files()
        self.check_current_batch_id()


    def create_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def create_batchfile(self, dfs, batch_file):
        if os.path.exists(batch_file):
            logging.info(f"Batch file {batch_file} already exists. Skipping creation.")
        else:
            messages = get_messages(dfs)
            BatchJob.create_from_messages(
                model=self.model,
                messages_batch=messages,
                file_path=batch_file,
                response_model=Suggestions,
                    max_tokens=MAX_TOKENS,
                    temperature=self.temp,
                )
        self.batch_files.append(batch_file)

    def init_batchfiles(self, df=None):
        if df is None:
            df = self.df
        if len(df) < BATCH_SIZE:
            dfs = [df]
            logging.info(f"Creating a single batch file with {len(df)} entries.")
        else:
            n_sections = len(df) // BATCH_SIZE
            split_idx = np.array_split(df.index, n_sections)
            dfs = [df.loc[idx] for idx in split_idx]

        if len(dfs) == 0:
            logging.warning("No data to process.")
        elif len(dfs) <= BATCH_PER_PART:
            logging.info(f"Creating a single batch file.")
            batch_filename = f"{self.output_dir}/batch-{self.filename}.jsonl"
            print(batch_filename)
            self.create_batchfile(dfs, batch_filename)
        elif len(dfs) > BATCH_PER_PART:
            part_size = (len(dfs) + BATCH_PER_PART - 1) // BATCH_PER_PART # ceiling division
            logging.info(f"Creating {part_size} batch files.")
            for i in range(part_size):
                first = int(i * n_sections / part_size)
                last = int((i + 1) * n_sections / part_size)
                batch_filename = f"{self.output_dir}/batch-{self.filename}-part{i}.jsonl"
                self.create_batchfile(dfs[first:last], batch_filename)

    def write_file_ids(self):
        ids_df = pd.DataFrame({"batch_file": self.batch_files, "uploaded_file": self.uploaded_files})
        ids_df.to_csv(self.file_ids_file, index=False)

    def read_file_ids(self):
        if os.path.exists(self.file_ids_file):
            ids_df = pd.read_csv(self.file_ids_file)
            self.batch_files = ids_df["batch_file"].tolist()
            self.uploaded_files = ids_df["uploaded_file"].tolist()
            if len(self.batch_files) != len(self.uploaded_files):
                logging.error("Batch files and IDs do not match in length. Skipping.")
                self.batch_files = []
                self.uploaded_files = []
            else:
                logging.info(f"Loaded existing batch file names and IDs from {self.file_ids_file}.")
        else:
            logging.info("No existing batch lists found. Initializing empty lists.")
            self.batch_files = []
            self.uploaded_files = []


    def upload_files(self):
        for batch_file in self.batch_files:
            if not os.path.exists(batch_file):
                logging.warning(f"Batch file {batch_file} does not exist. Skipping upload.")
                continue
            # see if the file is already uploaded by checking if there's corresponding id
            try:
                file_index = self.batch_files.index(batch_file)
                self.uploaded_files[file_index]
                continue  # file already uploaded, skip
            except IndexError:
                uploaded_file = self.client.files.create(
                    file=open(batch_file, "rb"),
                    purpose="batch",
                )
                self.uploaded_files.append(uploaded_file.id)
                logging.info(f"Uploaded batch file {batch_file} with ID {uploaded_file.id}.")
        self.write_file_ids()

    def write_current_batch_id(self):
        if self.curr_batch_id is not None:
            with open(self.curr_batch_file, "w") as f:
                f.write(self.curr_batch_id)
                logging.info(f"Wrote current batch ID {self.curr_batch_id} to file.")
    
    def check_current_batch_id(self):
        try:
            with open(self.curr_batch_file, "r") as f:
                self.curr_batch_id = f.read().strip()
                logging.info(f"Read current batch ID {self.curr_batch_id} from file.")
        except FileNotFoundError:
            logging.warning("Current batch ID file not found. Starting with no current batch ID.")
            self.curr_batch_id = None

    def get_processing_status(self):
        if self.curr_batch_id is None:
            logging.warning("No current batch ID set. Cannot retrieve processing status.")
            return None
        return self.client.batches.retrieve(self.curr_batch_id).status
    
    def get_curr_input_file(self):
        if self.curr_batch_id is None:
            logging.warning("No current batch ID set. Cannot retrieve current file ID.")
            return
        self.curr_input_file = self.client.batches.retrieve(self.curr_batch_id).input_file_id
    
    def get_current_index(self):
        if self.curr_batch_id is None:
            logging.warning("No current batch ID set. Cannot retrieve current index.")
            return None
        self.get_curr_input_file()
        index = self.uploaded_files.index(self.curr_input_file)
        return index
    
    def send_batch(self):
        batch = self.client.batches.create(
            input_file_id=self.curr_input_file,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        self.curr_batch_id = batch.id
        self.write_current_batch_id()
        logging.info(f"Started new batch with ID {self.curr_batch_id}.")

    
    def get_batch_output(self, filename):
        output_file_id = self.client.batches.retrieve(self.curr_batch_id).output_file_id
        if output_file_id is None:
            logging.warning("No output file ID found! Cannot retrieve output file.")
            return
        output = self.client.files.content(output_file_id).text
        # write output to file
        with open(filename, "w") as f:
            f.write(output)
        logging.info(f"Wrote output to file {filename}.")


    def send_batches(self):
        start_index = 0
        if self.curr_batch_id is not None:
            start_index = self.get_current_index()
            logging.info(f"Resuming from batch {self.curr_batch_id}.")
            progress = self.get_processing_status()

        for file in self.uploaded_files[start_index:]:
            self.curr_input_file = file
            curr_index = self.uploaded_files.index(file)
            filename = self.batch_files[curr_index].replace("batch-", "output-")
            if os.path.exists(filename):
                logging.info(f"Output file {filename} already exists. Skipping processing.")
            else:
                progress = "cued"
                if self.curr_batch_id is not None:
                    progress = self.get_processing_status()
                # if batch is not current being processed, send it
                
                while progress != "completed":
                    if progress not in ["validating", "in_progress", "finalizing", "completed"]:
                        if progress == "failed":
                            logging.error(f"Batch {self.curr_batch_id} failed. Retrying in 10 minutes.")
                            time.sleep(60 * 10)
                        self.send_batch()
                    logging.info(f"Batch {self.curr_batch_id} is {progress}. Waiting for completion.")
                    time.sleep(60 * 3) # wait for 1 minute
                    progress = self.get_processing_status()
                
                self.get_batch_output(filename)
            self.curr_batch_id = None

    def check_missing(self):
        def flatten(xss):
            return [x for xs in xss for x in xs]

        output_files = [f for f in os.listdir(self.output_dir) if f.startswith("output-")]
        parsed_list = []
        for output_file in output_files:
            output_path = os.path.join(self.output_dir, output_file)
            parsed, _ = BatchJob.parse_from_file(file_path=output_path, response_model=Suggestions)
            parsed_list.extend(parsed)

        if len(parsed_list) == 0:
            logging.warning("No parsed data found! Exiting.")
            return None
        # clean_parsed = [o.response for o in parsed_list if len(o.response) == BATCH_SIZE]
        clean_parsed = [o.response for o in parsed_list ]
        clean_parsed = flatten(clean_parsed)
        clean_df = pd.DataFrame([o.model_dump() for o in clean_parsed])
        def try_int(x):
            try:
                return int(x)
            except Exception as e:
                logging.warning(f"Error converting userid {x}: {e}")
                return None
        clean_df["iter"] = clean_df["userid"].str[:3].apply(try_int)
        clean_df["sampleid"] = clean_df["userid"].str[4:].apply(try_int)

        # clean_df.columns = clean_df.columns[-2:].to_list() + clean_df.columns[:-2].to_list()  # move iter and sampleid to the front
        clean_df = clean_df.drop(columns=["userid"])
        clean_df = clean_df.drop_duplicates(subset=["iter", "sampleid"])

        merged_df = self.df.merge(clean_df[["iter", "sampleid","location"]], on=["iter", "sampleid"], how="left")
        missing_df = merged_df[merged_df["location"].isna()].drop(columns=["location"])

        if len(missing_df) > 0:
            logging.info(f"Found {len(missing_df)} missing entries.")
            # randomly shuffle the missing entries, in a hope to fix errors in the next run
            return missing_df.sample(frac=1)

        logging.info("No missing entries found.")
        clean_df.to_parquet(f"{OUTPUT_PATH}/{self.model}-temp{self.temp}.parquet", index=False)
        logging.info("Saved cleaned data to parquet file.")
        return None
    # function to handle missing entries untile there are no missing entries
    def handle_missing(self):
        missing_df = self.check_missing()
        if missing_df is None:
            return
            
        logging.info(f"Handling {len(missing_df)} missing entries.")
        while len(missing_df) > 0:
            n_missing = len(missing_df)
            curr_time = int(time.time())

            self.filename = f"missing-{self.model}-temp{self.temp}-N{n_missing}-{curr_time}"
            self.init_batchfiles(missing_df)
            self.upload_files()
            self.send_batches()
            # check for missing entries again
            missing_df = self.check_missing()
            if missing_df is None:
                logging.info("No more missing entries found. Exiting.")
                break
# %%
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    model = input("Enter model name (default: gpt-4.1-nano-2025-04-14): ") or "gpt-4.1-nano-2025-04-14"
    temp = input("Enter temperature (default: 1.0): ")
    temp = float(temp) if temp else 1.0
    # model = "gpt-4.1-nano-2025-04-14"
    # temp = 1.5
    # model = "gpt-4.1-mini-2025-04-14"
    # temp = 1.0
    bm = BatchManager(model=model, temp=temp)
    bm.send_batches()
    bm.handle_missing()

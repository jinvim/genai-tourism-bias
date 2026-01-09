# create empirically drived, and completely randomized data for the random sample
# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product
from multiprocessing import Pool, cpu_count
from tqdm.contrib.concurrent import process_map

DIST_PATH = "../data/state-dist.csv"
OUTPUT_PATH = "../data/sim-emp"

# %%
# this is for reproducibility
# each seed in SEEDS will be used to construct random generator for each scenario
SEED = 2025
np.random.seed(SEED)
# 100 seeds are way overkill, but changing number of SEEDS also changes the results
# so by changing this number, we can avoid reusing seeds in different scripts
# (for ensuring true randomness & reproducibility across scripts)
SEEDS = np.random.randint(0, 2**32 - 1, 100).tolist()

# %%
# generate base dataframe with all combinations of org, dst, month
# this is used for "random" scenario with uniform probability
def gen_base_df():
    states = [ 1, 2, 4, 5, 6, 8, 9, 10, 11, 12,
        13, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 44,
        45, 46, 47, 48, 49, 50, 51, 53, 54, 55,
        56,
    ]
    dsts, orgs = zip(*list(product(states, states)))
    months = range(1, 13)
    return pd.DataFrame({
        "dst": dsts * len(months),
        "org": orgs * len(months),
        "month": np.repeat(months, len(dsts)),
    })

# %%
# %%
# function to draw random sample from the flow data
def draw_sample(flow_df, org, flow_col=None, seed=None):
    sub_df = flow_df[flow_df["org"] == org]
    weights = None
    if flow_col is not None:
        weights = sub_df[flow_col] / sub_df[flow_col].sum()
    sample = sub_df.sample(n=1, weights=weights, replace=True, random_state=seed).iloc[0]
    return sample["dst"], sample["month"]
# %%
# draw samples given the scenario
def draw_samples(demo_df, flow_df= None, flow_col=None, seed=None):
    # initialize random seeds for reproducibility
    orgs = demo_df["org"]
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**32 - 1, len(orgs))


    samples = []
    if flow_df is None:
        flow_df = gen_base_df()

    # could use apply here, but use for loop to show progress
    
    for i, org in enumerate(tqdm(
        orgs,
        position=SEEDS.index(seed) + 1,
        desc=f"seed {seed}",
        leave=True,
    )):
        dst, month = draw_sample(flow_df, org, flow_col=flow_col, seed=seeds[i])
        samples.append((dst, month))
    samples_df = pd.DataFrame(samples, columns=["dst", "month"])
    samples_df = pd.concat([demo_df[["iter", "sampleid", "org"]], samples_df], axis=1)
    # merge distance data for final output
    dist_df = pd.read_csv(DIST_PATH)
    return samples_df.merge(dist_df, on=["org", "dst"], how="left")
# %%
# I'm still suck on how to use multiprocessing with tqdm properly
# (or multilprocessing in python in general)
def process_scenario(args):
    name, flow_df, flow_col, seed = args
    demo_df = pd.read_parquet("../data/sample.parquet").rename(columns={"state": "org"})
    df = draw_samples(demo_df, flow_df=flow_df, flow_col=flow_col, seed=seed)
    df.to_parquet(f"{OUTPUT_PATH}/{name}.parquet", index=False)


if __name__ == "__main__":
    # load data
    advan = pd.read_csv("../data/csv/advan/advan-dist-cz-2022.csv", parse_dates=["date"])
    advan["month"] = advan["date"].dt.month
    nhts = pd.read_csv("../data/csv/nhts/nhts-2022.csv", parse_dates=["date"])
    nhts["month"] = nhts["date"].dt.month

    dfs_dict = {
        "rand": (None, None, SEEDS[0]),
        "advan-lei": (advan, "lei", SEEDS[1]),
        "advan-lei50mi": (advan, "lei50mi", SEEDS[2]),
        "nhts-lei": (nhts, "lei", SEEDS[3]),
        "nhts-lei50mi": (nhts, "lei50mi", SEEDS[4]),
    }

    # Prepare arguments for parallel processing
    args_list = [(name, flow_df, flow_col, seed) for name, (flow_df, flow_col, seed) in dfs_dict.items()]

    process_map(process_scenario, args_list, max_workers=cpu_count())
# draw random sample of 1,000 rows based on demographic weight
# repeat this for 1,000 times for a total of 1,000,000 rows
# %%
import pandas as pd
import numpy as np

N_SAMPLES = 1_000
N_ITER = 1_000
SEED = 2025

# %%
# create array of random seeds for reproducibility
np.random.seed(SEED)
seeds = np.random.randint(0, 2**32 - 1, N_ITER)

# %%
df = pd.read_parquet("../data/demo.parquet")
# %%
sub_dfs = []
for i, seed in enumerate(seeds):
    sub_df = df.sample(n=N_SAMPLES, weights="weight", replace=True, random_state=seed)
    sub_df["sampleid"] = range(N_SAMPLES)
    sub_df["iter"] = i
    sub_df = sub_df[["iter", "sampleid", "state", "sex", "agegrp", "incgrp"]]
    sub_dfs.append(sub_df)
# %%
all_df = pd.concat(sub_dfs, ignore_index=True)
# %%
all_df.groupby("sex", observed=True)["sampleid"].count()
# %%
all_df.groupby("agegrp", observed=True)["sampleid"].count()
# %%
all_df.groupby("incgrp", observed=True)["sampleid"].count()
# %%
all_df.to_parquet("../data/sample.parquet", index=False)
# %%

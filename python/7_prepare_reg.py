# prepare regression dataset combining simulation and empirical data
# %%
import pandas as pd
import numpy as np
from itertools import product

AI_DATA_PATH = "../data/sim-llm"
EMP_DATA_PATH = "../data/sim-emp"
OUTPUT_PATH = "../data/reg"
# very small constant to avoid log(0)
EPS = 1e-8
# %%
# generate base dataframe with all combinations of iter, org, dst, month
# without this, we would miss combinations with zero flows
def gen_base_df(max_iter = 999):
    states = [ 1, 2, 4, 5, 6, 8, 9, 10, 11, 12,
        13, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 44,
        45, 46, 47, 48, 49, 50, 51, 53, 54, 55,
        56,
    ]
    dsts, orgs = zip(*list(product(states, states)))
    iters = range(0, max_iter + 1)
    months = range(1, 13)
    month_df = pd.DataFrame({
        "dst": dsts * len(months),
        "org": orgs * len(months),
        "month": np.repeat(months, len(dsts)),
    })
    # base_df = pd.concat([month_df], ignore_index=True)
    base_df = pd.concat([month_df] * len(iters), ignore_index=True)
    base_df["iter"] = np.repeat(iters, len(months) * len(dsts))
    return base_df


# %%
# process simulation data
def process_sim_data(df):
    base_df = gen_base_df()
    df = (df
            .groupby(["iter", "dst", "org", "month"])["sampleid"]
            # .groupby(["dst", "org", "month"])["sampleid"]
            .count()
            .rename("flow_sim")
            .reset_index()
        )
    # whether a particular cell have any tourist flow or not
    df["any_sim"] = (df["flow_sim"] > 0).astype(int)
    df = base_df.merge(df, on=["iter", "dst", "org", "month"], how="left").fillna(0)
    df["out"] = df.groupby(["iter", "org"])["flow_sim"].transform("sum")
    # df["out"] = df.groupby(["org"])["flow_sim"].transform("sum")
    return df

# %%
# load empirical data and rename flow column
def load_emp_data(df, flow_col):
    # ensure date column is in datetime format
    if df["date"].dtype != "datetime64[ns]":
        df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    
    df = df.rename(columns={flow_col: "flow_emp"})

    return df[["dst", "org", "month", "flow_emp"]]

# process empirical data
# these will be used as empirical proportions in the models
def process_emp_data(df, flow_col):
    p_df = load_emp_data(df, flow_col)
    count_col = "flow_emp"
    p_col = "p_emp"
    # whether a particular cell have any tourist flow or not
    p_df["any_emp"] = (p_df[count_col] > 0).astype(int)
    # compute proportions
    p_df[p_col] = p_df[count_col] / p_df[count_col].sum()

    # destination j's share in the empirical data
    p_df["d"] = p_df.groupby(["dst"])[p_col].transform("sum")
    # month m's share in the empirical data
    p_df["m"] = p_df.groupby(["month"])[p_col].transform("sum")
    # origin i's share in the empirical data
    p_df["o"] = p_df.groupby(["org"])[p_col].transform("sum")
    
    # origin-destination pair i,j's share in the empirical data
    p_df["od"] = p_df.groupby(["org", "dst"])[p_col].transform("sum")
    # destination-month pair j,m's share in the empirical data
    p_df["dm"] = p_df.groupby(["dst", "month"])[p_col].transform("sum")
    
    # empirical probability of going to j in month m given origin i
    # (share of destination-month pair j,m given origin i)
    # later to be used for computing offsets
    # pe being zero will be handled in offset calculation
    p_df["pe"] = p_df.groupby(["org"])[p_col].transform(lambda x: x / x.sum())

    # association (lift) for destination-month pair j,m
    p_df["asdm"] = p_df["dm"] / (p_df["d"] * p_df["m"])
    # association (lift) for origin-destination pair i,j
    p_df["asod"] = p_df["od"] / (p_df["o"] * p_df["d"])

    # log-transform the proportion columns for modeling
    for col in ["d", "m", "asod", "asdm"]:
        # check if any of factors have zero values
        # if so, print a warning and add a small constant to avoid log(0)
        if p_df[col].min() == 0:
            print(f"column {col} has zero values. add small constant to avoid log(0).")
            p_df[f"ln_{col}"] = np.log(avoid_zero(p_df, col))
        else:
            # if not, use just log
            p_df[f"ln_{col}"] = np.log(p_df[f"{col}"])
    
    return p_df

# return values with small constant to avoid zero
# sum to 1 after adjustment
def avoid_zero(df, col):
    # grouping columns ensure proportions sum to 1
    group_cols = ["dst", "org", "month"]
    match col:
        case "d":
            group_cols.remove("dst")
        case "o":
            group_cols.remove("org")
        case "m":
            group_cols.remove("month")
        case "od" | "asod":
            group_cols.remove("org")
            group_cols.remove("dst")
        case "dm" | "asdm":
            group_cols.remove("dst")
            group_cols.remove("month")
        case "pe":
            group_cols = ["org"]
    return df.groupby(group_cols)[col].transform(lambda x: (x + EPS) / (x + EPS).sum())

# %%
# create dataset for regression
def process_reg_data(sim_df, emp_df, emp_flow_col):
    # process simulation data
    y_df = process_sim_data(sim_df)
    # process empirical data
    x_df = process_emp_data(emp_df, emp_flow_col)
    # merge simulation and empirical data
    reg_df = y_df.merge(x_df, on=["dst", "org", "month"], how="left")
    # create offset column
    reg_df["offset"] = reg_df["out"] * reg_df["pe"]
    # safe-guard against zero offsets
    reg_df["ln_offset"] = np.log(reg_df["offset"] + EPS)
    return reg_df

# %%
# load simulation data
sim_rand = pd.read_parquet(f"{EMP_DATA_PATH}/rand.parquet")
sim_advan = pd.read_parquet(f"{EMP_DATA_PATH}/advan-lei50mi.parquet")
sim_nhts = pd.read_parquet(f"{EMP_DATA_PATH}/nhts-lei50mi.parquet")
sim_gemini = pd.read_parquet(f"{AI_DATA_PATH}/gemini-2.5-flash-lite-preview-06-17.parquet")
sim_gpt = pd.read_parquet(f"{AI_DATA_PATH}/gpt-4.1-nano-2025-04-14.parquet")

# load empirical data
emp_advan = pd.read_csv("../data/csv/advan/advan-dist-cz-2022.csv")
emp_nhts = pd.read_csv("../data/csv/nhts/nhts-2022.csv")

# load distance data
dist_df = pd.read_csv("../data/state-dist.csv")
# %%
data_dfs = []
for sim_df, sim_name in [
    (sim_rand, "rand"),
    (sim_advan, "advan"),
    (sim_nhts, "nhts"),
    (sim_gemini, "gemini"),
    (sim_gpt, "gpt")
]:
    for emp_df, emp_name, emp_flow_col in [
        (emp_advan, "advan", "lei50mi"),
        (emp_nhts, "nhts", "lei50mi")
    ]:
        # skip if empirical-based simulation and empirical data do not match
        if (sim_name == "advan" or sim_name == "nhts") and (sim_name != emp_name):
            continue

        print(f"processing {sim_name} + {emp_name}")
        data_df = process_reg_data(sim_df, emp_df, emp_flow_col)

        data_df = data_df.merge(dist_df, on=["org", "dst"], how="left")
        data_df["self_loop"] = (data_df["org"] == data_df["dst"]).astype(int)
        
        data_df.to_parquet(f"{OUTPUT_PATH}/{sim_name}-{emp_name}.parquet", index=False)
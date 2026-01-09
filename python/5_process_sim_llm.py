# combine ai data with demographic and distance data for the analysis
# %%
import os
import pandas as pd
import geopandas as gpd

SHP_PATH = "../data/shp/tl_2023_us_state/tl_2023_us_state.shp"

OUTPUT_PATH = "../data/sim-llm"


def get_state_dict():
    states = gpd.read_file(SHP_PATH)
    states["GEOID"] = states["GEOID"].astype(int)
    states = states[states["GEOID"] < 60]  # filter out territories
    return states.set_index("GEOID")["NAME"].to_dict()


def clean_state(s):
    if "," in s:
        s = s.replace(",", "")
    if "/" in s:
        s = s.split("/")[0]
    if s == "Washington D.C.":
        s = "District of Columbia"
    return s.strip()

# %%
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
# %%
state_dict = get_state_dict()
state_dict = {v: k for k, v in state_dict.items()}
# %%
demo_df = pd.read_parquet("../data/sample.parquet").rename(columns={"state": "org"})
demo_df = demo_df[["iter", "sampleid", "org"]]
dist_df = pd.read_csv("../data/state-dist.csv")

# %%
# models = ["gpt-4.1-nano-2025-04-14", "gemini-2.5-flash-lite-preview-06-17", "gemini-2.5-flash-preview-05-20"]
models = [
    "gpt-4.1-nano-2025-04-14",
    "gpt-4.1-nano-2025-04-14-zero",
    "gpt-4.1-nano-2025-04-14-agent",
    "gpt-4.1-nano-2025-04-14-temp0.5",
    "gpt-4.1-mini-2025-04-14-temp1.0",
    "gemini-2.5-flash-lite-preview-06-17",
    "gemini-2.5-flash-lite-preview-06-17-zero",
    "gemini-2.5-flash-lite-preview-06-17-agent",
    "gemini-2.5-flash-lite-preview-06-17-temp0.5",
    "gemini-2.5-flash-lite-preview-06-17-temp1.5",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro",
    "grok-3-mini",
    "llama4-scout-instruct-basic",
]
model_dfs = []
for model in models:
    model_df = pd.read_parquet(f"../data/sim-llm-raw/{model}.parquet").rename(
        columns={"state": "dst", "recommended_month": "month"}
    )
    model_df["dst"] = model_df["dst"].apply(clean_state)
    model_df["dst"] = model_df["dst"].map(state_dict)
    model_df = demo_df.merge(model_df, on=["iter", "sampleid"], how="right")
    model_df = model_df[["iter", "sampleid", "org", "dst", "month"]]
    model_dfs.append(model_df)
# %%
for df, name in zip(model_dfs, models):
    df = df.merge(dist_df, on=["org", "dst"], how="left")
    df.to_parquet(f"{OUTPUT_PATH}/{name}.parquet", index=False)
# %%

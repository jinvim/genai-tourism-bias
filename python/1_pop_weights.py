# reads pums data, and creates probability weights stratifed by state, sex, age, and income
# %%
import pandas as pd

# %%
dfs = []
for file in ["a", "b", "c", "d"]:
    df = pd.read_parquet(
        f"../data/pums/psam_pus{file}.parquet",
        columns=["SERIALNO", "STATE", "SEX", "AGEP", "PWGTP"],
    )
    hou_df = pd.read_parquet(
        f"../data/pums/psam_hus{file}.parquet",
        columns=["SERIALNO", "ADJINC", "HINCP"],
    )
    # calculate income adjusted for inflation
    hou_df["ADJINC"] = hou_df["ADJINC"] // 1_000_000
    hou_df["income"] = hou_df["HINCP"] * hou_df["ADJINC"]
    # drop unnecessary columns, rename columns
    hou_df = hou_df.drop(columns=["HINCP", "ADJINC"])
    df = df.rename(
        columns={
            "STATE": "state",
            "SEX": "sex",
            "AGEP": "age",
            "PWGTP": "weight",
        }
    )
    df = df.merge(hou_df, on="SERIALNO", how="left")

    # create bins for income and age
    income_bins = [0, 24_999, 49_999, 74_999, 99_999, 149_999, float('inf')]
    income_labels = [
        "less than $25,000",
        "$25,000 to $49,999",
        "$50,000 to $74,999",
        "$75,000 to $99,999",
        "$100,000 to $149,999",
        "$150,000 or more",
    ]
    age_bins = [0, 17, 24, 34, 44, 54, 64, float('inf')]
    age_labels = [
        "under 18 years old",
        "18 to 24 years old",
        "25 to 34 years old",
        "35 to 44 years old",
        "45 to 54 years old",
        "55 to 64 years old",
        "65 years old or older",
    ]
    # convert sex to categorical
    df["sex"] = df["sex"].map({1: "male", 2: "female"})
    df["sex"] = df["sex"].astype("category")
    # convert age and income to categorical
    df["agegrp"] = pd.cut(df["age"], bins=age_bins, labels=age_labels)
    df["incgrp"] = pd.cut(df["income"], bins=income_bins, labels=income_labels)
    # exclude under 18 years
    df = df[df["agegrp"] != "under 18 years old"]
    # group by state, sex, agegrp, incgrp
    df = df.groupby(["state","sex", "agegrp", "incgrp"], observed=True)["weight"].sum().reset_index()
    df = df.rename(columns={"weight": "pop"})
    dfs.append(df)
# %%
all_df = pd.concat(dfs, ignore_index=True)
all_df["weight"] = all_df["pop"] / all_df["pop"].sum()
# %%
all_df.groupby("sex", observed=True)["weight"].sum()
# %%
all_df.groupby("agegrp", observed=True)["weight"].sum()
# %%
all_df.groupby("incgrp", observed=True)["weight"].sum()

# %%
all_df.to_parquet("../data/demo.parquet", index=False)

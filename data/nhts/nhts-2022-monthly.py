# process nhts mobility data
# %%
import pandas as pd
import geopandas as gpd
from itertools import product

SHP_PATH = "../../shp/raw/tl_2023_us_state/tl_2023_us_state.shp"

# %%
gdf = gpd.read_file(SHP_PATH)
gdf["GEOID"] = gdf["GEOID"].astype(int)
gdf = gdf[gdf["GEOID"] < 60]  # filter out territories
gdf = gdf.to_crs("EPSG:5070")  # not necessary, but needed for plotting

states = gdf["GEOID"].unique()
state_dict = gdf.set_index("STUSPS")["GEOID"].to_dict()
# %%
# create baseline df for all state pairs
pairs = [(pair[0], pair[1]) for pair in product(states, repeat=2)]
base_df = pd.DataFrame(pairs, columns=["dst", "org"])


# %%
dfs = []
for month in range(1, 13):
    print(f"Processing month: {month}")

    df = pd.read_csv(f"Passenger_OD_2022{month:02d}.csv")
    # calculate non-work trip that are at least 50 miles
    df["nonwork50mi"] = df["nonwork_50_75mi"] + df["nonwork_75_100mi"] + df["nonwork_100_150mi"] + df["nonwork_150_300mi"] + df["nonwork_gt300mi"]

    # exclude trips within the same CZ
    df = df[df["origin_zone_id"] != df["destination_zone_id"]]

    # rename columns
    df = df.rename(columns={"origin_state": "org", "destination_state": "dst",
                            "purpose_nonwork": "nonwork"})

    df = df.groupby(["dst", "org"], observed=True)[["nonwork", "nonwork50mi"]].sum().reset_index()

    # convert state abbreviations to FIPS
    df["org"] = df["org"].map(state_dict)
    df["dst"] = df["dst"].map(state_dict)
    # merge with base_df and fill missing values with 0
    df = pd.merge(base_df, df, on=["dst", "org"], how="left")
    
    # add date column
    date = f"2022-{month:02d}-01"
    df["date"] = date
    df = df.fillna(0)
    dfs.append(df)
# %%
df_2022 = pd.concat(dfs, ignore_index=True)

# %%
df_2022 = df_2022[["dst", "org", "date", "nonwork", "nonwork50mi"]]
df_2022 = df_2022.rename(columns={"nonwork": "lei", "nonwork50mi": "lei50mi"})
df_2022 = df_2022.sort_values(["dst", "date", "org"])
df_2022.to_csv("nhts-2022.csv", index=False)
# %%

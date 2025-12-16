# process advan mobility data
# %%
import libpysal
import pandas as pd
import geopandas as gpd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from itertools import product

DATA_PATH = "/Volumes/pny-2tb/Dewey/advan-processed"
DIST_PATH = "/Volumes/pny-2tb/tract-dist/tract-distkm"
SHP_PATH = "../../shp/raw/tl_2018_us_county/tl_2018_us_county.shp"
CZ_PATH = "../../shp/raw/NextGen_OD_Zone_ESRI_11152022/NextGen_Zone_0825.shp"
# TEMP_DIR = "/Volumes/pny-2tb/dask-temp"

# %%
# read flow data for a given year
def read_flow(date):
    df = dd.read_parquet(
        DATA_PATH,
        filters=[
            ("date", "==", date),
        ],
    )
    return df

# read distance data for a given county
def read_distance():
    dist_df = dd.read_parquet(
        DIST_PATH,
        columns=["dst", "org", "over50mi"],
    )
    return dist_df
# %%
if __name__ == "__main__":

    gdf = gpd.read_file(SHP_PATH)
    gdf["GEOID"] = gdf["GEOID"].astype(int)
    gdf["STATEFP"] = gdf["STATEFP"].astype(int)
    gdf = gdf[gdf["STATEFP"] < 60]  # filter out territories
    gdf = gdf.sort_values("GEOID")
    gdf = gdf.set_index("GEOID")
    gdf = gdf.to_crs("EPSG:5070")  # not necessary, but needed for plotting

    states = gdf["STATEFP"].unique()
    # create baseline df for all state pairs
    pairs = [(pair[0], pair[1]) for pair in product(states, repeat=2)]
    base_df = pd.DataFrame(pairs, columns=["dst", "org"])

    cluster = LocalCluster(
                            n_workers=4,
                            processes=True,
                            threads_per_worker=3,
                            memory_limit = "24GiB",
                            # local_directory=TEMP_DIR
                            )

    cz_gdf = gpd.read_file(CZ_PATH)
    cz_gdf = cz_gdf.to_crs("EPSG:5070")
    cz_dict = gpd.sjoin(gdf, cz_gdf[["zone_id", "geometry"]], how="left", predicate="intersects")["zone_id"].to_dict()

    with Client(cluster) as client:
        client.amm.start()
        dfs = [] 
        for month in range(1, 13):
            print(f"Processing month: {month}")
            date = f"2022-{month:02d}-01"
            df = read_flow(date)
            
            # state codes
            df["dstcounty"] = df["dstcbg"] // 10_000_000
            df["orgcounty"] = df["orgcbg"] // 10_000_000
            df["dststate"] = df["dstcbg"] // 10_000_000_000
            df["orgstate"] = df["orgcbg"] // 10_000_000_000
            
            # tract codes
            df["dst"] = df["dstcbg"] // 10
            df["org"] = df["orgcbg"] // 10

            # drop teritories
            df = df[df["dststate"] < 60]
            df = df[df["orgstate"] < 60]

            # drop flows within the same county
            df = df[df["dstcounty"] != df["orgcounty"]]

            # calculate leisure flows
            df["lei"] = df["home"] - df["work"]
            
            # first group by tract pairs to sum up flows
            df = df.groupby(["dst", "org"])[["lei"]].sum().reset_index()
            
            # add column whether travel distance is 50 miles or more
            dist_df = read_distance()
            df = df.merge(dist_df, on=["dst", "org"], how="left")

            # leisure flows that are over 50 miles
            df["lei50mi"] = df["lei"] * df["over50mi"]
            
            # group by county pairs to identify border flows
            df["dstcounty"] = df["dst"] // 1_000_000
            df["orgcounty"] = df["org"] // 1_000_000

            df = df.groupby(["dstcounty", "orgcounty"], observed=True)[["lei", "lei50mi"]].sum().reset_index().compute()
            
            # exclude trips within the same CZ
            df["dst_zone_id"] = df["dstcounty"].map(cz_dict)
            df["org_zone_id"] = df["orgcounty"].map(cz_dict)
            df = df[df["org_zone_id"] != df["dst_zone_id"]]

            # finally group by state pairs
            df["dst"] = df["dstcounty"] // 1_000
            df["org"] = df["orgcounty"] // 1_000

            # group by state
            df = df.groupby(["dst", "org"], observed=True)[["lei", "lei50mi"]].sum().reset_index()

            # merge with base_df and fill missing values with 0
            df = pd.merge(base_df, df, on=["dst", "org"], how="left")
            df["date"] = date
            df = df.fillna(0)
            dfs.append(df)
    df_2022 = pd.concat(dfs, ignore_index=True)

    df_2022 = df_2022[["dst", "org", "date", "lei", "lei50mi"]]
    df_2022.to_csv("advan-dist-cz-2022.csv", index=False)
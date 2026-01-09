# calculate physical distance (km) for each state pair
# using Vincenty formula and WGS84 ellipsoid
# %%
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import libpysal
import math
from itertools import product
from vincenty import vincenty

SHP_PATH = "../data/shp/tl_2023_us_state/tl_2023_us_state.shp"
# %%
# read shp files
gdf = gpd.read_file(SHP_PATH)
gdf["GEOID"] = gdf["GEOID"].astype(int)
gdf = gdf[gdf["GEOID"] < 60] # filter out territories
gdf = gdf.sort_values(by="GEOID")
# convert to projected CRS to calculate centroid
# then convert back to ellipsoid CRS for distance calculation
gdf = gdf.to_crs("EPSG:5070")
gdf["polygon"] = gdf["geometry"]
gdf["geometry"] = gdf["geometry"].centroid
gdf = gdf.to_crs("EPSG:4326")
cent_dict = gdf.set_index("GEOID")["geometry"].to_dict()
# %%
# create baseline df for all state pairs
pairs = [(pair[0], pair[1]) for pair in product(gdf["GEOID"], repeat=2)]
df = pd.DataFrame(pairs, columns=["dst", "org"])
# %%
df["dstp"] = df["dst"].map(cent_dict)
df["orgp"] = df["org"].map(cent_dict)

# %%
# convert to tuple of (lat, lon) for vincenty distance calculation
df["dstp"] = df["dstp"].apply(lambda x: (x.y, x.x))
df["orgp"] = df["orgp"].apply(lambda x: (x.y, x.x))

# %%
df["distkm"] = df.apply(lambda x: vincenty(x["dstp"], x["orgp"]), axis=1)
# %%
queen = libpysal.weights.Queen.from_dataframe(gdf, ids=gdf["GEOID"], geom_col="polygon")
# %%
df["border"] = df.apply(lambda x: x["org"] in queen.neighbors[x["dst"]], axis=1)
df["border"] = df["border"].astype(int)
# %%
df.drop(columns=["dstp", "orgp"]).to_csv("../data/state-dist.csv", index=False)
# %%

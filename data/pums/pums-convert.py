# reads pums data from csv to paquet format
# %%
import pandas as pd

# %%
for file in ["a", "b", "c", "d"]:
    for dtype in ["hus", "pus"]:
        df = pd.read_csv(f"psam_{dtype}{file}.csv")
        df.to_parquet(f"psam_{dtype}{file}.parquet", index=False)

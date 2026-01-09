# fit poisson and cloglog regression models
# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson

DATA_PATH = "../data/reg"
# %%
# fit poisson / cloglog regression model
def fit_reg(X, y, offset, family = "poisson"):
    model = sm.GLM(
        y,
        X,
        family=Poisson(),
        offset=offset
    )
    result = model.fit(cov_type='HC3')
    # extract parameters, p-values, standard errors
    params = result.params
    ses = result.bse
    pvals = result.pvalues

    return pd.DataFrame([params, ses, pvals], index=["coeff", "se", "pval"]).T

def reg(df):
    y = "flow_sim"
    X = ["ln_d", "ln_m", "ln_asdm", "ln_asod"]
    offset = "ln_offset"
    stat_df = (
        df.groupby("iter")[X + [y, offset]].apply(lambda x: fit_reg(x[X], x[y], x[offset])).reset_index()
    )
    return stat_df.rename(columns={"level_1": "param"})

# %%
sim_names = ["rand", "advan", "nhts", "gemini", "gpt"]
emp_names = ["advan", "nhts"]

result_dfs = []

for sim in sim_names:
    for emp in emp_names:
        # skip if empirical-based simulation and empirical data do not match
        if (sim == "advan" or sim == "nhts") and (sim != emp):
            continue
        print(f"fitting {sim} using {emp}")
        df = pd.read_parquet(f"{DATA_PATH}/{sim}-{emp}.parquet")
        result_df = reg(df)
        result_df["sim"] = sim
        result_df["emp"] = emp
        result_dfs.append(result_df)
result_df = pd.concat(result_dfs, ignore_index=True)

result_df.groupby(["sim","emp","param"])["coeff"].describe()
# %%
result_df.to_parquet("../data/reg/betas.parquet", index=False)

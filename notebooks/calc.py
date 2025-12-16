import pandas as pd
import geopandas as gpd
import numpy as np
from scipy import stats
import os
from itertools import product
import statsmodels.api as sm

DATA_PATH = "data"
# check if rendering within sections folder
if os.getcwd().split("/")[-1] == "notebooks":
    DATA_PATH = "../" + DATA_PATH

SHP_PATH = DATA_PATH + "/geo/tl_2023_us_state/tl_2023_us_state.shp"
DIST_PATH = DATA_PATH + "/geo/dist.csv"

dist = pd.read_csv(DIST_PATH)


def get_state_dict(acc=False):
    states = gpd.read_file(SHP_PATH)
    states["GEOID"] = states["GEOID"].astype(int)
    states = states[states["GEOID"] < 60]  # filter out territories
    if acc:
        return states.set_index("GEOID")["STUSPS"].to_dict()
    return states.set_index("GEOID")["NAME"].to_dict()


state_dict = get_state_dict()
states = list(state_dict.keys())
states.sort()


def gen_base_nodes(iters: list, states: list):
    return pd.DataFrame(
        {"iter": np.repeat(iters, len(states)), "dst": states * len(iters)}
    )


# generate base dataframe that populates edges for each iters
def gen_base_edges(iters: list, states: list):
    dsts, orgs = zip(*list(product(states, states)))
    return pd.DataFrame(
        {
            "iter": np.repeat(iters, len(dsts)),
            "dst": dsts * len(iters),
            "org": orgs * len(iters),
        }
    )


# generate base dataframe that has all dst and 12 months
def gen_base_dst_month(states: list):
    months = list(range(1, 13))
    return pd.DataFrame(list(product(states, months)), columns=["dst", "month"])


def mean_dist(df):
    return df.groupby("iter")["distkm"].mean()


def median_dist(df):
    return df.groupby("iter")["distkm"].median()


def median_dist_nozero(df):
    return df[df["distkm"] > 0].groupby("iter")["distkm"].median()


# calculate the correlation between two edge lists (similar to QAP)
def gcor(df1, df2):
    edges1 = edges(df1, fill=True)
    edges2 = edges(df2, fill=True)
    # merge the two edge lists on iter, dst, and org
    merged = pd.merge(edges1, edges2, on=["iter", "dst", "org"], suffixes=("_1", "_2"))
    # calculate the correlation between the two edge lists
    return (
        merged.groupby("iter")[["count_1", "count_2"]]
        .apply(lambda x: stats.pearsonr(x["count_1"], x["count_2"])[0])
        .mean()
    )


def edges(df, fill=False):
    edges_df = (
        df.groupby(["iter", "dst", "org"])["sampleid"]
        .count()
        .reset_index()
        .rename(columns={"sampleid": "count"})
    )
    # if fill, populate all combinations of dst and org for each iter
    if fill:
        base_edges = gen_base_edges(df["iter"].unique(), states)
        edges_df = base_edges.merge(edges_df, on=["iter", "dst", "org"], how="left")
        edges_df["count"] = edges_df["count"].fillna(0).astype(int)
    return edges_df


# return network as sociomatrix, by using mean of edges over all iterations
def sociomat(df, pct=False, exclude_self=False):
    edges_df = edges(df, fill=True)
    if exclude_self:
        edges_df = edges_df[edges_df["dst"] != edges_df["org"]]
    if pct:
        edges_df["total"] = edges_df.groupby("iter")["count"].transform("sum")
        edges_df["count"] = edges_df["count"] / edges_df["total"]

    edges_df = edges_df.groupby(["dst", "org"])["count"].mean().reset_index()
    return edges_df.pivot_table(index="dst", columns="org", values="count")


def indegree(df):
    edges_df = edges(df)
    indegree_df = edges_df.groupby(["iter", "dst"])["count"].sum().reset_index()
    indegree_df = indegree_df.rename(columns={"count": "indegree"})
    base_df = gen_base_nodes(df["iter"].unique(), states)
    return base_df.merge(indegree_df, on=["iter", "dst"], how="left").fillna(0)


def border(df):
    by_iter = (
        df.groupby(["iter"]).agg({"sampleid": "count", "border": "sum"}).reset_index()
    )
    return by_iter["border"] / by_iter["sampleid"]


def self_loops(df):
    by_iter = df.copy()
    by_iter["is_self"] = by_iter["dst"] == by_iter["org"]
    by_iter = (
        by_iter.groupby(["iter"])
        .agg({"sampleid": "count", "is_self": "sum"})
        .reset_index()
    )
    return by_iter["is_self"] / by_iter["sampleid"]


# calculate indegree centralization
# based on Freeman's (1979) definition
def centralization(df):
    indegree_df = indegree(df)

    def freeman_score(indegree_df):
        max_indegree = indegree_df["indegree"].max()
        size = len(indegree_df["dst"])
        max_possible_sum = (size - 1) * max_indegree
        diff_sum = (max_indegree - indegree_df["indegree"]).sum()
        return diff_sum / max_possible_sum

    return indegree_df.groupby("iter").apply(freeman_score)


# calculate indegree assortativity
# based on Newman (2003) definition
def assortativity(df):
    edges_df = edges(df)
    edges_df = edges_df[edges_df["dst"] != edges_df["org"]]
    indegree_df = indegree(df)

    edges_df = edges_df.merge(
        indegree_df.rename(columns={"indegree": "indegree_dst"}),
        on=["iter", "dst"],
        how="left",
    ).merge(
        indegree_df.rename(columns={"dst": "org", "indegree": "indegree_org"}),
        on=["iter", "org"],
        how="left",
    )
    # calculate the correlation between indegree_dst and indegree_org
    return edges_df.groupby("iter").apply(
        lambda x: pearsonr(x["indegree_dst"], x["indegree_org"])[0],
        include_groups=False,
    )


# calculate graph-level weighted reciprocity
# Sequartini et al. (2013)
def reciprocity(df):
    edges_df = edges(df)
    # remove self loops
    edges_df = edges_df[edges_df["dst"] != edges_df["org"]]
    # create inverse df where org and dst are swapped
    edges_df_inv = edges_df.rename(
        columns={"dst": "org", "org": "dst", "count": "countinv"}
    )
    # merge with inverse df
    # this way, we can calculate the minimum of the two counts
    edges_df = edges_df.merge(edges_df_inv, on=["iter", "dst", "org"], how="outer")
    edges_df = edges_df.fillna(0)  # fill missing values with 0 (no flows)
    # calculate the minimum of the weighted edges between the two nodes
    # (reciprocated edges)
    edges_df["min"] = np.minimum(edges_df["count"], edges_df["countinv"])
    # calculate sum of counts (total weight of network) and sum of reciprocated edges
    recip = edges_df.groupby(["iter"])[["count", "min"]].sum().reset_index()
    # since we double count the reciprocated edges, we divide by 2
    return recip["min"] / 2 / recip["count"]


def month(df):
    return df.groupby(["iter", "month"])["sampleid"].count().reset_index()


# computationally efficient Gini coefficient calculation
# by Gaëtan de Menten
# https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python
def calc_gini(x):
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    sorted_x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(sorted_x, dtype=float)
    # The above formula, with all weights equal to 1 simplifies to:
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def gini(df):
    indegree_df = indegree(df)
    return indegree_df.groupby("iter")["indegree"].apply(calc_gini)


def gini_month(df):
    month_df = month(df)
    return month_df.groupby("iter")["sampleid"].apply(calc_gini)


def long_stats(df, funcs: list):
    stats = [func(df).values for func in funcs]
    stat_dfs = [
        pd.DataFrame({"stat": funcs[i].__name__, "value": stat})
        for i, stat in enumerate(stats)
    ]
    return pd.concat(stat_dfs, ignore_index=True)


def add_datacol(dfs, labels=None):
    for i, df in enumerate(dfs):
        label = f"data{i}" if labels is None else labels[i]
        df.insert(1, "data", label)


def long_data(dfs, func, labels=None):
    if func.__name__ == "long_stats":
        # stats = [median_dist_nozero, self_loops, reciprocity, gini, resilience]
        stats = [
            gini_month,
            centralization,
            reciprocity,
            # resilience,
            mean_dist,
            border,
            self_loops,
        ]
        dfs = [func(df, stats) for df in dfs]
    else:
        dfs = [func(df) for df in dfs]
    add_datacol(dfs, labels)
    return pd.concat(dfs, ignore_index=True)


# calculate destination resilence as shannon entropy (- p * log(p))
# Lee & Pennington-Gray's (2025) REFLEX model
def resilience(df):
    edges_df = edges(df)
    edges_df["inflow"] = edges_df.groupby(["iter", "dst"])["count"].transform("sum")
    edges_df["p"] = edges_df["count"] / edges_df["inflow"]
    edges_df["resilience"] = -edges_df["p"] * np.log(edges_df["p"])
    resilience = edges_df.groupby(["iter", "dst"])["resilience"].sum()
    # return resilience.groupby("iter").median()
    return resilience.groupby("iter").mean()


def indegree_pct(df):
    indegree_df = indegree(df)
    indegree_df["total"] = indegree_df.groupby("iter")["indegree"].transform("sum")
    indegree_df["indegree_pct"] = indegree_df["indegree"] / indegree_df["total"]
    return indegree_df


def stat_mat(dfs, func):
    stat_dfs = [func(df) for df in dfs]
    stat_dfs = [df.groupby("dst")[func.__name__] for df in stat_dfs]
    means = [df.mean() for df in stat_dfs]

    index = [f"data{i}" for i in range(len(dfs))]
    means_df = pd.DataFrame(means, index=index).T
    return means_df


def dst_by_month(df):
    base_df = gen_base_dst_month(states)
    grouped = df.groupby(["iter", "dst", "month"])["sampleid"].count().reset_index()
    grouped = grouped.rename(columns={"sampleid": "count"})
    grouped = grouped[
        grouped["month"].isin(range(1, 13))
    ]  # remove months outside of 1-12
    grouped["total"] = grouped.groupby("iter")["count"].transform("sum")
    grouped["prop"] = grouped["count"] / grouped["total"]
    grouped = grouped.groupby(["dst", "month"])["prop"].mean().reset_index()
    return base_df.merge(grouped, on=["dst", "month"], how="left").fillna(0)


def fit_reg(X, y):
    # fit poisson regression
    model = sm.GLM(y, X, family=sm.families.Poisson())
    result = model.fit()
    params = result.params
    params["r2_p"] = result.pseudo_rsquared()
    return params


def reg(df):
    df = edges(df, fill=True)
    df = df.merge(dist, on=["dst", "org"], how="left")
    df["distkkm"] = df["distkm"] / 1000  # convert to thousand km
    df["self"] = (df["dst"] == df["org"]).astype(int)
    df["intercept"] = 1
    X = ["distkkm", "border", "self", "intercept"]
    y = ["count"]
    stat_df = (
        df.groupby("iter")[X + y].apply(lambda x: fit_reg(x[X], x[y])).reset_index()
    )
    return stat_df.melt(id_vars=["iter"], var_name="param", value_name="value")

# Perform permutation t-tests for each pair of data
def permutation_ttests(df, stat_col, data_names):

    results = []
    stats_list = df[stat_col].unique()
    for stat in stats_list:
        stat_data = df[df[stat_col] == stat]
        for i in range(len(data_names)):
            for j in range(i + 1, len(data_names)):
                x = stat_data[stat_data["data"] == data_names[i]]["value"]
                y = stat_data[stat_data["data"] == data_names[j]]["value"]
                res = stats.permutation_test(
                    (x, y),
                    lambda a, b: stats.ttest_ind(a, b).statistic,
                    permutation_type='independent',
                    n_resamples=10000
                )
                results.append({
                    "stat": stat,
                    "data_x": data_names[i],
                    "data_y": data_names[j],
                    "statistic": res.statistic,
                    "pvalue": res.pvalue
                })
    return pd.DataFrame(results)
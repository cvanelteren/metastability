"""
This was was used to preprocess the figure where random connected graphs
are shown in the paper
"""
import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy
from tqdm import tqdm

import swifter
from figure4 import *
from success_tipping import get_success

dfs = []

base = "/run/media/casper/9ee50225-d11d-4dae-81d9-9fa441997327/recal"
target_fp = "combined_errors_n.pkl"
if not os.path.exists(target_fp):
    dfs_ = pd.DataFrame()
    dfs_.to_pickle(target_fp)

for i in tqdm(range(100)):
    fp = f"{base}/kite_isi_g.id={i}_ts.pkl"
    # fp = "kite_intervention_beta=0.5732374683235916.pkl"
    try:
        df = pd.read_pickle(fp)
    except Exception as e:
        print(e)
        continue

    success = df.swifter.apply(get_success, axis=1)
    tmp = df.swifter.apply(estimate_white_noise, axis=1)

    errors = []
    for idx, row in tmp.reset_index().iterrows():
        errors.append(row.iloc[1])
    errors = pd.DataFrame(errors)

    idx = np.where(errors.label == "control")[0]

    for x in "less_w greater_w\tless_n greater_n".split("\t"):
        jdx, kdx = x.split()
        jdx = np.where(errors.columns == jdx)[0]
        kdx = np.where(errors.columns == kdx)[0]
        mu = np.stack((errors.iloc[idx, jdx], errors.iloc[idx, kdx])).squeeze()
        mu = np.nanmean(mu, axis=0)
        errors.iloc[idx, kdx] = mu
        errors.iloc[idx, jdx] = mu

    # errors["success"] = success
    errors["greater_t"] /= errors["greater_t"].max()
    errors["less_t"] /= errors["less_t"].max()
    errors["success_1_to_0"] = df["success_1_to_0"]
    errors["success_0_to_1"] = df["success_0_to_1"]
    errors["id"] = i

    dfs_ = pd.read_pickle(target_fp)
    dfs_.append(errors, ignore_index=True)
    dfs_.to_pickle(target_fp)

    del errors, tmp, df, dfs_
    # dfs.append(errors)

# dfs_ = pd.concat(dfs)


# print(df.columns)

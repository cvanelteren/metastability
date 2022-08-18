import proplot as plt, cmasher as cmr, pandas as pd,\
    numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy
from tqdm import tqdm

from figure4 import *
from success_tipping import get_success

dfs = []
for i in tqdm(range(100)):
    fp = f"/run/media/casper/9ee50225-d11d-4dae-81d9-9fa441997327/kite_isi_g.id={i}.pkl"
    # fp = "kite_intervention_beta=0.5732374683235916.pkl"
    df = pd.read_pickle(fp)

    success = df.apply(get_success, axis = 1)
    tmp = df.apply(estimate_white_noise, axis=1)

    errors = []
    for idx, row in tmp.reset_index().iterrows():
        errors.append(row.iloc[1])
    errors = pd.DataFrame(errors)

    idx = np.where(errors.label == "control")[0]
    jdx = np.where(errors.columns == "less_w")[0]
    kdx = np.where(errors.columns == "greater_w")[0]
    mu = np.stack((errors.iloc[idx, jdx], errors.iloc[idx, kdx])).squeeze()
    mu = np.nanmean(mu, axis = 0)
    errors.iloc[idx, kdx] = mu
    errors.iloc[idx, jdx] = mu

    idx = np.where(errors.label == "control")[0]
    jdx = np.where(errors.columns == "less_n")[0]
    kdx = np.where(errors.columns == "greater_n")[0]
    mu = np.stack((errors.iloc[idx, jdx], errors.iloc[idx, kdx])).squeeze()
    mu = np.nanmean(mu, axis = 0)
    errors.iloc[idx, kdx] = mu
    errors.iloc[idx, jdx] = mu


    errors["success"] = success
    errors["greater_t"] /= errors["greater_t"].max()
    errors["less_t"] /= errors["less_t"].max()
    errors["id"] = i
    dfs.append(errors)
dfs_ = pd.concat(dfs)
dfs_.to_pickle("combined_errors.pkl")




print(df.columns)

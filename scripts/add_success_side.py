import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy
from tqdm import tqdm

import multiprocessing as mp

fp = "../combined_errors.pkl"
dfs = pd.read_pickle(fp)

base = "/run/media/casper/9ee50225-d11d-4dae-81d9-9fa441997327/recal"
# base = "/home/casper/test/backup_data_meta/recal"


e = pd.DataFrame()
cols = "success_1_to_0 success_0_to_1 id seed nudge label".split()


def load_data(i: int):
    fp = f"{base}/kite_isi_g.id={i}_ts.pkl"
    # fp = "kite_intervention_beta=0.5732374683235916.pkl"
    try:
        df = pd.read_pickle(fp)[cols]
        return df
    except Exception as e:
        print(e)
        return


for idx in tqdm(range(100)):
    df = load_data(idx)
    if df is not None:
        e = e.append(df, ignore_index=1)

# with mp.Pool(1) as p:
#     for df in tqdm(p.imap_unordered(load_data, range(100))):
#         if df is not None:
#             e = e.append(df[cols], ignore_index=1)

e.to_pickle("success_n.pkl")
print(e)

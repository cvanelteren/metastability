import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy

from tqdm import tqdm

from experiment import small_tree


g = nx.krackhardt_kite_graph()
beta = 0.5732374683235916

# g = small_tree()
# beta = 0.9001444450539172
n_steps = 1e6
sample_size = 1

df = []


def get_isi(seed, nudges={}):
    m = models.Potts(
        g,
        t=1 / beta,
        sampleSize=sample_size,
        seed=seed,
    )
    m.nudges = nudges
    m.states = 0
    tmp = m.simulate(n_steps).mean(1)
    isi = np.diff(np.where(tmp == 0.5)[0])
    return isi, tmp


def run(seed):
    df = []
    isi, system = get_isi(seed)
    row = dict(label="control", isi=isi, seed=seed, system=system)
    df.append(row)
    for node in tqdm(g.nodes()):
        nudges = {node: np.inf}
        isi, system = get_isi(seed, nudges)
        row = dict(label=node, isi=isi, seed=seed, system=system)
        df.append(row)
    return df


import multiprocessing as mp

seeds = (
    0,
    12,
    123,
    1234,
    12345,
    123456,
    1234567,
)
with mp.Pool() as p:
    for i in p.imap_unordered(
        run,
        seeds,
    ):
        for j in i:
            df.append(j)

df = pd.DataFrame(df)
# df.to_pickle(f"small_tree_isi_{beta=}.pickle")
df.to_pickle(f"./kite_isi_{beta=}.pkl")
print(df)

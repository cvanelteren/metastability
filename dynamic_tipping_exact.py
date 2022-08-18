"""
Use the starting positions from
dynamic tipping forward in time
"""


import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy

import cupy as cp
from exact import *

g = nx.krackhardt_kite_graph()
A = nx.adjacency_matrix(g).todense()
A = cp.asarray(A)

from pathlib import Path

for fp in Path("./").iterdir():
    if (
        fp.name.startswith("dynamic_tipping_kite_single_side")
        and not "forward" in fp.name
    ):
        print(fp)
        df = pd.read_pickle(fp)

        name = fp.name.split(".")[0] + "_forward.pkl"
        n = len(g)
        states, allowed = gen_states(n)
        targets = [0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10]  # time index
        p = df.p.iloc[0]

        df = {}
        beta = 0.5732374683235916
        T = 100
        # settings = Settings(beta, T, g, NodeToSystem)
        settings = Settings(beta, T, g, NodeToMacroBit)
        for t in targets:
            if t not in p:
                continue
            pi = p[t]
            s = list(pi.keys())
            sub_allowed = []
            for idx, si in enumerate(s):
                si = np.asarray(si)
                sidx = to_binary(si)
                sub_allowed.append(sidx)

            structure = (A, states, allowed)
            dfi = simulate_reduced(settings, sub_allowed, structure)
            df[t] = dfi

        print(f"Saving to {name}")
        try:
            pd.to_pickle(df, name)
        except:
            continue

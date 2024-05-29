from exact import *
import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy


targets = [0.3]
etas = np.linspace(-1, 1, 5)
# etas = [0]
beta = 0.5732374683235916
g = nx.krackhardt_kite_graph()
T = 100  # time steps
settings = Settings(beta, T, g, SystemToNode)

df = sim2(settings, interventions=etas, targets=targets)
fp = f"experiment5b_kite_{beta=}_{targets=}.pkl"
df.to_pickle(fp)
print(fp)

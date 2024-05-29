from exact import *
import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy


targets = [0.3]
etas = np.linspace(0.3, 0.4, 10)
# etas = [0]
beta = 0.5732374683235916
g = nx.krackhardt_kite_graph()
T = 500  # time steps
settings = Settings(beta, T, g, SystemToNode)

df = experiment5(settings, interventions=etas, targets=targets)
fp = f"experiment5_kite_{beta=}_{targets=}.pkl"
df.to_pickle(fp)
print(fp)

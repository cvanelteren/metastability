import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy
from exact import Settings, SystemToNode, NodeToSystem, sim2

g = nx.krackhardt_kite_graph()
beta = 0.5732374683235916  # / 4
T = 1000_000
settings = Settings(beta, T, g, SystemToNode)
# settings = Settings(beta, T, g, NodeToSystem)

# invs = {-1 : 0}
invs = {node: 1 for node in g.nodes()}
invs[-1] = 0
df = sim2(settings, intervention=invs)
fp = f"kite_intervention_{beta=}.pkl"
df.to_pickle(fp)

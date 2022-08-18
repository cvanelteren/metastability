import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy

from exact import *

g = nx.krackhardt_kite_graph()
# g = nx.Graph()
# g.add_node(0)
# g.add_node(1)
# g.add_edge(0, 1)
beta = 0.5732374683235916  # / 4
# beta = 10.0

T = 310
settings = Settings(beta, T, g, SystemToNode)
# settings = Settings(beta, T, g, NodeToSystem)

# invs = {-1 : 0}
dfs = []
infs = np.linspace(-1, 1, 11) * 1
# infs = [1.0]
# infs = [0.5]
targets = []
df = sim2(settings, interventions=infs, targets=targets)

fp = f"kite_intervention_{beta=}_{targets=}_{T=}_multiple_large2.pkl"
print(f"saving to {fp=}")
df.to_pickle(fp)

# fig, ax = plt.subplots(
# for idx, dfi in df.groupby("node"):
#     p = np.stack(dfi.p0)
#     q = np.stack(dfi.largest)
#     KL = np.sum((p - q) ** 2, axis=-1)
#     ax.plot(dfi.eta, KL)
# fig.show()
# plt.show(block=1)

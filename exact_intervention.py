import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy
from exact import Settings, SystemToNode, NodeToSystem, sim2

g = nx.krackhardt_kite_graph()
beta = 0.5732374683235916  # / 4
T = 100
settings = Settings(beta, T, g, SystemToNode)
# settings = Settings(beta, T, g, NodeToSystem)

# invs = {-1 : 0}
INF = 0.01
invs = {node: INF for node in g.nodes()}
invs[-1] = 0
# see tipping point effect
targets = [0.5]

# targets = np.linspace(0, 1, 11)
print(targets)
# see evolution from [0]
df = sim2(settings, intervention=invs, targets=targets)
fp = f"kite_intervention_{beta=}_{targets=}_{T=}.pkl"
df.to_pickle(fp)


import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy

fp = f"kite_intervention_beta=0.5732374683235916_targets=[0.5]_T={T}.pkl"

df = pd.read_pickle(fp)
df.columns

from scipy.special import kl_div, rel_entr

p = np.stack(df.ps)
from imi.infcy import KL

kl = np.zeros(p.shape[:2])
for idx, pi in enumerate(p):
    s = pi.shape
    kl[idx] = KL(p[-1], pi)

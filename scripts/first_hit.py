import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy
from pathlib import Path

import re

base = Path("./data")
fp = "exact_kite_dyn=ising_beta=0.5732374683235916_T=200.pkl"
n_samples = int(re.findall("T=\d+", fp)[0].split("=")[1])
df = pd.read_pickle(base / Path(fp))

from exact import gen_states, ising, get_transfer

g = nx.krackhardt_kite_graph()

n = len(g)
states, allowed = gen_states(n)
# states = states.get()
x = []
for idx, s in enumerate(states):
    if s.mean() == 0.5:
        x.append(s)
x = np.array(x)

ntrials = 1

hits = {}
from tqdm import tqdm

for node in g.nodes():
    m = models.Potts(g, t=1 / 0.5732374683235916, sampleSize=1)
    m.nudes = {node: np.inf}

    times = []
    for trial in tqdm(range(ntrials)):
        for s in x:
            m.states = s
            idx = 0
            while True:
                m.updateState(m.sampleNodes(1)[0])
                side = np.mean(m.states)
                if side == 0 or side == 1:
                    times.append(1 if side == s[node] else 0)
                    break
                idx += 1
    hits[node] = times

df = []
for k, v in hits.items():
    counts, edges = np.histogram(v, density=True, bins=[-0.5, 0.5, 1.5])
    print(counts)
    row = dict(label=k, x=edges, y=counts)
    df.append(row)
df = pd.DataFrame(df)

import seaborn as sbs

fig, ax = plt.subplots()

sbs.catplot(x="x", y="y", data=df)
ax.legend()
fig.show()
plt.show(block=1)


# E = ising(states, A)
# p, p0 = get_transfer(n, E, settings.beta, allowed)

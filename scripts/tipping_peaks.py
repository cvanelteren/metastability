import matplotlib.pyplot as plt, cmasher as cmr, pandas as pd
import numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy, signal
from tqdm import tqdm

g = nx.Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(0, 3)

g.add_edge(1, 4)
g.add_edge(1, 5)

g.add_edge(2, 6)
g.add_edge(2, 7)

g.add_edge(3, 8)
g.add_edge(3, 9)

g.add_edge(1, 2)


temps = np.linspace(0, 5, 10)
m = models.Potts(g)
results = {}
n = 1e6
from tqdm import tqdm

for t in tqdm(temps):
    m.t = t
    results[t] = signal.detect_peaks(m, n, threshold=0.5, allowance=0, burnin=1e6)

import proplot as pplt

fig = pplt.figure(sharex=0)
layout = [[1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10]]
ax = fig.add_subplots(layout)
ax[0].plot(temps, [len(i) for i in results.values()])
ax[0].set_xlabel("Temperature (T)")
ax[0].set_ylabel(f"Peak count after {n=} samples")

for (k, v), axi in zip(results.items(), ax[1:]):
    axi.hist(np.diff(v), density=1)
    axi.set_title(f"T={round(k,2)}")
    axi.set_xlabel("Inter stimulus interval")
    axi.set_ylabel("PMF")
fig.show()
plt.show(block=1)

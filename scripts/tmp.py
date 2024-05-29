import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy

from matplotlib.pyplot import pause

n = 24
g = nx.grid_graph((n, n), periodic=1)
m = models.Bornholdt(
    g,
    t=1.5,
    alpha=4,
    sampleSize=1,
)
s = m.simulate(10000)

import time

fig, ax = plt.subplots()
h = ax.imshow(s[0].reshape(n, n), vmin=0, vmax=1)
idx = 0
fig.show()
t = ax.set_title(f"T={idx}")
print("running")
while True:
    h.set_data(s[idx].reshape(n, n))
    idx += 100
    t.set_text(f"T={idx}")
    idx = idx % s.shape[0]
    fig.canvas.start_event_loop(1e-10)
    fig.canvas.draw_idle()
    # pause(1e-20)

plt.show(block=1)

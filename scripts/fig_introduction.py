import matplotlib.pyplot as plt, cmasher as cmr, pandas as pd
import numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy

import proplot as pplt
from plexsim.utils.graph import recursive_tree


g = recursive_tree(5)
m = models.Potts(g, t=0.8)
s = m.simulate(10000).mean(1)

fig, ax = pplt.subplots()
ax.plot(s)
inax = ax.inset_axes((0.5, 0.5, 0.5, 0.5))
nx.draw_forceatlas2(g, ax=inax)
fig.show()

plt.show(block=1)

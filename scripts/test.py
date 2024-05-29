import cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
import matplotlib as pplt, proplot as plt

# print(plt.__version__)
print(pplt.__version__)
g = nx.krackhardt_kite_graph()
pos = nx.kamada_kawai_layout(g)

fig, ax = plt.subplots()

# for i in range(3):
    # nx.draw(g, pos=pos, ax=ax[0, i])
    # nx.draw(g, pos=pos, ax=ax[1, 0])
    # ax[2, i].plot(*np.random.rand(2, 100) + 1e2 * i)
# ax.set_xlabel("Test", fontsize=30)
#
# ax.set_xlabel("TEST", fontsize=30)
# ax[2, :].set_ylabel("y", fontsize=15)

# ax[0, 0].axis("equal")

# x = [0, 1]
# ax.plot(x,x)
# ax.annotate("test", xy = (0,0), xytext = (0, 1),
#             xycoords = "data", textcoords = "data",
#             arrowprops=dict(facecolor='black', shrink=0.05))


r = np.random.rand(10, 10)
h = ax.imshow(r)
ax.colorbar(h, loc = "t", length = 0.1, align = "r")
ax.format(abc = 1, titleabove = 0)
fig.show()

plt.show(block=1)

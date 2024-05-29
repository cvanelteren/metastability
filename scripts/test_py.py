import matplotlib.pyplot as plt, cmasher as cmr, pandas as pd,\
    numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy

from itertools import product
from random import choice

n, t = 100, 100
data = np.zeros((n, t))

x, y = int(np.random.rand() * n), int(np.random.rand() * t)
data[x, y] = 1

def update(data):
    x, y = np.where(data > 0)
    idx = np.random.choice(np.arange(x.size))
    x, y = x[idx], y[idx]

    coords = list(product([-1, 0, 1], [-1, 0, 1]))
    xii, yii = choice(coords)
    if x + xii < 0 or y + yii < 0: return
    try:
        change = data[x, y] * 1
        data[x + xii, y + yii] += np.clip(change + data[x + xii, y + yii], 0, 1)
    except Exception as e:
        print(e)
        return

x = np.random.randn(1000)
y = np.random.randn(1000)
fig, ax = plt.subplots()
ax.hexbin(x, y, gridsize = 10)
ax.axis("equal")
fig.show()
plt.show(block = 1)


# from matplotlib.pyplot import pause
# fig, ax = plt.subplots()
# h = ax.hexbin(data, cmap = "inferno", vmin = 0, vmax = 1)
# plt.colorbar(h, ax = ax)
# ti = 0
# while True:
#     update(data)
#     h.set_data(data)
#     ax.set_title(ti)
#     pause(1e-16)
#     ti += 1
# fig.show()
# plt.show(block = 1)

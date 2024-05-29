import cmasher as cmr, pandas as pd,\
    numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy
import matplotlib.pyplot as plt

from sklearn import manifold

fp = "bundling.pkl"
tmp = pd.read_pickle(fp)
for k, v in tmp.items():
    globals()[k] = v


# tsne = manifold.TSNE(init = "pca",
                    # learning_rate = "auto")
n = 3
m = manifold.LocallyLinearEmbedding(n_neighbors = n, n_components = 2, method = "hessian")
m = manifold.Isomap(n_neighbors = n, n_components = 2)
# m = manifold.MDS(n_components = 2, max_iter = 100, n_init = 1)
# m = manifold.SpectralEmbedding(n_neighbors = n, n_components = 2)
#
P  = 5
m = manifold.TSNE(n_components = 2, perplexity = P,
                  # metric = "euclidean",
                  metric = "l1",
                  # metric = "haversine",
                  # metric = "cosine",
                  # random_state = 100,
                  # init = "pca",
                  # angle = .1,
                  learning_rate = "auto",
                  n_jobs = 15)
# tdata = m.fit_transform(states)
# print(tdata.shape)


# idxs = np.array([], dtype = int)
# t = 0.1

# from utils import ccolors
# C = ccolors(12)
# b = np.linspace(0, 1, 11)
# colors = np.zeros((len(states), 4))
# for sidx, s in enumerate(states.mean(1)):
#     idx = np.digitize(s, b)
#     colors[sidx] = C[idx]

import matplotlib.pyplot as pplt
import sklearn
print(sklearn.__version__)

# x, y = tdata.T
# xr = np.linspace(x.min(), x.max(), 300)
# yr = np.linspace(y.min(), y.max(), 300)
# X, Y = np.meshgrid(x, y)

# Z  = np.zeros(X.shape)
# Z.fill(np.nan)
# xy  = np.stack((x, y)).T
# print(xy.shape)
n = len(states)

E_landscape = np.zeros((n, n))
E_landscape.fill(np.nan)
for start, other in allowed.items():
    for y in other:
        E_landscape[start, other] = (E[other].sum() - E[start].sum())

# import proplot as plt
cmap = cmr.pride
cmap.set_bad("k")
print(E_landscape)
fig, ax = pplt.subplots()
ax.imshow(E_landscape, cmap = cmap, interpolation = "none")
fig.show()

plt.show(block = 1)
print(E_landscape)

pos = {}
for start, other in allowed.items():
    for o in other:
        pos[start] =

fig, ax = plt.subplots(projection = "3d")
xr = np.arange(0, n)
X, Y = np.meshgrid(xr, xr)
ax.plot_surface(X, Y, E_landscape, cmap = cmap)
# # ax.scatter(*tdata.T, zs = E.sum(-1), color = colors)
# # ax.plot_surface(X, Y, Z,
# #                 # rstride=1,
#                 # cstride=1,
#                 # cmap="plasma",
#                 # linewidth=0, antialiased=False,
# #                 )
# # # ax.stem(*tdata.T, z = E.sum(-1))
# # ax.imshow(m.dist_matrix_)
# # ax.collections[0].set_color(colors)
# #
# # print(m.embedding_, m.n_features_in_)
# # print(np.unique(m.embedding_, axis = 0).shape)
# # print(m.feature_names_in_)
# # # ax.scatter(*tdata.T)
# # # ax.scatter(*tdata.T, color = colors)




fig.show()
plt.show(block = 1)

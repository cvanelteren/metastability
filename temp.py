import proplot as plt, cmasher as cmr, pandas as pd,\
    numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy

fp = "bundling.pkl"
tmp = pd.read_pickle(fp)
for k, v in tmp.items():
    globals()[k] = v

c = np.asarray(list(pos.values()))
m, mm  = c.min(0), c.max(1)

t = {}
for idx, s in enumerate(states):
    b = s.mean()
    t[b] = t.get(b, []) + [idx]

n = 0
for k, v in t.items():
    if len(v) > n:
        n = len(v)

xr = np.arange(0, n) #- n/2
yr = np.linspace(0, 1, len(t))

X, Y = np.meshgrid(xr, yr)

Z = np.zeros(X.shape)
# Z.fill(np.nan)

ZZ = np.zeros(X.shape)
l = {}
for idx, (k, v) in enumerate(t.items()):
    jdx = np.arange(0, len(v))
    # print(E[v].shape)
    Z[idx, jdx] = np.sort(E[v].sum(-1))


import matplotlib.pyplot as pplt
fig, ax = plt.subplots(projection = '3d')
# nx.draw(G, pos = pos, ax = ax)

# print(X.shape)
l, m = [], []

from utils import ccolors
C = ccolors(10)
for idx, jdx in allowed.items():
    for other in jdx:
        # if idx >= other: break
        target = [idx, other]
        y = states[target].mean(1)

        a, b = y

        a_ = np.array(t[a])
        b_ = np.array(t[b])
        t1 = np.where(a_ == idx)[0]
        t2 = np.where(b_ == other)[0]

        # t1 = a_[t1]
        # t2 = b_[t2]
        x = np.array([t1, t2]).squeeze()
        e = E[target].sum(-1)
        # print(a, b, idx, other, t[a], t[b], t1, t2)
        l.append((x, y, e))
        c = p[idx, other]
        kdx = np.digitize(c, np.linspace(0, .2, 10))
        c = C[kdx]
        m.append(c)
        form = np.ones(x.size) * np.nan

# from matplotlib.collections import LineCollection
# from mpl_toolkits.mplot3d.art3d import Line3DCollection

l = np.array(l)
l = np.moveaxis(l, 1, 2)

# seg = Line3DCollection(l, alpha = 0.05,
                       # )
# ax.add_collection3d(seg, zdir = "z")
# ax.scatter(*l.reshape(-1, 3).T)


# ax.set_ylabel(r"$\langle S \rangle$")
# ax.set_xlabel("")
# ax.set_zlabel("Energy $\mathbb{H}(S)$")

print(Z, X, Y)

cmap = cmr.pride
facecolors=cmap(Z / np.nanmax(Z))
ax.plot_surface(X, Y, Z,
                edgecolor = "none",
                cmap = "warmcool_r",
                # facecolors = facecolors,
                shade = True,
                lw = 0,
                # rstride = 1,
                # cstride = 1,
                # alpha = 1,
                # zorder = 10,
                )

ax.set_ylabel(r"$\langle S \rangle$")
ax.set_zlabel("Energy $\mathbb{H}(S)$")
# seg = LineCollection3D(l)

# n = 20 #
# x = np.random.uniform(0, 1, n)
# y  = np.random.uniform(0, 1, n)
# z = np.arange(0,n)
# points = np.array([x, y, z]).T.reshape(-1, 1, 3)
# segments = np.concatenate([points[:-1], points[1:]], axis=1)
# lc = Line3DCollection(segments)
# lc.set_array(z)
# lc.set_linewidth(2)
# print(segments.shape)
# ax.add_collection3d(lc, zdir='z')

# ax.plot(l[0], l[1], l[2],
#         alpha = .3,
#         marker = 'o')

# ax.scatter(X, Y, Z)

fig.show()
plt.show(block = 1)

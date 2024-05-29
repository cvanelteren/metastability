import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy

from exact import *

fp = "bundling.pkl"
if os.path.exists(fp):
    tmp = pd.read_pickle(fp)
    for k, v in tmp.items():
        globals()[k] = v

else:
    g = nx.krackhardt_kite_graph()
    n = len(g)
    states, allowed = gen_states(len(g))
    A = nx.adjacency_matrix(g).todense()
    E = ising(states, A)
    beta = 0.5732374683235916
    p, p0 = get_transfer(n, E, beta, allowed, states)

    G = nx.from_numpy_matrix(p)

    from fa2 import ForceAtlas2

    pos = np.linspace(0, 2 * np.pi, len(G))
    bins = np.linspace(-1.1 * np.pi, 1.1 * np.pi, len(G) + 1)

    tmp = (states.mean(1) * 2 - 1) * np.pi
    print(tmp.max(), tmp.min())
    idxs = np.digitize(tmp, bins)
    print(idxs.max())
    pos_ = {}
    for idx, b in enumerate(idxs):
        angle = bins[b]
        c = np.array([np.sin(angle), np.cos(angle)])
        pos_[idx] = c

    pos = pos_
    pos = ForceAtlas2(
        # scalingRatio=10,
        # edgeWeightInfluence=1.0,
        # strongGravityMode=True,
    ).forceatlas2_networkx_layout(G)
    # pos = nx.kamada_kawai_layout(G)
    # pos = nx.spring_layout(G)
    # pos = nx.circular_layout(G)
    from plexsim.utils.graph import nx_layout

    nodes, edges, hb = nx_layout(G, pos)

if not os.path.exists(fp):
    data = {
        k: v
        for k, v in dict(globals()).items()
        if k in "g n states allowed A E beta p p0 G pos nodes edges hb".split()
    }
    pd.to_pickle(data, fp)

target = 0.0
targets = np.array([], dtype=object)
node_color = []
from matplotlib import colors

for idx, si in enumerate(states.mean(1)):
    c = "tab:blue"
    if si == target:
        targets = np.append(targets, idx)
        c = "tab:red"
    node_color.append(colors.to_rgb(c))


def walk(starts, G, edges, pos, n_walks=10, t=10, seed=0):
    np.random.seed(seed)
    from random import choice

    hbnp = edges.to_numpy()
    rpos = {tuple(np.asarray(v).round(1)): k for k, v in pos.items()}
    splits = (np.isnan(hbnp[:, 0])).nonzero()[0]
    e = {}
    for start, stop in zip(splits[::2], splits[1::2]):

        a = hbnp[start + 1].round(1)
        b = hbnp[stop - 1].round(1)

        tmp = (rpos[tuple(a)], rpos[tuple(b)])
        e[tmp] = (start + 1, stop - 1)
        e[reversed(tmp)] = reversed(e[tmp])

    # print(e)
    # walks = np.zeros((len(starts), n_walks, t, 2, 2))
    walks = []
    for idx, start in enumerate(starts):
        for walk in range(n_walks):
            current = start
            print(current)
            if current not in G.nodes():
                break
            for ti in range(1, t):
                if current not in list(G.neighbors(current)):
                    break
                next = choice(list(G.neighbors(current)))
                e1_c = pos.get(current)
                e2_c = pos.get(next)

                if (current, next) not in e:
                    print("breaking")
                    break
                start, end = e[(current, next)]
                # start, end = e[(next, current)]

                print(start, end)
                # walks[idx, walk, ti] = hbnp[current:next]
                seg = hbnp[start:end]
                walks.append(seg)

                current = next
    print(walks)
    print("end")
    return np.array(walks)


from matplotlib.collections import LineCollection as LC

# targets = [targets]
# walks = walk(targets, G, hb, pos, 10, t=100)


from utils import ccolors

# N = 10
# np.random.seed(0)
# C = cmr.pride(np.linspace(0, 1, N, 0))
# C = C[np.random.randint(0, N, size=len(walks))]

# segs = LC(walks, color=C, alpha=1, zorder=1)
#
import matplotlib.pyplot as pplt

fig, ax = plt.subplots()
# ax.add_artist(segs)
# hb.plot(x="x", y="y", ax=ax, alpha=0.1, color="k", zorder=0, legend=False)

pos_ = states.mean(1)
p = {}
for idx, si in enumerate(pos_):
    p[si] = p.get(si, []) + [idx]
p_ = {}

node_size = 4
for macro, si in p.items():
    si = np.array(si)
    tmp = np.arange(0, len(si)) - len(si) / 2
    # tmp *= node_size * 10

    # if len(si) > 1:
    #     spacing = abs(np.diff(tmp))[0]
    #     tmp *= 10 * node_size

    for node, sj in enumerate(si):
        p_[sj] = np.array([macro, tmp[node]])

for node in G.nodes():
    G.remove_edge(node, node)
nx.draw_networkx_nodes(G, node_color=node_color, ax=ax, pos=p_, node_size=node_size)
nx.draw_networkx_edges(G, ax=ax, pos=p_, alpha=0.05)
# ax.scatter(pos[0], color="red", s=300)
# ax.axis("equal")
# ax.margins(-0.3)
# ax.axis("on")
ax.tick_params(
    axis="both",
    which="both",
    bottom=True,
    left=True,
    labelbottom=True,
    labelleft=True,
)
ax.set_axis_on()
ax.grid(1)
ax.set_xlabel(r"System macrostate ($\langle S \rangle$)")

# x = np.linspace(0, 0.5)
# ax.axhline(1100, xmax=0.5, ls="solid", c="k")

fig.savefig("test")
fig.show()
plt.show(block=1)

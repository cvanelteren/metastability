import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy

from matplotlib import colors

seed = 12345
seed = 8
# seed = 1234
t = 1 / 0.5732374683235916
np.random.seed(seed)
g = nx.krackhardt_kite_graph()
# m = models.Potts(g, t = t , seed = seed, sampleSize = 1)
m = models.Potts(g, t=t, seed=seed, sampleSize=1)

# s = m.simulate(100000)#[100:]
m.states = 0
s = m.simulate(10000)  # [100:]
idx = np.logical_and(s.mean(1) >= 0.5 - 1 / m.nNodes, s.mean(1) <= 0.5 + 1 / m.nNodes)
tipping = np.where(idx)[0]


def ccolors(n, cmap=cmr.pride):
    return cmap(np.linspace(0, 1, n, 0))


C = ccolors(2)

x = np.arange(s.shape[0])


import matplotlib.pyplot as pplt

layout = [[1, 2], [1, 3]]
fig = plt.figure(
    share=0,
)
ax = fig.add_subplots(
    layout,
)
# fig, ax = plt.subplots(ncols=3, share=0, layout=layout)
ax[0].plot(x, s.mean(1), lw=1)

tmp = np.where(s.mean(1) == 0.5)[0][0] + 1
bounds = (0.1, 0.35, 0.2, 0.2)

lc = ["tab:orange", "tab:purple", "tab:red"]

state_idxs = []  # fetch states that belong to trajectory in a
for idx, si in enumerate([20, 5, 0]):
    b = list(bounds)
    b[1] += (0.2 + 0.01) * idx

    props = dict(edgecolor=lc[idx], alpha=1, lw=1.25)
    inax = ax[0].inset_axes(b, zoom_kw=props)

    # ax.indicate_inset(
    #     bounds,
    # )

    state_idx = tmp - si
    state_idxs.append(x[state_idx])
    inax.set_xlim(x[state_idx], x[state_idx] + 0.01)
    inax.set_ylim(s[state_idx].mean(), s[state_idx].mean() + 0.01)

    rec, indicators = inax.indicate_inset_zoom()
    # rec, indicators = ax.indicate_inset_zoom(inax, **props)
    [indicator.set_visible(0) for indicator in indicators]
    indicators[2].set_visible(1)
    print(inax, indicators)

    # other = inax.inset_axes((0, 0, 1, 1))

    # rec, indicators = inax.indicate_inset_zoom()

    other = ax[0].inset_axes(b, zoom=False)
    p = nx.kamada_kawai_layout(g)
    nx.draw(g, ax=other, node_color=C[s[state_idx].astype(int)], node_size=25, pos=p)

    other.axis("equal")
    other.axis("on")
    other.set_facecolor("none")
    other.grid(0)
    other.margins(0)

    inax.axis("off")
    inax.set_facecolor("none")
    inax.grid(0)


# ax[0].scatter(x[idx], s[idx].mean(1), color = "red")
# ax.axhline(0.5, ls = "dashed")
ax[0].set_xlabel("Time ($t$)")
ax[0].plot(
    x[0:tmp], np.ones(tmp) * -0.05, ls="solid", color="k", label="Ground state 0"
)

from labellines import labelLine, labelLines

ax[0].plot(
    x[tmp:], np.ones(x[tmp:].size) * 1.05, ls="solid", color="k", label="Ground state 1"
)


x1 = (x[tmp] // 2, -0.10)
x2 = (x[tmp] + (x.max() - x[tmp]) // 2, 1.10)
ax[0].annotate(r"$\tau_1$", x1, transform=ax.transData, ha="center", va="center")
ax[0].annotate(r"$\tau_2$", x2, transform=ax.transData, ha="center", va="center")
ax[0].set_ylim(-0.15, 1.15)
ax[0].set_xlim(0, x[-1] + 100)

from matplotlib.pyplot import Line2D

h = [
    Line2D([], [], color=C[0], label="0", marker="o", ls="none"),
    Line2D([], [], color=C[1], label="1", marker="o", ls="none"),
]
ax[0].legend(handles=h, loc="lr", title="Agent state")

ax[0].set_ylabel(r"Instantaneous system macrostate $(\langle S^t \rangle)$")

from iFlow.exact import ising, gen_states
from figure1 import show_bistability

import cupy as cp

states, _ = gen_states(len(g))
A = nx.adjacency_matrix(g).todense()
E = ising(states, A).sum(1)
beta = 0.3
P = np.exp(-beta * E) / np.exp(-beta * E).sum()

n = len(g)
bins = np.linspace(0 - 1 / (2 * n), 1 + 1 / (2 * n), n + 2)
mu = states.mean(1)

counts = {}
E_plot = {}  # fetch energies
dp = np.diff(bins)[0]
d = {}
c = {}

tmp = {}
for S, e, p in zip(mu, E, P):
    idx = np.digitize(S, bins)
    b = bins[idx]
    counts[b] = counts.get(b, 0) + p
    E_plot[b] = E_plot.get(b, 0) + e
    tmp[b] = tmp.get(b, 0) + 1
    d[b] = d.get(b, 0) + e
    c[b] = c.get(b, 0) + 1
x = np.asarray(list(counts.keys()))
x -= 1 / (2 * n)  # correct for binning
y = np.asarray(list(counts.values()))


show_bistability(x, y, ax[2])
twinx = ax[2].twinx()
tmp_e = np.asarray(list(E_plot.values())) / np.asarray(list(tmp.values()))
twinx.plot(x, tmp_e, color="k", ls="--")
twinx.set_ylabel(r"Energy ($\langle \mathbb{H}(S) \rangle$)")


# px = ax[0].panel("r", share = 0, pad = 0)
# t = np.where(s.mean(1) == 1)[0]
# print(t, state_idxs)
# idx = np.arange(state_idxs[0], t[0] + 1)
# Y = s[idx].mean(1)
# Z = np.array([m.siteEnergy(i) for i in s[idx]]).sum(-1)

# l = np.linspace(0, 1, 11)
# k = np.zeros(l.size)
# for kdx, li in enumerate(l):
#     t = np.where(Y == li)[0]
#     if t.size:
#         k[kdx] = np.nanmean(Z[t])
#     else:
#         k[kdx] = Z[kdx]

# print(l, k)
# # px.plot(Z, Y, marker = "o")
# # px.barh(l, k)
# # px.plot(k, l)
# # px.invert_xaxis()

# yy = np.asarray(list(d.values()))
# yyy = np.asarray(list(c.values()))
# px.plot(yy / yyy, x)
# px.set_xlabel("Energy $\mathbb{H}(S^t)$")
# px.set_ylim(-0.15, 1.15)

fp = "bundling.pkl"
if os.path.exists(fp):
    tmp = pd.read_pickle(fp)
    for k, v in tmp.items():
        globals()[k] = v


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

# # targets = [targets]
# walks = walk(targets, G, hb, pos, 10, t=100)


# from utils import ccolors

# N = 10
# np.random.seed(0)
# C = cmr.pride(np.linspace(0, 1, N, 0))
# C = C[np.random.randint(0, N, size=len(walks))]


# segs = LC(walks, color=C, alpha=1, zorder=1)
# ax[2].add_artist(segs)
# # hb.plot(x="x", y="y", ax=ax[2], alpha=0.1, color="k", zorder=0, legend=False)
# nx.draw_networkx_nodes(G, node_color=node_color, ax=ax[2], pos=pos, node_size=7)
# # nx.draw_networkx_edges(G, ax=ax, pos=pos)
# # ax.scatter(pos[0], color="red", s=300)
# ax[2].axis("equal")
# ax[2].axis("off")
# # ax.margins(-0.3)
# ax[2].grid(0)
# ax[2].set_xlabel("")


# convert state to state transition in to a graph
pos_ = states.mean(1)
p__ = {}
for idx, si in enumerate(pos_):
    p__[si] = p__.get(si, []) + [idx]
p_ = {}

node_size = 2
node_sizes = np.zeros(len(G))
seen = set()

mapped_coordinate_colors = {}
for macro, si in p__.items():
    si = np.array(si)
    # tmp = np.linspace(0, 1, len(si))
    # if len(si):
    # tmp -= 0.5
    tmp = np.arange(0, len(si)) - len(si) / 2
    node_sizes[si] = 0.95 * 1 / len(si)

    # convert to gaussian coordinates
    tmp_p = P[si]  # index into "prob"
    p_gauss = np.exp(-((tmp_p - tmp_p.max()) ** 2))
    p_gauss = np.argsort(p_gauss)

    center = len(si) / 2
    count = 0
    for sign, idx in enumerate(p_gauss[::-1]):
        sign = -1 if idx % 2 == 0 else 1
        pos_node = np.round((center + count * sign), 0).astype(int)
        pos_node = np.clip(pos_node, 0, len(si) - 1)
        pos_node = tmp[pos_node]
        sj = si[idx]
        from copy import deepcopy

        p_[sj] = np.array([macro, deepcopy(pos_node)])
        mapped_coordinate_colors[sj] = p_[sj]

        # increase count to the other side
        if idx % 2 == 0:
            count += 1
    # for node, sj in enumerate(si):
    #     p_[sj] = np.array([macro, tmp[node]])
    # mapped_coordinate_colors[sj] = p_[sj]

target = 0.0
targets = np.array([], dtype=object)
node_color = []

from iFlow.exact import to_binary

targets = np.array([to_binary(s[i]) for i in state_idxs], dtype=int)


N = 100
# percs = np.percentile(P, np.linspace(0, 100, N))
percs = np.geomspace(P.min(), P.max(), N)
# percs = np.linspace(0, P.max(), N)
# percs = np.insert(percs, 0, 0)
print(percs)

from matplotlib.pyplot import cm

cmap = cm.inferno
C = ccolors(N + 2, cmap)
node_color_targets = []
L = set()

for state, state_pos in mapped_coordinate_colors.items():
    # for idx, si in enumerate(states.mean(1)):
    tmp = np.digitize(P[state], percs)
    print(state, tmp, P[state])
    c = C[tmp]
    if state in targets:
        # if si == target:
        # targets = np.append(targets, idx)
        c = lc[int(np.where(targets == state)[0])]
        c = np.array(colors.to_rgba(c))
        node_color_targets.append(c)
        # c = "tab:red"
        #
    node_color.append(c)
    # node_color.append(colors.to_rgb(c))


from fa2 import ForceAtlas2

# for i, j in G.edges():
# G[i][j]["weight"] = 1
# p__ = ForceAtlas2(scalingRatio=2000,
# gravity=10.0, edgeWeightInfluence = 0.0).forceatlas2_networkx_layout(G, pos = p_,
# iterations = 100)
# p__ = nx.kamada_kawai_layout(G, pos = p_)
# p_ = {k: np.array([v[0], p__[k][1]]) for k, v in p_.items()}
# p_ = {k: np.array([p__[k][0], p__[k][1]]) for k, v in p_.items()}


node_color = np.stack(node_color).squeeze()
print(node_color)
for node in G.nodes():
    G.remove_edge(node, node)
walk = []
for current in np.arange(state_idxs[0], state_idxs[-1] + 1):
    current = to_binary(s[current])
    walk.append(p_[current])

walk = np.array(walk).T


ec = []

N = 1000

np.fill_diagonal(p, 0)
tmp_c = np.linspace(0, p.max() * 2, N)
print(p.max())
C = colors.ListedColormap("greys", 10)
C = ccolors(N, cmr.ember)
for i, j in G.edges():
    pi = p[i, j]  # + p[j, i]
    ec.append(C[np.digitize(pi, tmp_c)])


inax = ax[1]
# inax = ax[2].inset_axes((0,0,1, 1), zoom = False)
inax.plot(walk[0], walk[1], color="tab:red", zorder=1, alpha=0.8)

nx.draw_networkx_nodes(
    G,
    node_color=node_color_targets,
    ax=inax,
    pos=p_,
    node_size=node_size * 10,
    nodelist=targets,
    cmap="tab:default",
)


nx.draw_networkx_edges(G, ax=inax, pos=p_, alpha=0.025, edge_color="gray")

l = np.array(list(p_.values()))

# node_sizes = np.arange(node_sizes.size)
tmp = np.asarray(list(mapped_coordinate_colors.values()))
# plot p-value in network representation
inax.scatter(*tmp.T, c=node_color, alpha=1, s=4, cmap=cmap)

inax.tick_params(
    axis="both",
    which="both",
    bottom=True,
    # left=True,
    # labelbottom=True,
    # labelleft=True,
)

norm = colors.Normalize(vmin=0, vmax=P.max().round(2))
h = cm.ScalarMappable(cmap=cmap, norm=norm)

inax.colorbar(
    h,
    loc="t",
    align="r",
    title="$P(S)$",
    length=0.25,
    width=0.1,
    pad=-2,
    # frameon = 0
)

# inax.set_ylabel("System state ($S$)")

# inax.grid(1)
# for spine in "top right".split():
#     inax.spines[spine].set_visible(False)
#     ax[0].spines[spine].set_visible(False)
#     ax[1].spines[spine].set_visible(False)
#     ax[2].spines[spine].set_visible(False)
# inax.spines["left"].set_visible(0)

from matplotlib import patches

style = "Simple, head_length=1, head_width=1, tail_width=1, tail_width=30"
style = patches.ArrowStyle(style)
style = "->"

x1 = (0, 140)
x2 = (0.5, 140)
x3 = (1, 140)

x4 = (0.0, -140)
x5 = (0.5, -140)
x6 = (1.0, -140)

arrow = dict(
    posA=x1,
    posB=x2,
    mutation_scale=10,
    color="tab:red",
)
sett = [
    (x1, x2, "tab:red"),
    (x3, x2, "tab:red"),
    (x5, x4, "tab:green"),
    (x5, x6, "tab:green"),
]

for (a, b, c) in sett:
    arrow["posA"] = a
    arrow["posB"] = b
    arrow["color"] = c
    p = patches.FancyArrowPatch(**arrow)
    inax.add_patch(p)

xy = p_[0].copy()
xy[1] = -105
print(xy, p_[0])
inax.annotate(
    "$S_i \in S$",
    xytext=xy,
    xy=p_[0],
    xycoords="data",
    textcoords="data",
    ha="center",
    va="center",
    arrowprops=dict(
        facecolor="black",
        # mutation_scale=10,
        arrowstyle="-|>",
        # connectionstyle = "arc3,rad=0.5"
    ),
)
inax.annotate(
    "Stabilizing dynamics",
    (0.5, -150),
    transform=ax.transData,
    ha="center",
    va="top",
)
inax.annotate(
    "Destabilizing dynamics",
    (0.5, 150),
    transform=ax.transData,
    ha="center",
    va="bottom",
)
inax.set_xlim(-0.09, 1.09)

inax.set_ylim(-200, 200)

# ax[2].set_xlabel(r"System macrostate ($\langle S \rangle$)")

# fig.auto_layout(resize = 0, tight = 0, aspect = 0)
#
fig.patch.set_alpha(0.0)
fig.savefig("./figures/figure1_alt.png")
fig.show()
# plt.show(block=1)
# tmp = np.asarray(list(p_.values()))
# x, y = tmp.T
# print(x.min(), x.max(), y.min(), y.max())
# xr = np.linspace(x.min(), x.max(), 11)
# yr = np.linspace(y.min(), y.max(), 50)

# X, Y = np.meshgrid(xr, yr)
# Z = np.zeros(X.shape)
# Z.fill(np.nan)
# for sidx, s in enumerate(states.mean(1)):
#     # get coordinate
#     xi, yi = p_[sidx]
#     # bin the coordinate
#     xdx = np.argmin( abs(xr - xi))
#     ydx = np.argmin( abs(yr - yi))
#     # Z[xdx, ydx] += 1
#     if np.isnan(Z[ydx, xdx]):
#         Z[ydx, xdx] = 0
#     Z[ydx, xdx] += np.log(p0[sidx])
# z = np.ma.masked_invalid(Z)
# print(Z.max())
# # Z = np.ma.masked_equal(Z,np.nan)
# inax = ax[2].inset_axes((0, 0, 1, 1), zoom = 0, projection = "3d")
# cmap = cm.coolwarm

# from matplotlib import colors
# normC = colors.Normalize(vmax=z.max(), vmin=z.min())
# C = pplt.get_cmap("plasma")(normC(z))
# # cmap.set_bad("grey")
# print(Z)
# inax.plot_surface(X, Y, z, cmap = cmap, antialiased = False, facecolors = C)

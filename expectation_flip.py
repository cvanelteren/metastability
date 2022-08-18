import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy


from exact import *

beta = 0.5732374683235916
g = nx.krackhardt_kite_graph()
n = len(g)
states, allowed = gen_states(n)

bins = np.linspace(0, 1 + 2 / n, n + 2)
print(bins)

A = nx.adjacency_matrix(g).todense()
E = ising(states, A)
p, p0 = get_transfer(n, E, beta, allowed, states)

binned = np.digitize(states.mean(1), bins)
idxs = np.arange(0, bins.size)

p_node = np.zeros((bins.size, n, 2))
for i in idxs:
    targets = np.where(binned == i)[0]
    tmp = np.zeros(p0.size)
    tmp[targets] = p0[targets]
    tmp /= tmp.sum()
    for target in targets:
        for to_state in allowed[target]:
            if to_state != target:
                difference = np.where(states[to_state] - states[target])[0]
                p_node[i, difference, 1] += p[target, to_state] * p0[target]
p_node[..., 0] = 1 - p_node[..., 1]


print(p_node)
pos = nx.kamada_kawai_layout(g)
from utils import ccolors

C = ccolors(n)
N = bins.size // 2
fig, ax = plt.subplots(ncols=N - 1, share=1)
for idx, bin in enumerate(bins[1:N]):
    axi = ax[idx]
    s = p_node[idx, :, 1]
    s /= s.max()
    inax = axi.inset_axes((0, 0, 1, 1), zoom=0)
    nx.draw(g, ax=inax, node_color=C, node_size=s * 500, pos=pos)
    inax.axis("equal")
    axi.axis("equal")
    axi.set_title(round(bin, 1))
    axi.axis("off")
fig.subplots_adjust(wspace=0)
fig.suptitle("Fraction of nodes +1")
fig.savefig("./figures/expectation_flip.pdf")
fig.show()
# plt.show(block=1)

import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy


def plot_st_compl(T, temps, H, D, ax=None):
    from sklearn.preprocessing import minmax_scale

    C = H * D
    D = minmax_scale(D)
    C = minmax_scale(C)
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(temps, C, label="$C$")
    ax.plot(temps, H, label="$\overline{H}(S)$")
    ax.plot(temps, D, label="$D(S)$")
    ax.legend(loc="cr", ncols=1)
    # ax.legend(loc="ur", )
    ax.axvline(T, color = 'k', linestyle = 'dashed')
    ax.format(xlabel="Temperature (T)")


import re

fp = "exact_kite_dyn=ising_beta=0.5732374683235916_T=200.pkl"
s = fp.rstrip(".pkl")

t = 1 / float(re.findall("beta=\d+.\d+", fp)[0].split("=")[1])
df = pd.read_pickle(f"data/{fp}")

p0 = df.p0[df.mag == -1].iloc[0]
from exact import gen_states, ising

g = df.attrs["settings"].g
n = len(g)
states = gen_states(n)[0].get()
bins = np.linspace(0 - 1 / n, 1 + 1 / n, n + 2)
idx = np.digitize(states.mean(-1), bins)
vals = np.zeros(bins.size)
for idx, i in enumerate(idx):
    vals[i] += p0[idx]

b = (bins[:-1] + bins[1:]) / 2


from exact_utils import get_p_gibbs

temps = np.linspace(0.1, 10, 1000)
A = nx.adjacency_matrix(g).todense()
E = ising(states, A).get().squeeze().sum(-1)
p = np.array([get_p_gibbs(E, 1 / t if t > 0 else np.inf) for t in temps])
print(p[0])

H = -np.nansum(np.log2(p) * p, axis=-1) / len(g)
D = np.nansum((p - 1 / p.shape[1]) ** 2, axis=-1)


fig, ax = plt.subplots(nrows=2, share=0, abc=1)

plot_st_compl(t, temps, H, D, ax=ax[0])
ax[1].bar(bins, vals)
ax[1].format(xlabel="Fraction of nodes in +1", ylabel="P(S)")
fig.savefig(f"./figures/{s}_statistical_complexity.png")
plt.show(block=1)

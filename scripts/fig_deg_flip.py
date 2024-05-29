import proplot as pplt, cmasher as cmr, pandas as pd
import matplotlib.pyplot as plt
import numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy

# helper functions
def p(x: float, b: float):
    # compute transition probability
    return 1 / (1 + np.exp(b * 2 * x))


from scipy.special import comb


def p_switch(frac: float, k: int, t: float):
    assert 0 <= frac <= 1
    result = 0
    b = 1 / t
    for ki in range(k + 1):
        x = ki - (k - ki)  # number of aligned spins - not aligned spins
        result += comb(k, ki) * frac**ki * (1 - frac) ** (k - ki) * p(x, b)

    mag = -np.nansum((frac * np.log2(frac), (1 - frac) * np.log2(1 - frac)))
    mag = frac
    # mag = frac
    # mag = (1 - frac) - frac
    # print(f"Magnetization is {mag}")
    return mag, result


# generate data
t = 0.5
k = np.linspace(1, 10, 10).astype(int)
fracs = np.linspace(0, 1)  # fraction of nodes that have -1
res = np.zeros((2, fracs.size, k.size))
for jdx, frac in enumerate(fracs):
    for idx, ki in enumerate(k):
        res[:, jdx, idx] = p_switch(frac, ki, t)

# start plot
fig, ax = pplt.subplots(dpi=300)
colors = cmr.pride(np.linspace(0, 1, k.size, 0))
handles = []
print(res.shape)
for idx, ki in enumerate(k):
    x, y = res[:, ::-1, idx]
    ax.plot(x, y, color=colors[idx])
    h = plt.Line2D([], [], color=colors[idx], marker="o", linestyle="none", label=ki)
    handles.append(h)
ax.legend(handles=handles, title="Degree (k)", facecolor="white", loc="r", ncol=1)

ax.axvline(0.5, color="gray", linestyle="dashed", lw=3)
ax.annotate(
    "Tipping point",
    (0.5, 0.6),
    xycoords="data",
    va="center",
    ha="center",
    bbox=dict(boxstyle="round", fc="white", ec="gray", lw=2),
)
# ax.set_xlim(0, 1.2)
# ax.set_ylim(0, 0.7)
ax.grid()
fig.format(facecolor="white")
x1 = "System magnetization $M(S^t)$"
x1 = "System entropy $H(S)$"
x1 = "Neighborhood entropy $H(N)$"
x1 = "Fraction of nodes being in +1"

x2 = "$E[p(s_i = +1 | N)]$"
ax.set_xlabel(x1)
ax.set_ylabel(x2)
fig.savefig("./figures/fig_majority_flip.pdf", dpi=300, transparent=1)
fig.show()

plt.show(block=1)

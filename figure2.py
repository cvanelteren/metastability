import warnings

warnings.filterwarnings("ignore")
import cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
import matplotlib.pyplot as plt

# import proplot as plt,
from plexsim import models
from imi import infcy
from pathlib import Path
import re

base = Path("./data")
fp = "exact_kite_dyn=ising_beta=0.5732374683235916_T=200.pkl"
fp = "exact_kite_dyn=ising_beta=0.5732374683235916_T=100_g.is_directed()=False.pkl"
fp='exact_kite_dyn=ising_beta=0.5732374683235916_T=2000_g.is_directed()=False.pkl'
fp='exact_kite_dyn=ising_beta=0.5732374683235916_T=310_g.is_directed()=False.pkl'
p='exact_kite_dyn=ising_beta=0.5732374683235916_T=310_g.is_directed()=False.pkl'
fp='exact_kite_dyn=ising_beta=0.5732374683235916_T=310_g.is_directed()=False.pkl'
fp='exact_kite_dyn=ising_beta=0.5732374683235916_T=200_g.is_directed()=False.pkl'
fp ="exact_kite_dyn=sis_beta=9.122104355448496_T=200.pkl"
fp = "exact_kite_dyn=ising_beta=1.0_T=200.pkl"
fp = "exact_kite_dyn=ising_beta=0.5732374683235916_T=200.pkl"
fp = "exact_kite_dyn=ising_beta=0.5732374683235916_T=2000.pkl"
fp='exact_kite_dyn=ising_beta=0.5732374683235916_T=2000_g.is_directed()=False.pkl'
fp='exact_kite_0_dyn=ising_beta=0.5732374683235916_T=300_g.is_directed()=False_NEW.pkl'
fp='exact_kite_0_dyn=ising_beta=0.5732374683235916_T=300_g.is_directed()=False_NEW.pkl'
fp='exact_kite_0_dyn=ising_beta=0.5732374683235916_T=300_g.is_directed()=False_NEW.pkl'

fp='exact_kite_0_dyn=ising_beta=0.5732374683235916_T=300_g.is_directed()=False_NEW.pkl'
# fp = (
# "exact_florentine_dyn=ising_beta=0.5732374683235916_T=310_g.is_directed()=False.pkl"
# )

from exact_utils import fit_curve

df = pd.read_pickle(base / Path(fp))
f = lambda x, a, b, c, d, g: a * np.exp(-b * x) + c * np.exp(-d * x) + g
# f = lambda x, a, b, c, d, g: a * np.exp(-b * x) + g
ndf = fit_curve(df, f, offset=True)


import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy

from sklearn.preprocessing import minmax_scale
from utils import ccolors

print(ndf.columns)
imi = np.stack(ndf.imi)
asy = np.stack(ndf.asymp)
mi = np.stack(ndf.mi)

# fig, ax = plt.subplots()
# h = ax.imshow(asy[:-1])
# ax.colorbar(h)
# fig.show()
# plt.show(block = 1)


z_imi = imi.max()
z_ai = asy.max()

h = np.stack(ndf.h)
mag = ndf.mag.unique() - 0.05
imi = imi[:-1]
# imi = (imi - imi.min(1)[:, None]) / (imi.max(1)[:, None] - imi.min(1)[:, None])
g = df.attrs["settings"].g

targets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
# targets =  [ 0.0125,  0.1375,  0.2625,  0.3875,  0.5125, 0.6375 ]
pos = nx.kamada_kawai_layout(g)

# n = len(targets)
n = 5

from matplotlib.collections import LineCollection as LC

c = ccolors(len(g))

import matplotlib.pyplot as pplt

fig, axs = plt.subplots(
    nrows=3,
    ncols=n,
    # sharex = 'row',
    # sharey = 'row',
    share=1,
    # span="row",
    # height_ratios = [0.5, 1],
    # figsize = (5,5),
    # constrained_layout = True
)

fs = 32

plot_mag = h[:, 0].sum(-1)
tits = []
pos_ = np.array([i for i in pos.values()])


for ax, axj, axk, t in zip(axs[0, :], axs[1, :], axs[2, :], targets):
    lc = LC([[pos[x], pos[y]] for x, y in g.edges()], color="k", zorder=0)
    ax.add_artist(lc)
    ax.axis("off")
    ax.axis("equal")

    lc = LC([[pos[x], pos[y]] for x, y in g.edges()], color="k", zorder=0)
    axj.add_artist(lc)

    axj.axis("off")
    axj.axis("equal")

    # print(t, np.where(mag == t), mag == t, mag)
    idx = np.where(mag.round(2) == round(t, 2))[0][0]
    f = mag[idx]
    tit = f  # - np.nansum((f * np.log2(f), (1 - f) * np.log2( 1 - f)))
    tits.append(tit)
    # ax.set_title(f"{tit:.2f}",
    # y = 1.3,
    # fontsize = 8,
    # )

    for node, (ci, x) in enumerate(zip(c, pos.values())):
        s = imi[idx]
        s /= s.max()
        # s /= z_imi
        print(s.max(), z_imi)
        # s[s < 1e-3] = 0
        ax.scatter(*x, s=s[node] * 600, color=ci)

        s = asy[idx]
        print(">", s.max(), s.min(), z_ai)
        # s /= s.max()
        s /= z_ai
        axj.scatter(*x, s=s[node] * 600, color=ci)

        # axk.scatter(asy[idx], imi[idx], color = c)
        axk.plot(mi[idx, :, node], color=ci)
        axk.axvline(imi[idx, node], color = ci)
        print(node, imi[idx, node], imi.shape)
        # axk.set_xscale("log")
        # axk.set_yscale("log")

    # axk.scatter(asy[idx], imi[idx], color = c, s = 500)
    # axk.axis("equal")

# axs.format(fontsize = fs)
# axs.axis('off')
#
S = 30

axs[2, :].format(
    # xlim=(0, 100),
    fontsize=0.6 * S,
    # yscale = "log",
    # xscale = "log",
)

# axs[2, :].format(yscale = "log", xscale = "log")
axk.set_ylabel(r"$I(s_i : S^t | \langle S \rangle)$", fontsize=0.8 * S)
# axk.set_xlabel("Time(t)", fontsize=0.8 * S)

# axs[2,:].set_xlabel("Asymptotic information")
# axk.set_ylabel("Integrated mutual\ninformation")

fig.suptitle(
    "System Stability\nFractions of nodes in +1",
    fontsize=24,
)

dyn = re.findall("dyn=[a-z]+", fp)[0].split("=")[1]

axs[0, 0].format(abc=1, abcloc="l", abc_kw=dict(text="a", fontsize=S))
axs[1, 0].format(abc=1, abcloc="l", abc_kw=dict(text="b", fontsize=S))
axs[2, 0].format(abc=1, abcloc="l", abc_kw=dict(text="c", fontsize=S))

for idx, t in enumerate(tits):
    axs[0, idx].set_title(round(t, 2), y=0.95, fontsize=0.9 * S)

# axs[2, :].set_xlabel("Time(t)", fontsize=400000000000000)
xy = (0.5, -0.0)
axs[2, 2].annotate("Time(t)", xy, fontsize=0.8 * S, xycoords="figure fraction")

# fig.savefig(f"./figures/{dyn}_kite_graph.pdf")
# fig.savefig("./figures/figure2.pdf")
fig.savefig("test")
# plt.show(block = 1)

# exit()

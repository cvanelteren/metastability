import warnings
warnings.filterwarnings('ignore')
import proplot as plt, cmasher as cmr, pandas as pd,\
    numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy
from pathlib import Path
import re

base = Path("./data")
fp = "exact_kite_dyn=ising_beta=0.5732374683235916_T=200.pkl"
fp='exact_kite_0_dyn=ising_beta=0.5732374683235916_T=300_g.is_directed()=False_NEW.pkl'

from exact_utils import fit_curve
df = pd.read_pickle(base / Path(fp))
f = lambda x, a, b, c, d, g: a * np.exp(-b * x) +  c * np.exp(-d * x) + g
# f = lambda x, a, b, c, d, g: a * np.exp(-b * x) +  g
ndf = fit_curve(df, f, offset = True)
#+end_src

imi = np.stack(ndf.imi)
ai = np.stack(ndf.asymp)
mi = np.stack(ndf.mi)
from utils import ccolors
C = ccolors(10)

g = nx.krackhardt_kite_graph()
pos = nx.kamada_kawai_layout(g)
mi = np.stack(ndf.mi)
mag = (np.asarray(ndf.mag.copy()) - 0.05).round(1)[:-1]
print(mag)
targets = np.linspace(0.1, 0.5, 5)


layout = [
          [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
          [6, 6, 6, 6, 6, 7, 7, 7, 7, 7]
]

# layout = [[3, 4, 5, 6, 7],
          # [1, 1, 1, 2, 2]]


fig = plt.figure(share = 0, spanx = 1, refwidth='10cm')
fig, ax = plt.subplots(layout, refnum = 5, share = 0, spanx = 1, refwidth = "10cm")

tmp = np.stack((imi, ai))[:, :-1]

for node_imi, c in zip(tmp.T, C):
    ax[5].plot(mag, node_imi[...,0], color = c, marker = 'o', alpha = 1)
    ax[6].plot(mag, node_imi[...,1], color = c, marker = 'o', alpha = 1)


for target, axi in zip(targets, ax[0, :]):
    idx = np.argmin( abs(mag  - target))
    for c, node in zip(C, mi[idx].T):
        axi.plot(node, color = c)

    title = ""
    # if target == 0.1:
        # title = "Ground state\n"
    if target == 0.5:
        title = "Tipping point\n"
    title += f"$\\langle S\\rangle$ = {round(target, 1)}"
    axi.set_title(title)
    # inax = axi.inset_axes((0.5, 0.4, 0.5, 0.6), zoom = 0)

    s = tmp[0, idx].copy()
    s /= s.max()


    # if target == 0.1:
    #     inax.set_title("Integrated mutual information")

    ymin = tmp[0, idx].min()
    ymax = tmp[0, idx].max()

    y_offset = ymax / tmp[0, :-1].max()
    y_offset = np.clip(y_offset + 0.01, 0, 1)
    y_offset  = 0.75
    b = (target - .15, target + 0.24, 0.25, 0.25)
    inax = ax[5].inset_axes(b, zoom = 1)
    rec, indicators = inax.indicate_inset_zoom()
    [indicator.set_visible(0) for indicator in indicators]
    # indicators[1].set_visible(1)
    # indicators[0].set_visible(1)
    # indicators[2].set_visible(1)
    inax.axis("off")

    xlim = (target - 0.025, target + 0.025)
    ylim = (ymin * 0.90, ymax * 1.10)
    inax.format(xlim = xlim, ylim = ylim)

    inax = ax[5].inset_axes(b, zoom = 0)

    nx.draw_networkx_edges(g, pos, ax = inax, alpha = 0.2)
    l = np.asarray(list(pos.values())).T
    inax.scatter(*l, c = C, s = s*60)
    inax.axis("equal")
    inax.set_facecolor("none")
    inax.grid(0)
    inax.axis('off')
    inax.margins(0)


    s = tmp[1, idx].copy()
    s /= tmp[1].max()

    ymin = tmp[1, idx].min()
    ymax = tmp[1, idx].max()

    y_offset = ymax / tmp[0, :-1].max()
    y_offset = np.clip(y_offset + 0.01, 0, 1)
    y_offset  = 0.75
    b = (target - .15, target + 0.24, 0.25, 0.25)
    inax = ax[6].inset_axes(b, zoom = 1)
    rec, indicators = inax.indicate_inset_zoom()
    [indicator.set_visible(0) for indicator in indicators]
    # indicators[1].set_visible(1)
    # indicators[0].set_visible(1)
    # indicators[2].set_visible(1)
    inax.axis("off")

    xlim = (target - 0.025, target + 0.025)
    ylim = (ymin * 0.90, ymax * 1.10)
    inax.format(xlim = xlim, ylim = ylim)

    inax = ax[6].inset_axes(b, zoom = 0)

    nx.draw_networkx_edges(g, pos, ax = inax, alpha = 0.2)
    l = np.asarray(list(pos.values())).T
    inax.scatter(*l, c = C, s = s.round(3)*60)
    inax.axis("equal")
    inax.set_facecolor("none")
    inax.grid(0)
    inax.axis('off')
    inax.margins(0)



ax[5].set_ylim(-1, 20)
ax[6].set_ylim(-0.05, 0.75)
ax[0, :].format(xlim = (0, 150), xlabel = "Time ($t$)")
ax[0, 0].format(ylabel = "Information flow\n$I(s_i : S^{\\tau + t }| \\langle S^{\\tau} \\rangle)$")
ax.format(fontsize = 21, abc = True, titleabove = False)

ax[6].set_ylabel("Asymptotic information ($\omega$)")
ax[5].set_ylabel("Integrated mutual information\n($\mu(s_i)$)")
ax[1,:].format(xlabel = "System macrostate ($\\langle S \\rangle $)")
fig.suptitle("Fraction of nodes in state +1",
             fontsize = 32)
# fig.savefig("./figures/figure2_alt.pdf")
fig.savefig("test")
fig.show()

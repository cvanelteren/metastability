import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy
from figure4 import *


def get_category(row):
    g = graphs[int(row.id)]
    node = row.label

    # orc = OllivierRicci(g, alpha=0.5, verbose="INFO")
    # orc.compute_ricci_curvature()
    if node == "control":
        deg = -1
    else:
        # deg = bins[np.digitize(nx.pagerank(g)[node], bins)]
        deg = g.degree(node)
        # deg = bins[np.digitize(g.nodes[node]["ricciCurvature"], bins)]
        # deg = bins[np.digitize(g.nodes[node]["ricciFlow"], bins)]
        # deg = orc.G.nodes[node]["ricciCurvature"]
    return deg


def norm(row):

    id = row.id
    control = errors.query("id == @id & label == 'control' & seed == @row.seed").iloc[0]

    c1 = row.less_n / control.less_n
    c2 = row.less_w / control.less_w
    c3 = row.greater_n / control.greater_n
    c4 = row.greater_w / control.greater_w

    if row.label == "control":
        c1 = c3
        c2 = c4

    # print(row.label)
    # print("X>", row.less_w, row.less_n)
    # print(">", control.less_w, control.less_n)
    # print(">>", control.greater_n, control.greater_w)
    # print(c1, c2, c3, c4, )

    return c1, c2, c3, c4


fp = "combined_errors.pkl"
errors = pd.read_pickle(fp)
graphs = pd.read_pickle("graphs_seed=0.pkl")
graphs = pd.read_pickle("org_graphs.pkl")
roles = pd.read_pickle("graphs_seed=0_roles.pkl")
bins = np.linspace(-1, 1, 21)

# print(errors.nudge.unique())

e = np.stack(errors["success"])
e = e[:, 0] / e[:, 1]
errors["success"] = e
# errors = errors.groupby("id label".split()).agg(np.nanmean).reset_index()
# errors = pd.DataFrame(errors)
errors["deg"] = errors.apply(get_category, axis=1)
# errors["nudge"] = np.inf

try:
    errors = pd.read_pickle("reweighted.pkl")
except:
    tmp = np.array([i for i in errors.apply(norm, axis=1).values])
    errors["c_less_n"] = tmp[:, 0]
    errors["c_less_w"] = tmp[:, 1]
    errors["c_greater_n"] = tmp[:, 2]
    errors["c_greater_w"] = tmp[:, 3]

    tmp = []
    for idx, row in errors.iterrows():
        if row.label != "control":
            role = roles.query("id == @row.id").roles.iloc[0]
            tmp.append(role[int(row.label)])
        else:
            tmp.append(np.nan)
    errors["role"] = tmp
    errors.to_pickle("reweighted.pkl")


bins = np.linspace(-1.1, 1.1, 31)
binned = np.digitize(errors.role, bins) - 1

errors["grouped"] = binned.astype(int)

print(errors["role"])
print(errors["deg"].unique())
print(errors.role.max(), errors.role.min())


from utils import ccolors

cmap = cmr.prinsenvlag_r
cmap = cmr.redshift
cmap = cmr.infinity_s
cmap = cmr.guppy_r
# cmap = cmr.fusion_r
# cmap = cmr.pride
C = cmap(np.linspace(0, 1, bins.size, 0))
print(errors.grouped)
a = 0.3

cc = C[errors.grouped]


s = 10
print(errors[errors.label == "control"].grouped)
idx = np.unique(np.where(errors[errors.label == "control"])[0])

errors["c_greater_n"].iloc[idx] = np.nan
errors["c_greater_w"].iloc[idx] = np.nan
errors["c_less_n"].iloc[idx] = np.nan
errors["c_less_w"].iloc[idx] = np.nan


import matplotlib.pyplot as pplt

fig, ax = plt.subplots(journal="nat1")
ax.scatter(
    errors["c_greater_n"],
    errors["c_greater_w"],
    c=cc,
    alpha=a,
    s=s,
    marker="s",
)

ax.scatter(
    errors["c_less_n"],
    errors["c_less_w"],
    c=cc,
    alpha=a,
    s=s,
)
from matplotlib.pyplot import cm

norm = cm.colors.Normalize(vmin=-1, vmax=1)
h = cm.ScalarMappable(norm=norm, cmap=cmap)
aa = "\langle S \rangle"
bb = "\langle S_{control} \rangle < 0.5"
sm = "$\\frac{1}{\\alpha^2 |S_{w}|}  \sum_{w} {S_{w}^{t}}^2$"

ax.set_xlabel(r"Time spent below tipping point ($\frac{T}{T_{control}}$)")
ax.set_ylabel(r"Second moment ($\frac{\chi }{ \chi_{control} }$)")

p = dict(arrowstyle="<-", connectionstyle="arc3", lw=2)


# cax = ax.panel_axes("r", space=0.3, width=0.1, share=0)
cbar = fig.colorbar(h, ax=ax)


# cbar = fig.colorbar(h, ax=ax)
# cbar = ax.colorbar(h)

# print(cbar.ax.get_yticklabels())
# bbox = cbar.ax.get_yticklabels()[0].get_window_extent()
# print(bbox)
# _, __ = cbar.ax.transAxes.inverted().transform([bbox.x1, 0])


X = 3.5
cbar.ax.annotate(
    "Initiator",
    (X, 0.7),
    xytext=(X, 1),
    xycoords="axes fraction",
    # xycoords=cbar.ax.transAxes,
    arrowprops=p,
    va="center",
    ha="center",
)

cbar.ax.annotate(
    "Stabilizer",
    (X, 0.3),
    xytext=(X, 0),
    xycoords="axes fraction",
    # xycoords=cbar.ax.transAxes,
    arrowprops=p,
    va="center",
    ha="center",
)

ax.axvline(1, ls="dashed", c="k")
ax.axhline(1, ls="dashed", c="k")

inax = ax.inset_axes((0.6, 0.4, 0.35, 0.35), zoom=True)

inax.scatter(errors["c_less_n"], errors["c_less_w"], c=cc, alpha=a, s=s)
inax.format(xlabel="", ylabel="")
inax.axvline(1, ls="dashed", c="k")
inax.axhline(1, ls="dashed", c="k")
# ax.indicate_inset_zoom(inax, edgecolor="black")
# inax.set_xlim(0.8, 1.2)
# inax.set_ylim(0.8, 1.2)


S = 10
h = [
    plt.pyplot.Line2D(
        [],
        [],
        marker="s",
        markersize=S,
        c="k",
        ls="none",
        label=r"$\langle S \rangle > 0.5$",
    ),
    plt.pyplot.Line2D(
        [],
        [],
        marker="o",
        markersize=S,
        c="k",
        ls="none",
        label=r"$\langle S \rangle < 0.5$",
    ),
]
ax.legend(handles=h, loc="upper right", ncols=1)

from utils import ccolors

fig.show()
fig.savefig("figures/fig4_ER_all_alt")
plt.show(block=1)

assert 0

counter = 0
groups = "nudge".split()

GROUP = "deg"
GROUP = "grouped"
fig, ax = plt.subplots()
for x, e in errors.groupby(groups):
    nudge = x
    # g = graphs[id]
    # if id > 10: break
    show_wn(
        e,
        # "n_tips",
        # "success",
        "less_n",
        "less_w",
        ax,
        # Z="less_t",
        by=GROUP,
    )
    show_wn(
        e,
        # "n_tips",
        # "success",
        "greater_n",
        "greater_w",
        ax,
        Z="greater_t",
        marker="s",
        by=GROUP,
    )

    ax.format(
        xlabel=r"Fraction time spent $\langle S \rangle$ < 0.5",
        # ylabel=r"Variance ($\frac{1}{N} \sum_i (s_{w_i} - \overline{s_{w_i}})$)",
        # ylabel=r"Second moment ($\frac{1}{n a^2} \sum s_{w_i}^2$)",
        ylabel="Second moment ($\\frac{1}{\\alpha^2 |S_{w}|}  \sum_{w} {S_{w}^{t}}^2$)",
    )
    from utils import ccolors
    from matplotlib.pyplot import Line2D

    L = np.unique(errors["deg"])
    C = ccolors(L.size)

    handles = []
    for c, l in zip(C, L):
        if l in ["control", -1]:
            continue
            c = "tab:blue"
            l = "control"
        else:
            l = np.round(l, 2)
        h = Line2D([], [], color=c, ls="none", marker="o", label=l)
        handles.append(h)

    ax.legend(handles=handles, title="Degree", loc="r", ncols=1)
    print(counter, end="\r")
    # ax.set_yscale("symlog")

    inax = ax.inset_axes((0.6, 0.4, 0.35, 0.35), zoom=1)
    show_wn(
        e,
        # "n_tips",
        # "success",
        "less_n",
        "less_w",
        inax,
        # Z="less_t",
        by=GROUP,
    )
    inax.format(xlabel="", ylabel="")

    ax.axhline(1, c="k", ls="dashed", lw=0.5)
    ax.axvline(1, c="k", ls="dashed", lw=0.5)

    inax.axhline(1, c="k", ls="dashed", lw=0.5)
    inax.axvline(1, c="k", ls="dashed", lw=0.5)

    # ax.set_xscale("symlog")
    # fig.savefig(f"{counter=}.pdf")
    plt.close(fig)
    counter += 1
    # break

boxes = [
    Line2D(
        [], [], marker="o", ls="none", color="k", label=r"$\langle S \rangle < 0.5$"
    ),
    Line2D(
        [], [], marker="s", ls="none", color="k", label=r"$\langle S \rangle > 0.5$"
    ),
]
ax.legend(boxes, loc="ur", ncols=1, fontsize=6, frameon=0)

from matplotlib.pyplot import cm

norm = cm.colors.Normalize(vmin=-1, vmax=1)
h = cm.ScalarMappable(norm=norm, cmap="cmr.pride")
fig.colorbar(h)
# inax = ax.inset_axes((0.6125, 0.2, 0.35, 0.35))
# inax.set_ylim(0, 1.1)
# inax.set_xlim(1.15, 2.)
# show_wn(
#         e,
#         "c_less_n",
#         "c_less_w",
#         inax,
#         Z="less_t",
#         by = "deg",
#     )
# inax.set_xlabel("")
# inax.set_ylabel("")

# ax.set_yscale("log")
# ax.set_xscale("symlog")
fig.savefig(f"./figures/fig4_ER_all.pdf")
# plt.close(fig)

plt.show(block=1)

# from tqdm import tqdm
# orc_graphs = []
# for g in tqdm(graphs):
#     orc = OllivierRicci(g, method="OTD", alpha=0.5, verbose="INFO")
#     # orc.compute_ricci_curvature()
#     orc.compute_ricci_flow()
#     orc_graphs.append(orc.G.copy())
# pd.to_pickle(orc_graphs, "org_graphs.pkl")

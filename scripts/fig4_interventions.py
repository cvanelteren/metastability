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

    columns = [i for i in row.index if i not in "seed label id deg nudge".split()]
    normalized_data = {}

    for idx, c in enumerate(columns):
        norm_label = f"c_{c}"
        if row.label == "control":
            normalized_data[norm_label] = np.nan
        else:
            normalized_data[norm_label] = row[c] / control[c]

    # copy extra columns for plotting later
    data_labels = "nudge label seed id deg less_t greater_t less_n greater_t less_w greater_t n_tips success".split()
    for add_label in data_labels:
        normalized_data[add_label] = row[add_label]
    return normalized_data


fp = "combined_errors.pkl"
errors = pd.read_pickle(fp)
graphs = pd.read_pickle("graphs_seed=0.pkl")
# graphs = pd.read_pickle("org_graphs.pkl")
roles = pd.read_pickle("graphs_seed=0_roles.pkl")

# fig, ax = plt.subplots()
# ax.plot(roles.max_val - roles.min_val)
# fig.show()
# plt.show(block=1)

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
    print("loading from disk")
    errors = pd.read_pickle("reweighted.pkl")
except:
    print("normalizing")
    errors = pd.DataFrame(errors.apply(norm, axis=1).values.tolist())
    # append roles
    tmp = []
    for idx, row in errors.iterrows():
        if row.label != "control":
            role = roles.query("id == @row.id").roles.iloc[0]
            tmp.append(role[int(row.label)])
        else:
            tmp.append(np.nan)
    errors["role"] = tmp
    print("erhelrjel;jk")
    errors.to_pickle("reweighted.pkl")

bins = np.linspace(-1.1, 1.1, 31)
binned = np.digitize(errors.role, bins) - 1

# bins = np.linspace(errors.n_tips.min(), errors.n_tips.max())
# binned = np.digitize(errors["n_tips"], bins) - 1
print(bins)


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
# a = 0.3
a = 1

cc = C[errors.grouped]


s = 10
print(errors[errors.label == "control"].grouped)
idx = np.unique(np.where(errors[errors.label == "control"])[0])

errors["c_greater_n"].iloc[idx] = np.nan
errors["c_greater_w"].iloc[idx] = np.nan
errors["c_less_n"].iloc[idx] = np.nan
errors["c_less_w"].iloc[idx] = np.nan

errors["c_less_t"].iloc[idx] = np.nan
errors["c_greater_t"].iloc[idx] = np.nan


import matplotlib.pyplot as pplt

fig, ax = plt.subplots(
    ncols=2,
    # nrows=2,
    share=0,
    wratios=[1, 0.8],
    # refwidth=0.1,
    # tight=0,
    # wspace=10,
    # figsize=(10, 5),
    # aligny=0,
    journal="nat2",
)

print(errors.columns)

X = errors["c_greater_t"]
X = errors["c_success"]

ax[0].scatter(
    errors["c_success"],
    errors["c_greater_w"],
    c=cc,
    alpha=a,
    s=s,
    marker="s",
)

ax[0].format(
    # yscale="log",
    # xlim=(-0.1, 2),
)

X = errors["c_less_n"]
X = errors["c_success"]
print(errors.columns)
ax[0].scatter(
    # errors["c_less_t"],
    errors["c_success"],
    errors["c_less_w"],
    # errors["less_w"],
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

xlabel = r"Time spent below tipping point ($\frac{T}{T_{control}}$)"
xlabel = r"Number of succesful tipping points"
ylabel = r"Second moment ($\frac{\chi }{ \chi_{control} }$)"
ax[0].set_xlabel(xlabel)
ax[0].set_ylabel(ylabel)

p = dict(arrowstyle="<-", connectionstyle="arc3", lw=2)

ax.format(xlabelsize=7, ylabelsize=7)


# cax = ax.panel_axes("r", space=0.3, width=0.1, share=0)
cbar = fig.colorbar(h, ax=ax[0], title="Node role ($r_i$)")


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

ax[0].axvline(1, ls="dashed", c="k")
ax[0].axhline(1, ls="dashed", c="k")


inax = ax[0].inset_axes((0.6, 0.4, 0.35, 0.35), zoom=True)
inax.scatter(
    errors["c_success"],
    errors["c_less_w"],
    c=cc,
    alpha=a,
    s=s,
)

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
ax[0].legend(handles=h, loc="upper right", ncols=1)

from utils import ccolors


fp = "kite_isi_beta=0.5732374683235916_sanity.pkl"
# fp = "kite_intervention_beta=0.5732374683235916.pkl"
df = pd.read_pickle(fp)
system_trajectory(df, ax[1], np.inf)
g = nx.krackhardt_kite_graph()
pos = nx.kamada_kawai_layout(g)
c = ccolors(len(g))
inax = ax[1].inset_axes((1.02, 0.6, 0.2, 0.5), zoom=0)
inax.axis("equal")
inax.axis("off")
inax.invert_xaxis()
# inax.invert_yaxis()

# ax[0].axis("off")
nx.draw(g, pos=pos, ax=inax, node_color=c, node_size=30)
inax.axis("equal")

# ax[0].axis("equal")
# ax[1].axis("equal")
fs = 10.0
ax.format(abc=1, xlabelsize=fs, ylabelsize=fs)
fig.show()
# fig.savefig("figures/fig4_interventions")
plt.show(block=1)

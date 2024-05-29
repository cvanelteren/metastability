import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings

import warnings

warnings.filterwarnings("ignore", module="matplotlib\..*")


def show_energy_graph(g, pos, ax, seed=0):
    np.random.seed(seed)
    # make "state" distribution centered
    # on the node in the graph
    rec_width = 0.05
    radius = 800  # empirically

    from matplotlib.patches import Rectangle
    from matplotlib.collections import LineCollection as LC
    from copy import deepcopy

    from utils import ccolors

    c = ccolors(len(g))

    inax = ax.inset_axes((0, 0, 1, 1), zoom=0)
    # nx.draw(g, pos=pos, ax=inax, node_color=c)
    for ci, (node, p) in zip(c, pos.items()):
        h = np.random.rand()
        # add emulated distribution
        for idx, hi in enumerate([h, 1 - h]):
            x = deepcopy(p[:, None])
            x[0] -= rec_width
            x[1] -= rec_width / 2
            x[0] += rec_width * idx * 1
            a = Rectangle(
                deepcopy(x),
                width=rec_width * 0.8,
                height=hi * 0.15,
                zorder=10,
                color="tab:blue",
            )
            inax.add_artist(a)

            # add state labels
            x[1] -= 1 / 4 * rec_width
            x[0] += 0.5 * rec_width
            inax.annotate(
                idx,
                x,
                va="top",
                ha="center",
                fontsize=6,
                color="white",
                weight="extra bold",
            )

    p = np.array(list(pos.values()))
    # plot the nodes

    inax.scatter(*p.T, color=c, zorder=1, s=radius, edgecolor="k")
    # plot the edges
    edges = np.array([[pos[x], pos[y]] for x, y in g.edges()])
    lc = LC(edges, edgecolor="k", zorder=0)
    inax.add_artist(lc)
    # ax.axis("equal")
    ax.margins(0.1)
    inax.margins(0.1)
    # ax.set_ylim(-1.2, 0.8)
    # ax.set_xlim(-576.8, -575.4)
    ax.axis("off")
    inax.axis("off")
    inax.axis("equal")
    # ax.margins(0)


def show_bistability(x, p, ax, large=200, annotate=False):
    # plot the distribution
    ax.bar(x, p, zorder=0)
    ax.format(
        xlabel=r"System macrostate ($\langle S \rangle$)",
        ylabel=r"$P(\langle S \rangle)$",
    )

    # annotate the graph with transitions
    # plot markers
    markers = dict(green=100, blue=50, red=1)
    levels = []

    if annotate:
        for color, level in markers.items():
            idx = np.argmin(abs(p - np.percentile(p, level)))
            xy = (x[idx], p[idx] + 0.02)
            levels.append(xy)
            ax.scatter(*xy, color=color, s=large, zorder=1)

        for start, stop in zip(levels[:-1], levels[1:]):
            ax.annotate(
                "",
                xytext=start,
                xy=stop,
                arrowprops=dict(
                    fc="black",
                    shrink=0.14,
                    connectionstyle="arc3, rad=-0.3",
                ),
            )


def show_system_time(s, ax):
    mu = s.mean(1)
    x = np.arange(0, s.shape[0])
    from scipy.ndimage import gaussian_filter as gf

    # inax = ax.inset_axes((0, 0, 1, 1), zoom=0)
    # inax.plot(x, mu, zorder=0)
    ax.format(xlabel="Time (t)", ylabel="System\nmacrostate")
    ax.plot(x, mu, zorder=0)
    # inax.axis("equal")
    # ax.axis("off")
    limit = 3000

    markers = dict(green=0, blue=0.25, red=0.5)
    relative = None
    levels = []
    # find index relative to the tipping point
    for color, level in reversed(markers.items()):
        idx = np.argmin(abs(mu - level))
        if color == "red":
            relative = idx
        else:
            # find closest index
            idx = np.argwhere(mu == mu[idx])
            kdx = np.argmin(abs(idx - relative))
            idx = idx[kdx]
        xy = (x[idx], mu[idx])
        levels.append(xy)
        ax.scatter(*xy, color=color, s=256, zorder=1, marker="s")

    # annotate circles
    for start, stop in zip(levels[:-1], levels[1:]):
        ax.annotate(
            "",
            start,
            xytext=stop,
            arrowprops=dict(
                fc="black",
                shrink=0.05,
                connectionstyle="arc3,rad=-0.3",
            ),
        )


def show_mi_vs_offset(ax):
    f = lambda x, a, b, c, d, g: a * np.exp(-b * x) + c * np.exp(-d * x) + g
    x = np.linspace(0, 10)
    y = f(x, 0.5, 10, 0.5, 0.3, 0.2)

    ax.plot(x, y)
    ax.axhline(0.2, linestyle="dashed", color="k")
    ax.set_ylim(0, 1)
    ax.annotate(
        "Asymptotic information",
        (5, 0.15),
        va="top",
        ha="center",
        fontsize=16,
    )
    lower = np.ones(x.size) * 0.2

    ax.fill_between(
        x,
        y,
        lower,
        color="gray",
        alpha=0.8,
    )
    ax.annotate(
        "Integrated mutual information\n $\mu(s_i)$",
        (2, 0.4),
        xytext=(7, 0.6),
        ha="center",
        va="center",
        fontsize=16,
        arrowprops=dict(fc="black", shrink=0.05, connectionstyle="arc3,rad=0.3"),
    )
    ax.format(xlabel="Time (t)", ylabel="$I(s_i : S^t)$")


def show_progression(ax):
    # load data
    from pathlib import Path

    base = Path("./data")
    fp = "exact_recursive_tree_4_beta=0.567_T=200.pkl"
    fp = "exact_sedgewick_beta=0.8539043106824097_T=200.pkl"
    fp = "exact_kite_dyn=ising_beta=0.5732374683235916_T=200.pkl"
    # fp = "exact_circle_beta=0.567_T=200.pkl"
    df = pd.read_pickle(base / Path(fp))
    mi = df.mi.iloc[6]
    g = df.attrs["settings"].g
    pos = nx.kamada_kawai_layout(g)

    from utils import ccolors

    c = ccolors(len(g))

    # index in the correct data set
    iloc_targets = [1, 2, 6]
    renorm = lambda x: (x - x.min()) / (x.max() - x.min())

    def shift(pos: dict, idx=0, xshift=8):
        rpos = {}
        for k, v in pos.items():
            rpos[k] = v + np.array([idx * xshift, 0])
        return rpos

    # put a dot at the places where the transitions
    # occurs. For ising this would be M(S) = x
    markers = dict(green=0.1, blue=0.3, red=0.5)
    n = len(markers)
    dot_pos = {}

    # macrostate was binned in the "mag" label
    # print(df.columns)
    macro_state = df.mag.round(2)
    for idx, (color, level) in enumerate(markers.items()):
        # draw networks at shift
        shifted = shift(pos, idx, 1)
        # compute the node size
        idx = np.argmin(abs(macro_state - level))
        node_size = df.mi.iloc[idx].sum(0)
        node_size = renorm(node_size)
        node_size *= 200
        inax = ax.inset_axes((0, 0, 1, 1), zoom=0)

        nx.draw(
            g,
            pos=shifted,
            ax=inax,
            node_size=node_size,
            node_color=c,
            # with_labels=1,
        )
        ax.axis("off")
        inax.axis("off")
        inax.axis("equal")
        # compute the dot location
        # relative to the 0 node
        label = shifted[3]
        dot = label + np.array([0, 0.5])
        dot_pos[color] = dot
    # put dot on  top to emulate subplots
    tipping_size = 100

    for color, xy in dot_pos.items():
        ax.scatter(*xy, color=color, s=tipping_size)

    # show arrows between transitions
    xy = np.asarray(list(dot_pos.values()))
    for start, stop in zip(xy[:-1], xy[1:]):
        ax.annotate(
            "",
            stop,
            xytext=start,
            arrowprops=dict(
                fc="black",
                shrink=0.20,
                connectionstyle="arc3,rad=-0.3",
            ),
        )
    ax.axis("equal")


def show_progression_restricted(level, level_color, ax, annotate=False):
    from pathlib import Path

    base = Path("./data")
    fp = "exact_recursive_tree_4_beta=0.567_T=200.pkl"
    fp = "exact_sedgewick_beta=0.8539043106824097_T=200.pkl"
    fp = "exact_kite_dyn=ising_beta=0.5732374683235916_T=200.pkl"
    # fp = "exact_circle_beta=0.567_T=200.pkl"
    df = pd.read_pickle(base / Path(fp))
    mi = df.mi.iloc[6]
    g = df.attrs["settings"].g
    pos = nx.kamada_kawai_layout(g)

    from utils import ccolors

    c = ccolors(len(g))
    # macrostate was binned in the "mag" label
    # print(df.columns)
    macro_state = df.mag.round(2)
    idx = np.argmin(abs(macro_state - level))
    mi = df.mi.iloc[idx]
    targets = [9, 7, 3]
    for target in targets:
        ax.plot(mi[:, target], color=c[target])
    ax.format(
        # xscale = "log",
        yscale="log",
        xlabel="Time(t)",
        ylabel="$I(s_i : S'^t)$",
    )
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 1)
    from matplotlib.patches import Circle

    p = Circle((0.5, 1.07), radius=0.05, color=level_color, transform=ax.transAxes)
    ax.figure.add_artist(p)
    # show asymptotic versus
    if annotate:
        target = targets[2]
        offset = mi[-1, target]
        ax.axhline(offset, linestyle="dashed", color="k", zorder=0)

        # ax.annotate(
        #     "Asymptotic\ninformation",
        #     (0.25, 0.05),
        #     ha="center",
        #     xycoords="axes fraction",
        # )

        # add shading
        t = np.arange(mi.shape[0])
        lower = np.ones(t.size) * offset
        y = mi[:, target]
        ax.fill_between(
            t,
            y,
            lower,
            color=c[target],
            alpha=0.3,
            hatch="x",
            label="Intergrated\ninformation",
        )

        ax.fill_between(
            t,
            lower,
            color=c[target],
            alpha=0.4,
            hatch="o",
            label="Asymptotic\ninformation",
        )
        ax.legend(
            loc="b",
            ncols=2,
        )
    # ax.axis("image")

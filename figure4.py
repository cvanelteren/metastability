# import matplotlib as mpl
# mpl.use("TkAgg")  # or whatever other backend that you want
import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy


def make_windows(idx):
    # get where large transitions occur
    # wdx = idx
    wdx = np.where(np.diff(idx) > 1)[0]
    windows = []
    start = 0
    for wdxi in wdx:
        window = np.arange(start, wdxi)
        windows.append(window)
        start = wdxi
    return windows


def get_var(row):
    rmses = np.zeros((2, row.seed.unique().size))
    for sidx, (seed, srow) in enumerate(row.groupby("seed")):
        for pidx, operator in enumerate((np.less, np.greater)):
            system = srow.system.iloc[0]
            idx = np.where(operator(system, 0.5))[0]
            windows = make_windows(idx)
            for window in windows:
                tmp = system[idx[window]]
                rmses[pidx, sidx] += np.nanmean(((tmp - np.nanmean(tmp)) ** 2)) / len(
                    windows
                )
    return rmses


def system_trajectory(df, ax, nudge, seed=1234, spacing=0.6, max_t=20000):
    from utils import ccolors

    c = ccolors(df.label.unique().size - 1)
    dfi = df.query("nudge == @nudge & seed == @seed")
    print(dfi.shape)

    starts = []
    yt = []

    up = 0
    from matplotlib.pyplot import Line2D

    h = []
    print(dfi.columns, dfi.shape)
    for adx, (idx, dfj) in enumerate(dfi.iterrows()):
        if dfj.label == "control":
            ci = "tab:blue"
        else:
            ci = c[dfj.label]
        yt.append(up)
        yt.append(up + 1)
        y = dfj.system[:max_t] + up

        hi = Line2D(
            [],
            [],
            label=str(dfj.label).capitalize(),
            color=ci,
            marker="o",
            linestyle="none",
        )
        h.append(hi)
        ax.plot(
            y,
            color=ci,
            lw=0.5,
        )
        starts.append(up)
        up += 1 + spacing
    ax.format(
        xlabel="Time(t)",
        ylabel="Fraction of nodes +1",
        # ylim = (-0.05, 1.05)
    )
    # ax.figure.legend(
    #     loc="r",
    #     handles=h,
    #     ncols=1,
    #     title="Pinning\nintervention\non",
    # )

    yl = []
    for s in starts:
        yl.append(0)
        yl.append(1)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl)


def make_windows(idx):
    # get where large transitions occur
    wdx = np.where(np.diff(idx) > 1)[0]
    windows = []
    for start, stop in zip(wdx[:-1], wdx[1:]):
        window = np.arange(start, stop)
        windows.append(window)
    return windows


def estimate_white_noise(row, tipping=0.5):
    from scipy.stats import sem

    macrostate = row.system
    output = {}
    # print(row.label, np.where(macrostate > tipping)[0].size / macrostate.size)
    for idx, op in enumerate((np.greater, np.less)):
        where = np.where(op(macrostate, tipping))[0]

        other_name = op.__name__ + "_n"
        name = op.__name__ + "_w"
        num_tips = op.__name__ + "_t"

        output[other_name] = where.size / macrostate.size
        # output[other_name] = (macrostate.size - len(where)) / (macrostate.size)
        # output[op.__name__ + "_w"] = (tmp**2).sum()

        output[name] = 0
        # output[other_name] = 0
        windows = make_windows(where)
        output[num_tips] = len(windows)
        for window in windows:
            d = macrostate[where[window]]
            # d = sem(d, nan_policy="omit")
            a = 1
            if op == np.greater:
                d = abs(1 - d)
            if row.label != "control" and op == np.greater:
                a = 4 / 5
            # p = {}
            # for di in d:
            # p[di] = p.get(di, 0) + 1 / len(window)
            # l = 0
            # for k, v in p.items():
            # l += k**2 * v
            # d = np.var(d)
            d = np.mean(d**2) / a**2  # / len(windows)
            # d = np.var(d)
            # d = np.nanmean((d - np.nanmean(d)) ** 2)  # / len(window)
            # d = a**2 * l
            # d = a**2 * (d**2).sum() / len(window)
            output[name] += d  # / macrostate.size
            # output[other_name] += len(window) / len(windows)
    output["label"] = row.label
    output["seed"] = row.seed
    output["nudge"] = row.nudge
    return output


def show_wn(df, X, Y, ax, Z=None, marker="o"):
    from utils import ccolors
    from scipy.stats import sem

    c = ccolors(df.label.unique().size)
    N = 2
    for intervention, dfi in df.groupby("label"):
        x = dfi[X]
        y = dfi[Y]

        s = 10  # size of the nodes
        if Z is not None:
            s = 100 * np.nanmean(dfi[Z])
            # print(X, intervention, s)

        mux = np.nanmean(x)
        muy = np.nanmean(y)

        sx = sem(x, nan_policy="omit") * N
        sy = sem(y, nan_policy="omit") * N

        # sx = x.std(ddof=0) * N
        # sy = y.std(ddof=0) * N

        # print(f"{intervention=}\n\t{mux=}\t{muy=}\t{sx=}\t{sy=}")
        if intervention != "control":
            color = c[intervention]
        else:
            color = "tab:blue"
            alpha = 0.05

            ax.axvspan(
                mux - sx,
                mux + sx,
                color=color,
                alpha=alpha,
            )

            ax.axhspan(
                muy - sy,
                muy + sy,
                color=color,
                alpha=alpha,
            )

        yerr = np.array((y.min(), y.min()))[:, None]
        # ax.scatter(mux, muy)

        ax.errorbar(
            mux,
            muy,
            xerr=sx,
            yerr=sy,
            # yerr=yerr,
            color=color,
            # markersize=0,
            marker=marker if intervention != "control" else "",
        )

        # ax.scatter(
        #     mux,
        #     muy,
        #     s=s,
        #     color=color,
        #     marker=marker if intervention != "control" else "",
        # )


fp = "kite_isi_beta=0.5732374683235916.pkl"
# fp = "kite_intervention_beta=0.5732374683235916.pkl"
df = pd.read_pickle(fp)
print(df.columns)
print(df.seed.unique())
print("Loaded data")
tmp = df.apply(estimate_white_noise, axis=1)
errors = []
for idx, row in tmp.reset_index().iterrows():
    errors.append(row.iloc[1])
errors = pd.DataFrame(errors)

print(errors.columns)
# plt.config.use_style("seaborn-poster")
# tmp = df.groupby("label").apply(get_var)
# mu = errors[errors.label == "control"]["less_n greater_n".split()].mean()
# errors[errors.label == "control"]["less_n"] = mu
# errors[errors.label == "control"]["greater_n"] = mu

# mu = errors[errors.label == "control"]["less_w greater_w".split()]

idx = np.where(errors.label == "control")[0]
jdx = np.where(errors.columns == "less_w")[0]
kdx = np.where(errors.columns == "greater_w")[0]
errors.iloc[idx, jdx] = (errors.iloc[idx, jdx] + errors.iloc[idx, kdx]) / 2
idx = np.where(errors.label == "control")[0]
jdx = np.where(errors.columns == "less_n")[0]
kdx = np.where(errors.columns == "greater_n")[0]
errors.iloc[idx, jdx] = (errors.iloc[idx, jdx] + errors.iloc[idx, kdx]) / 2

# errors[errors.label == "control"]["greater_w"] = mu

errors["greater_t"] /= errors["greater_t"].max()
errors["less_t"] /= errors["less_t"].max()

print("making figure")

for x, e in errors.groupby("nudge"):
    nudge = x
    fig, ax = plt.subplots(ncols=3, share=0)
    show_wn(
        e,
        "less_n",
        "less_w",
        ax[2],
        Z="less_t",
    )
    show_wn(
        e,
        "greater_n",
        "greater_w",
        ax[2],
        Z="greater_t",
        marker="s",
    )
    system_trajectory(df, ax[1], x)
    g = nx.krackhardt_kite_graph()
    from fa2 import ForceAtlas2 as fa2

    pos = nx.kamada_kawai_layout(g)
    from utils import ccolors

    fig.suptitle("Effect of pinning intervention to state +0")

    # ax[1].format(title="Fraction of nodes < 0.5")
    # ax[2].format(title="Fraction of nodes > 0.5")
    ax[2].format(
        xlabel="Fraction time spent <S> < 0.5",
        # ylabel=r"Variance ($\frac{1}{N} \sum_i (s_{w_i} - \overline{s_{w_i}})$)",
        # ylabel=r"Second moment ($\frac{1}{n a^2} \sum s_{w_i}^2$)",
        ylabel="Second moment ($\\frac{1}{\\alpha^2 |S_{w}|}  \sum_{w} {S_{w}^{t}}^2$)",
    )

    from matplotlib.pyplot import Line2D

    handles = [
        Line2D(
            [],
            [],
            color="k",
            marker="o",
            label="<S> < 0.5",
            linestyle="none",
        ),
        Line2D(
            [],
            [],
            color="k",
            marker="s",
            label="<S> > 0.5",
            linestyle="none",
        ),
    ]

    ax[2].legend(
        handles=handles,
        loc="ur",
        # pad=0,
        # space=0,
        ncols=1,
        fontsize=6,
    )
    ax[2].set_title("Noise dependent tipping behavior")
    ax[1].set_title("System trajectory under intervention")
    # ax[2].set_title("")

    labels = np.array(df.label.unique())
    h = []
    c = ccolors(len(g))
    for l in labels:
        if l == "control":
            C = "tab:blue"
        else:
            C = c[l]
        h.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                label=str(l).capitalize(),
                color=C,
            )
        )

    fig.legend(h, loc="r", title="Pinning\nintervention\non", ncols=1)

    ax.format(abc=True, abc_kw=dict(fontsize=14))

    inax = ax[0].inset_axes((0.0, 0.0, 1, 1), zoom=0)
    inax.axis("off")
    ax[0].axis("off")
    nx.draw(g, pos=pos, ax=inax, node_color=c)
    inax.axis("equal")
    # ax[0].axis(
    # "equal",
    # )
    # ax[2].axis("auto")
    # ax[1].axis("auto")
    # ax[1].axis("square")
    # nudge = "all"
    fig.savefig(f"./figures/figure4_{nudge=}.pdf")

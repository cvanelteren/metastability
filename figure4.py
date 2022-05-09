import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy


def make_windows(idx):
    # get where large transitions occur
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


def system_trajectory(df, ax, seed=1234):
    from utils import ccolors

    c = ccolors(df.label.unique().size - 1)
    dfi = df[df.seed == seed]

    starts = []
    spacing = 0.6
    yt = []

    up = 0
    from matplotlib.pyplot import Line2D

    h = []
    for adx, (idx, dfj) in enumerate(dfi.iterrows()):
        if dfj.label == "control":
            ci = "tab:blue"
        else:
            ci = c[dfj.label]
        yt.append(up)
        yt.append(up + 1)
        y = dfj.system[:20000] + up

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
    ax.figure.legend(
        loc="r",
        handles=h,
        ncols=1,
        title="Pinning\nintervention\non",
    )

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
    for idx, op in enumerate((np.greater, np.less)):
        where = np.where(op(macrostate, tipping))[0]

        other_name = op.__name__ + "_n"
        name = op.__name__ + "_w"

        output[other_name] = len(where)
        # output[op.__name__ + "_w"] = (tmp**2).sum()

        tmp = macrostate[where]
        if op == np.greater:
            tmp -= 0.5
        output[name] = 0
        output[other_name] = 0
        windows = make_windows(where)
        for window in windows:
            try:
                d = tmp[where[window]]
                if tmp.size <= 1:
                    continue

                # d = sem(d, nan_policy="omit")
                d = np.nanmean((d - np.nanmean(d)) ** 2) / len(windows)
                # d = (d**2).sum() / len(windows)
                output[name] += d
                output[other_name] += len(window) / len(windows)
            except:
                continue
    output["label"] = row.label
    output["seed"] = row.seed
    return output


def show_wn(df, X, Y, ax):
    from utils import ccolors
    from scipy.stats import sem

    c = ccolors(df.label.unique().size - 1)
    print(df.columns, df.shape)
    for intervention, dfi in df.groupby("label"):
        x = dfi[X]
        y = dfi[Y]

        mux = np.nanmean(x)
        muy = np.nanmean(y)

        sx = sem(x, nan_policy="omit") * 2
        sy = sem(y, nan_policy="omit") * 2

        # sx = x.std()
        # sy = y.std()
        print(f"{intervention=}\n\t{mux=}\t{muy=}\t{sx=}\t{sy=}")
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

        ax.errorbar(
            mux,
            muy,
            xerr=sx,
            yerr=sy,
            color=color,
            marker="o",
        )


fp = "kite_isi_beta=0.5732374683235916.pkl"
# fp = "kite_intervention_beta=0.5732374683235916.pkl"
df = pd.read_pickle(fp)

tmp = df.apply(estimate_white_noise, axis=1)
errors = []
for idx, row in tmp.reset_index().iterrows():
    errors.append(row.iloc[1])
errors = pd.DataFrame(errors)

# plt.config.use_style("seaborn-poster")
# tmp = df.groupby("label").apply(get_var)
fig, ax = plt.subplots(ncols=3, share=0)
show_wn(errors, "less_n", "less_w", ax[0])
show_wn(errors, "greater_n", "greater_w", ax[1])
system_trajectory(df, ax[2])


ax.format(abc=True)
ax[0].format(title="Fraction of nodes < 0.5")
ax[1].format(title="Fraction of nodes > 0.5")
ax[:2].format(
    xlabel="Average time spent",
    ylabel="White noise",
)

fig.savefig("test")

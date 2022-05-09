import matplotlib.pyplot as plt, cmasher as cmr, pandas as pd
import numpy as np, os, sys, networkx as nx, warnings
import proplot as pplt
from plexsim import models
from pathlib import Path


f = lambda x, a, b, c, d, g: a * np.exp(-b * x) + c * np.exp(-d * x)


def get_half_time_and_imi(mi, f, a, offset=True):
    nt, nodes = mi.shape
    halfimi = np.zeros((3, nodes))

    # for integration: int_f
    # for minimization: min_f
    if offset:
        int_f = lambda x, coeffs: f(x, *coeffs) - coeffs[-1]
        min_f = lambda x, coeffs, theta: np.abs((f(x, *coeffs) - theta))
    else:
        int_f = lambda x, coeffs: f(x, *coeffs)
        min_f = lambda x, coeffs, theta: np.abs((f(x, *coeffs) - theta))

    from scipy import optimize, integrate
    from functools import partial

    for idx, node in enumerate(mi.T):
        coeffs = a[idx]
        theta = 0.5 * (f(0, *coeffs) - coeffs[-1]) + coeffs[-1]  # halftime value
        # theta = 0.5 * (f(0, *coeffs))

        # compute half time
        res = optimize.minimize(min_f, x0=0, args=(coeffs, theta))
        halftime = res.x[0]
        assert res.x[0] >= 0, res.x

        # # compute asymptotic
        # theta = coeffs[-1]
        # res = optimize.minimize(min_f, x0=0, args=(coeffs, theta))
        # assert res.x[0] >= 0, res.x
        # asymp = res.x[0]
        asymp = coeffs[-1]

        # compute imi
        imi, _ = integrate.quad(int_f, 0, np.inf, args=(coeffs,))

        halfimi[:, idx] = (imi, halftime, asymp)
    return halfimi


def fit_curve(mi, f):
    from scipy import optimize
    import inspect

    nt, nodes = mi.shape
    xdata = np.repeat(np.arange(nt)[None], nodes, axis=0).T
    # n_params = len(inspect.signature(f).parameters) - 1
    opts = []
    for idx, (x, y) in enumerate(zip(xdata.T, mi.T)):
        # if y[0] < 1e-6:
        # y.fill(0)
        # y[y < 1e-10] = 0
        # y = np.sort(y)[::-1]
        popt, pcov = optimize.curve_fit(f, x, y, bounds=(0, np.inf), maxfev=1e5)
        opts.append(popt)
    return np.asarray(opts)


def do_fit(data, f, redo=False):
    from tqdm import tqdm

    fn = data["file"]
    fn = f"./params/{fn}_coeffs.pkl"
    if not redo:
        print("loading from disk")
        if Path(fn).exists():
            return pd.read_pickle(fn)

    mis = load_mis(data)
    coeffs = {}
    for k, mi in tqdm(mis.items()):
        c = fit_curve(mi, f)
        coeffs[k] = c
    pd.to_pickle(coeffs, fn)
    return coeffs


def load_mis(data):
    if "new_mis" in data:
        mis = data["new_mis"]
    else:
        mis = data["mis"]
    # return mis
    return data["mis"]


import cmasher as cmr


def ccolors(n, cmap=cmr.pride):
    return cmap(np.linspace(0, 1, n, 0))


def do_halfimi(data, f=f, redo=False):
    # load from disk
    fn = data["file"]
    output = f"./params/{fn}_halfimi.pkl"
    if not redo:
        print("loading from disk")
        if Path(output).exists():
            return pd.read_pickle(output)

    half_imi = {}
    from tqdm import tqdm

    fn = f"./params/{fn}_coeffs.pkl"
    coeffs = pd.read_pickle(fn)
    mis = load_mis(data)
    for idx, (k, mi) in enumerate(tqdm(mis.items())):
        c = coeffs[k]
        halfimi = get_half_time_and_imi(mi, f, c)
        half_imi[k] = dict(coeffs=c, halfimi=halfimi)
    pd.to_pickle(half_imi, output)
    return half_imi


# plot functions
def show_fit(mi, f, a, x=None):
    import scprep

    m = np.linspace(0, 1, len(mi.T), 0)
    colors = scprep.plot.colors.tab40()(m)
    fig, ax = plt.subplots()
    n = mi.size
    if x is not None:
        n = x
    spacing = 1
    for idx, mii in enumerate(mi.T):
        xr = np.linspace(0, mii.size, mii.size)
        ax.scatter(xr, mii, color=colors[idx], zorder=1, alpha=0.2)

        x = np.linspace(0, 1 * mii.size, 100)
        y = f(x, *a[idx])
        ax.plot(x, y, color=colors[idx], linestyle="solid", zorder=2)
    return fig, colors


def get_binds(data):
    # get degree to nodes mapping
    binds = {}
    m = models.Potts(**data["model_settings"])
    for k, v in m.graph.degree():
        binds[v] = binds.get(v, []) + [m.adj.mapping[k]]
    binds = dict(sorted(binds.items(), key=lambda x: x[0]))
    return binds


def show_time_decay(data, f=f):
    # makes figure for all the mi
    from fa2 import ForceAtlas2
    from tqdm import tqdm
    import networkx as nx

    m = models.Potts(**data["model_settings"])
    mis = load_mis(data)
    fn = data["file"]
    half_imi = do_halfimi(data, f=f)
    center, deg = max(m.graph.degree(), key=lambda x: x[1])
    pos = nx.spring_layout(m.graph)
    deg = dict(sorted(m.graph.degree(), key=lambda x: x[1], reverse=1))
    tdeg = {}
    for k, v in deg.items():
        tdeg[v] = tdeg.get(v, []) + [k]
    l = list(tdeg.values())

    pos = nx.shell_layout(m.graph, nlist=l)
    pos = ForceAtlas2(verbose=0).forceatlas2_networkx_layout(
        m.graph, pos=pos, iterations=20
    )

    for k, mi in tqdm(mis.items()):
        coeffs = half_imi[k]["coeffs"]
        fig, colors = show_fit(mi, f, coeffs)
        ax = fig.axes[0]
        ax.set_title(np.round(float(k), 2))
        ax.set_xlabel("Time")
        ax.set_ylabel("$I(s_i^{t + \\tau} : S^t)$")

        inax = ax.inset_axes((0.5, 0.5, 0.5, 0.5))
        colors_graph = np.zeros((len(mi.T), 4))
        s = np.zeros((len(mi.T)))
        tmp = half_imi[k]["halfimi"]
        for node in m.graph.nodes():
            idx = m.adj.mapping[node]
            colors_graph[idx] = colors[idx]
            s[idx] = mi[:, idx].sum()
            ax.axvline(tmp[0, idx], color=colors[idx], linestyle="dashed", alpha=0.5)

        nx.draw(m.graph, pos, ax=inax, node_color=colors_graph, node_size=s)

        inax.axis("equal")
        inax.axis("off")
        # print(f"saving at ./figures/{fn}={k}_.png")
        fig.savefig(f"./figures/{fn}={k}_.png", dpi=200, transparent=0)
    return fig


def show_state_dist(data):
    from collections import Counter

    df = data["df"]

    b = df.bin.values
    c = df.state_count.values

    bins = np.linspace(0, 1, 10)
    idx = np.digitize(b, bins)
    y = np.zeros(bins.size)

    n = {}
    for jdx, i in enumerate(idx):
        y[i] += c[jdx]
    y /= y.sum()
    fig, ax = pplt.subplots()
    ax.bar(bins, y)
    ax.format(xlabel="$M(S)$", ylabel="PMF")
    fig.savefig(f"./figures/state_dist_{m.graph}.png")
    fig.show()


def show_degree_social(
    gs=dict(
        florentine_families=nx.florentine_families_graph(),
        les_miserables=nx.les_miserables_graph(),
        davis_southern_women=nx.davis_southern_women_graph(),
        karateclub=nx.karate_club_graph(),
    )
):

    fig, ax = pplt.subplots(nrows=2, ncols=2)
    for idx, (name, gi) in enumerate(gs.items()):
        print(name, gi)
        axi = ax[idx]
        d = dict(gi.degree())
        axi.hist(list(d.values()), density=1)
        axi.set_title(name.capitalize().replace("_", " "))
    ax.format(xlabel="Degree (k)", ylabel="PMF")
    fig.savefig("./figures/social_dist_degree.png")


def make_error_df(data):
    # make a dataframe collecting imi/halftime etc
    # over the different magnetization
    binds = get_binds(data)
    colors = cmr.pride(np.linspace(0, 1, len(binds), 0))
    # open plot
    from scipy.stats import sem

    fn = data["file"]
    half_imi = pd.read_pickle(f"./params/{fn}_halfimi.pkl")
    tmp = []
    for k, v in half_imi.items():
        halfimi = v["halfimi"]
        k = float(k)
        for idx, (deg, nodes) in enumerate(binds.items()):
            y = halfimi[0][nodes].mean()
            sy = sem(halfimi[0][nodes])
            if np.isnan(sy):
                sy = 0
            yy = halfimi[1][nodes].mean()
            syy = sem(halfimi[1][nodes])
            if np.isnan(syy):
                syy = 0

            yyy = halfimi[2][nodes].mean()
            syyy = sem(halfimi[2][nodes])
            if np.isnan(syyy):
                syy = 0

            row = dict(
                imi=y,
                e_imi=sy,
                halftime=yy,
                e_halftime=syy,
                asymp=yyy,
                e_asymp=syyy,
                deg=deg,
                ms=float(k),
            )
            tmp.append(row)

    tmp = pd.DataFrame(tmp)
    return tmp


def show_cumulative_halfimi(data):
    tmp = make_error_df(data)
    binds = get_binds(data)
    colors = cmr.pride(np.linspace(0, 1, len(binds), 0))

    fig, ax = pplt.subplots(ncols=3, sharey=0)
    for idx, (labels, i) in enumerate(tmp.groupby("deg".split())):
        ax[0].errorbar(i.ms, i.imi, i.e_imi, color=colors[idx])
        ax[1].errorbar(i.ms, i.halftime, i.e_halftime, color=colors[idx])
        ax[2].errorbar(i.ms, i.asymp, i.e_asymp, color=colors[idx])

    # ax.format(yscale = "symlog")
    handles = [
        plt.Line2D([], [], linestyle="none", label=l, marker="o", color=c)
        for c, l in zip(colors, np.unique(tmp.deg))
    ]

    ax[1].legend(loc="t", ncols=len(handles), handles=handles, title="Degree (k)")
    ax.format(xlabel="System magnetization $M(S)$")
    ax[0].set_ylabel("IMI")
    ax[1].set_ylabel("halftime")
    ax[2].set_ylabel("Asymptotic")
    for idx in range(3):
        ax[idx].format(yscale="symlog", ymin=0)

    m = models.Potts(**data["model_settings"])
    fig.savefig(f"./figures/cumulative_{m.graph}.png")
    fig.show()
    return fig


def list_results(data):
    s = f"{data['file']=}"
    for k, v in data.items():
        if k not in ["file"]:
            s += f"{k=}\n"
    s += data["model_settings"]["graph"]
    print(s)


def plot_degree(g, **kwargs):
    deg = dict(g.degree())
    fig, ax = plt.subplots()
    ax.hist(deg.values(), **kwargs)
    return fig


rmse = lambda x, y: np.sqrt(((x - y) ** 2).mean())


def get_rmse(data, f):
    mis = load_mis(data)
    half_imi = do_halfimi(data)
    rmses = []
    for idx, (k, mi) in enumerate(mis.items()):
        for node, mii in enumerate(mi.T):
            c = half_imi[k]["coeffs"][node]
            x = np.arange(0, len(mii))
            x = f(x, *c)
            rmses.append(mse(x, mii))
    return rmses


def show_fit_rmse(data, f=f):
    mses = get_rmse(data, f)
    fig, ax = plt.subplots()
    ax.hist(mses, density=True, bins=100)
    ax.set_xlabel("Root mean squared error")
    ax.set_ylabel("PMF")
    # ax.set_xscale("log")
    return fig


def kshell_layout(g, **kwargs):
    import networkx as nx

    # sort degree
    deg = dict(sorted(dict(g.degree()).items(), key=lambda x: x[1], reverse=1))
    rdeg = {}
    for k, v in deg.items():
        rdeg[v] = rdeg.get(v, []) + [k]
    nlist = list(rdeg.values())
    return nx.shell_layout(g, nlist=nlist, **kwargs)


def rebin_states(data, rebin):
    df = data["df"]
    snaps = dict()
    cond = dict()
    cond_counter = dict()
    print("Rebinning data")
    for idx, dfi in df.iterrows():
        # compute new abs distance
        state = dfi.state
        conditional = dfi.conditional
        val = abs(np.mean(state) - 0.5)
        assert val <= 0.5
        state = tuple(state)
        snaps[state] = snaps.get(state, 0) + dfi.state_count
        cond[state] = cond.get(state, np.zeros(conditional.shape)) + conditional
        cond_counter[state] = cond_counter.get(state, 0) + 1

    assert np.allclose(sum(snaps.values()), 1), sum(snaps.values())
    print("Creating new dataframe")
    # recompute mi and restore data
    new_df = []
    for state, val in snaps.items():
        # renormalize conditional
        # print(cond[state].max(), cond_counter[state])
        # assert cond_counter[state] <= 2 and cond_counter[state] > 0, cond_counter[state]
        cond[state] = cond[state] / cond_counter[state]
        # print(f"Found {cond_counter[state]=}")
        conditional = cond[state]
        mag = abs(np.mean(state) - 0.5)
        mag = rebin[np.digitize(mag, rebin)]
        row = dict(
            state=state,
            state_count=val,
            conditional=conditional,
            bin=mag,
            cond_counter=cond_counter,
        )
        new_df.append(row)
    new_df = pd.DataFrame(new_df)
    return new_df


def rebin_mis(new_df):
    new_mis = {}
    from imi.infcy import mutualInformation as MI

    for x, dfi in new_df.groupby("bin"):
        states = np.stack(dfi.state)
        vals = np.stack(dfi.state_count)
        # make valid state distribution
        snap = {tuple(s): v / sum(vals) for s, v in zip(states, vals)}
        # snap = {states: vals}
        assert np.allclose(sum(snap.values()), 1), sum(snap.values())

        # make valid conditional distribution
        cond = np.stack(dfi.conditional)
        # print(cond[:, -1, :, 0])
        assert cond.max() <= 1.0, cond.max()
        cond = {tuple(s): i for s, i in zip(states, cond)}

        px, mi = MI(cond, snap)
        # print(mi[0], px[0])
        # fig, ax = plt.subplots()

        # ax.plot(mi)
        # fig.show()
        # plt.show(block=1)

        # print(mi[0], px[0], snap)
        bin = x
        new_mis[bin] = mi
    return new_mis


def compute_abs_mi_decay(data, rebin=np.linspace(-0.1, 0.6, 7)):
    # map the states  as a function of  absolute distance to
    # the stipping point
    # extra state distribution and create new df
    assert "file" in data
    new_df = rebin_states(data, rebin)
    new_mis = rebin_mis(new_df)
    data["new_mis"] = new_mis
    data["new_df"] = new_df
    # fig, ax = plt.subplots()
    # for k, v in new_mis.items():
    #     ax.plot(v, label=k)
    # # for k, v in data["mis"].items():
    # # ax.plot(v)
    # # ax.legend(ncol=5)
    # fig.show()
    # plt.show(block=1)

    base = Path("data/")
    new_file = base / data["file"]
    print(f"saving to {new_file=}")
    pd.to_pickle(data, new_file)
    return data

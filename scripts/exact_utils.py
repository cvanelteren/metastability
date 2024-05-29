import proplot as plt, pandas as pd, numpy as np, networkx as nx
from plexsim import models
from utils import fit_curve as fc
from exact import ising
from exact import gen_states
from scipy import optimize

from utils import fit_curve, get_half_time_and_imi, rmse

try:
    import cupy as np

    np.eye(5)
    print("Using cupy")

except:
    print("Using numpy")
    import numpy as np


def show_panel(df, p, which="system"):
    g = df.settings.g
    pos = nx.circular_layout(g)
    m = models.Potts(g)
    binds = {}
    for node in g.nodes():
        deg = g.degree(node)
        binds[deg] = binds.get(deg, []) + [node]

    mag = df.mag.unique()
    N = df.mag.unique().size
    print(mag)

    layout = np.arange(0, mag.size, dtype=int) + 1

    layout = np.zeros((2, 7))
    layout[0] = [1, 2, 3, 4, 5, 6, 7]
    layout[1] = [1, 12, 11, 10, 9, 8, 7]

    # tmp = np.zeros((3,4))
    # tmp.flat[:layout.size] = layout
    # layout = tmp.reshape(3,4)
    # layout = layout.reshape(3,4)

    fig = plt.figure(sharey=True)
    ax = fig.add_subplots(layout)
    yl = "$I(s_i^t ; S)$"
    from utils import ccolors

    n = len(g)
    colors = ccolors(n)
    for idx, (mag, dfi) in enumerate(df.groupby("mag")):
        mi = dfi.mi.iloc[0]
        for node in range(n):
            ax[idx].plot(mi[:, node], colors=colors[node])
        ax[idx].set_xlabel("Time (t)")
        ax[idx].set_ylabel(yl)
        ax[idx].set_title(round(mag, 2))

        axo = ax[idx].twinx()
        axo.plot(dfi.system.iloc[0])
        # ax[idx].set_xlim(0, 30)
        # ax[idx].set_ylim(0, 1.1)

        if mag != -1:
            if which == "system":
                mi = dfi.nodes.iloc[0]
            else:
                mi = dfi.node_system.iloc[0]
            # mi = df[np.round(df.mag, 2) == np.round((1 - mag), 2)].nodes.iloc[0]
            for node in range(n):
                ax[idx].plot(mi[:, node], colors=colors[node], linestyle="dashed")
    inax = ax[0, -1].inset_axes((0.5, 0.5, 0.5, 0.5), zoom=False)
    nx.draw(g, pos=pos, ax=inax, node_color=colors, node_size=16)
    fig.savefig(f"./figures/{p.name}_exact_panel.png")


def fit_curve(df, f, offset=False):
    d = []
    for idx, (mag, dfi) in enumerate(df.groupby("mag")):
        print(mag, end="\r")
        if mag != 0.0 or mag != 1.0:
            mi = dfi.mi.iloc[0]
            coeffs = fc(mi, f)
            imi, half, asymp = get_half_time_and_imi(mi, f, coeffs, offset)
            rmses = []
            for node, mii in enumerate(mi.T):
                c = coeffs[node]
                x = np.arange(0, len(mii))
                x = f(x, *c)
                rmses.append(rmse(x, mii))
            rmses = np.array(rmses)
            row = dict(
                mag=mag,
                coeffs=coeffs,
                half=half,
                asymp=asymp,
                imi=imi,
                rmse=rmses,
            )
            d.append(row)

    # from .exact import MetaDataFrame

    tmp = pd.DataFrame(d)
    new_df = pd.merge(df, tmp, left_on="mag", right_on="mag")
    new_df.attrs = df.attrs
    # new_df._name = df._name
    # pd.save_pickle(new_df._name)

    return new_df


def get_p_gibbs(e, beta):
    p = np.exp(-e * beta)
    p /= np.nansum(p)
    # if np.any(np.isnan(p)):
    # p[np.isnan(p)] = 0
    # p /= np.nansum(p)
    # print(p, e, beta, np.exp(-beta * e))
    # assert 0
    return p


def match_temp_exact(t, s, e, theta):
    # match temperature based on Boltzmann distribution
    beta = 1 / t if t > 0 else np.inf
    p = get_p_gibbs(e, beta)
    return np.abs((p @ s) - theta)


def sc_match(t, s, e):
    beta = 1 / t if t > 0 else np.inf
    assert np.isnan(beta) == False
    p = get_p_gibbs(e, beta)
    # optimize therefore we negate the minus here for entropy(!)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nansum(np.log2(p) * p, axis=-1) * np.nansum((p - 1 / len(p)) ** 2)


def match_temp_stc(g, structure=None, e_func=ising):
    # statistical complexity matching procedure

    n = len(g)
    if structure is None:
        states, _ = gen_states(n)
        A = np.asarray(nx.adjacency_matrix(g).todense())
        states, allowed = gen_states(n)
    else:
        A, states, allowed = structure
    print(type(states), type(A))
    E = e_func(states, np.asarray(A))

    E = np.array(E).squeeze().sum(-1)
    # p = np.array([get_transfer(n, E, beta, allowed)[1] for beta in  1/temps])
    s = np.array(abs(states.mean(1) - 0.5) * 2)

    res = optimize.minimize(
        sc_match,
        100,
        args=(
            np.asarray(s),
            np.asarray(E),
        ),
        method="COBYLA",
    )

    print(res)
    if not res.success:
        raise ValueError("Optimalization failed")
    return res.x[0]

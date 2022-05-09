import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy
from functools import partial
from fa2 import ForceAtlas2 as fa2
import multiprocessing as mp
from itertools import product
from scipy import optimize
from experiment import small_tree
from exact import gen_states
from exact import ising, sis, bornholdt

# from exact import gen_states, get_transfer
# from exact import store_results, gen_information_curves

n = 10
T = 200
beta = 0.5


from plexsim.utils.graph import recursive_tree
from exact import simulate, Settings, NodeToSystem, SystemToNode
from exact_utils import get_p_gibbs, match_temp_stc

# lcf graph experiments
def gen_shift(comb):
    n, repeat, shift_list = comb
    g = nx.LCF_graph(n=n, shift_list=shift_list, repeats=repeat)
    # return g, name, params
    yield g, "LCF_graph", dict(
        n=n,
        repeat=repeat,
        shift_list=shift_list,
    )


def run(g, name, params={}, e_func=ising):
    # g = nx.random_tree(10)
    # t = match_temp_stc(g, e_func=e_func)
    t = 1
    beta = 1 / t
    # beta = 0.1

    # make identifier
    s = ""
    for k, v in params.items():
        s += f"_{k}={v}"
    s = f"exact_{name}{s}_dyn={e_func.__name__}_{beta=}_{T=}"

    print(f"beta = {beta}")
    settings = Settings(beta, T, g, NodeToSystem)
    # settings = Settings(beta, T, g, SystemToNode)
    df = simulate(settings, e_func=e_func)
    df.to_pickle(f"./data/{s}.pkl")
    # df.attrs["H"] = H
    # df.attrs["D"] = D


def lemke_graph():
    el = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 0),
        (7, 6),
        (7, 4),
        (7, 2),
        (6, 4),
        (5, 2),
        (3, 6),
    ]
    return nx.from_edgelist(el)


# g = nx.random_tree(10)
# for f in (ising, sis):
# run(nx.krackhardt_kite_graph(), "kite", e_func=f)
# run(g, "random_tree", e_func=f)

# run(nx.krackhardt_kite_graph(), "kite", e_func=ising)
run(nx.krackhardt_kite_graph(), "kite", e_func=sis)

from experiment import small_tree

# run(small_tree(), "small_tree", e_func=ising)

# cycles = np.arange(0, 10).astype(int)
# combs = (gen_shift((10, c, [-2])) for c in cycles)
# for c in combs:
#     p = tuple(c)[0]
#     print(p)
#     run(*p, e_func=sis)

# for c in cycles:
# run(c)
# with mp.Pool(mp.cpu_count() - 1) as p:
# for i in p.imap_unordered(run, combs):
# continue

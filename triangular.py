import matplotlib as mpl

mpl.use("TkAgg")  # or whatever other backend that you want
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
T = 300


from plexsim.utils.graph import recursive_tree
from exact import *
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


def run(g, name, params={}, e_func=ising, target = "./data"):
    # g = nx.random_tree(10)
    t = match_temp_stc(g, e_func=e_func)
    # t = 1
    # t = 1.2
    # t *= 0.95
    beta = 1 / t
    # beta = 1

    # make identifier
    s = ""
    for k, v in params.items():
        s += f"_{k}={v}"
    s = f"exact_{name}{s}_dyn={e_func.__name__}_{beta=}_{T=}_{g.is_directed()=}_NEW"

    print(f"beta = {beta}")
    settings = Settings(beta, T, g, NodeToSystem, mag = [])
    # settings = Settings(beta, T, g, NodeToMacroBit)
    # settings = Settings(beta, T, g, SystemToNode)
    df = simulate(settings, e_func=e_func)
    fp = f"{target}/{s}.pkl"
    df.to_pickle(fp)
    print(f"Saving to {fp=}")
    # df.attrs["H"] = H
    # df.attrs["D"] = D



# g = nx.random_tree(10)
# for f in (ising, sis):
# run(nx.krackhardt_kite_graph(), "kite", e_func=f)
# run(g, "random_tree", e_func=f)

g = nx.florentine_families_graph()
g = nx.krackhardt_kite_graph()
# g.remove_edge(7, 8)
# g.remove_edge(8, 9)
# g.remove_node(8)
# g.remove_node(9)
# g.remove_edge(8, 9)
# g.remove_edge(9, 8)
# g.remove_edge(7, 8)
# g.remove_edge(8, 7)
#
# g.remove_node(8)

# g = nx.path_graph(10)
# nx.draw(g)
# plt.show(block=1)
graphs = pd.read_pickle("graphs_seed=0.pkl")
# graphs = [g]
# graphs =
target = "/run/media/casper/9ee50225-d11d-4dae-81d9-9fa441997327"
for id, g in enumerate(graphs):
    run(g, f"{id}", e_func=ising, target = target)
# run(g, "florentine", e_func=ising)
# run(nx.krackhardt_kite_graph(), "kite", e_func=sis)

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

import proplot as plt, cmasher as cmr, pandas as pd,\
    numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy
import random

def make_sparse(n, p, attempts = 100, seed = None):
    attempt = 0
    while attempt < attempts:
        g = nx.erdos_renyi_graph(n, p)
        if nx.is_connected(g):
            break
        attempt += 1
    # print(f"{attempt=}")
    return g

def make_graphs(n, p, N = 100, seed = 0):
    random.seed(seed)
    graphs = []
    counter = 0
    while counter < N:
       print(counter, end = "\r")
       while True:
           g = make_sparse(n, p, seed = random)
           if graphs:
               if all([nx.is_isomorphic(g, G) == False for G in graphs]):
                   break
           else:
                break
       graphs.append(g)
       counter += 1
    return graphs

def main():
    seed = 0
    graphs = make_graphs(10, 0.2, 100, seed = seed)

    angle = np.linspace(0, 2 * np.pi, len(graphs), 0)
    rad = 15

    np.random.seed(0)
    fig, ax = plt.subplots()
    r = 0
    thresh = 0
    counter = 0
    angle = np.array([0])
    from fa2 import ForceAtlas2
    for g in graphs:
        if counter > thresh:
            thresh += 1
            counter = 0
            r += 35
            angle = np.linspace(0, 2 * np.pi, thresh + 1)

        pos = ForceAtlas2(verbose = 0).forceatlas2_networkx_layout(g, nx.circular_layout(g))
        # pos = nx.kamada_kawai_layout(G)
        # pos = nx.kamada_kawai_layout(g)
        a = angle[counter]
        pos = {node: p + r * np.array([np.cos(a), np.sin(a)]) for node, p in pos.items()}
        nx.draw(g, pos, ax = ax, node_size = 8)
        counter += 1
    ax.axis("equal")
    pd.to_pickle(graphs, f"graphs_{seed=}.pkl")

main()
# fig, ax = plt.subplots(ncols = 2)
# # np.random.seed(0)
# # random = np.random
# g = nx.erdos_renyi_graph(10, 0.2, seed = random)
# G = nx.erdos_renyi_graph(10, 0.2, seed = random)
# nx.draw(g, ax = ax[0], node_size = 5)
# nx.draw(G, ax = ax[1], node_size = 5)
# ax.axis("on")
# fig.show()

plt.show(block = 1)

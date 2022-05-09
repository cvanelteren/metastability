import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy

g = nx.krackhardt_kite_graph()
t = 1 / 0.5732374683235916
m = models.Potts(g, t=t)

m.states = 0
states = np.asarray(m.states)

w = {}
h = nx.DiGraph()
for x, y in g.edges():
    idx = m.adj.mapping[x]
    jdx = m.adj.mapping[y]
    m.states = 0
    states[idx] = 1
    # flip both states
    es = np.zeros(2)
    for kdx, i in enumerate([0, 1]):
        states[jdx] = i
        m.states = states
        e = np.sum(m.siteEnergy(m.states))
        es[kdx] = e
    delta = es[1] - es[0]
    states[idx] = 0
    states[jdx] = 0
    p = 1 / (1 + np.exp(m.beta * (es[1] - es[0])))
    w[(idx, jdx)] = p * 10

    m.states = 0
    states[jdx] = 1
    # flip both states
    es = np.zeros(2)
    for kdx, i in enumerate([0, 1]):
        states[idx] = i
        m.states = states
        e = np.sum(m.siteEnergy(m.states))
        es[kdx] = e
    delta = es[1] - es[0]
    states[idx] = 0
    states[jdx] = 0
    p = 1 / (1 + np.exp(m.beta * (es[1] - es[0])))
    print(idx, jdx, p)
    w[(jdx, idx)] = p * 10
    h.add_edge(idx, jdx)
    h.add_edge(jdx, idx)

pos = nx.kamada_kawai_layout(h)
fig, ax = plt.subplots()
nx.draw_networkx_nodes(h, pos)
nx.draw_networkx_labels(h, pos)
nx.draw_networkx_edges(h, pos, width=list(w.values()))
fig.show()
plt.show(block=1)

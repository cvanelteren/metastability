import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy
from exact import gen_states, to_binary, gen_binary


def gen_states_node(n) -> tuple:
    """
    Creates binary state space and returns allowed transitions
    """
    states = np.zeros((2**n, n))
    allowed = {}
    # ntrans = {}
    from tqdm import tqdm

    for idx in tqdm(range(2**n)):
        states[idx] = gen_binary(idx, n)
        for node in range(n):
            state = states[idx].copy()
            node_state = state[node]
            if state[node] == 0:
                state[node] = 1
            else:
                state[node] = 0
            kdx = to_binary(state)
            allowed[(idx, node, node_state)] = allowed.get(idx, []) + [kdx]
    return states, allowed


n = 10
states, allowed = gen_states_node(n)
A = nx.adjacency_matrix(nx.path_graph(n)).todense()

p = np.zeros((2**n, 2**n, n, 2))

E = -np.multiply(states * 2 - 1, (states * 2 - 1) @ A)
beta = 1.0

for (idx, node, node_state), trans in allowed.items():
    s = states[idx]
    # for node, s_i in enumerate(s):
    # p[idx, node, s_i]
    e1 = E[idx].sum()
    for other in trans:
        s = states[other]
        e2 = E[other].sum()
        p[idx, other, node, int(node_state)] = 1 / (1 + np.exp(beta * (e2 - e1)))

from tqdm import tqdm

for node in tqdm(range(n)):
    np.fill_diagonal(p[..., node, 0], 1 - p[..., node, 0].sum(1))
    np.fill_diagonal(p[..., node, 1], 1 - p[..., node, 1].sum(1))

d = p[..., 0, 0] - p[..., 0, 1]
fig, ax = plt.subplots()
h = ax.imshow(d)
ax.colorbar(h)
ax.format()
fig.show()
plt.show(block=1)

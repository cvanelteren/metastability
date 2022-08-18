import proplot as plt, cmasher as cmr, pandas as pd,\
    numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy

graphs = pd.read_pickle("graphs_seed=0.pkl")
g = graphs[0]

temps = np.linspace(0, 10, 100)

N = int(1e5)
from exact import gen_states, ising

A = nx.adjacency_matrix(g).todense()
states, allowed = gen_states(len(g))
E = ising(states, A)

# simulate for different temperatures and extract the
# noise below and above the tipping point
seeds = (
    0,
    12,
    123,
    1234,
    12345,
    123456,
)

from isi import estimate_white_noise
combs = product(seeds, temps)

df = []
for (seed, t) in tqdm(combs):
    m = Potts(m, t = t, sampleSize = 1, seed = seed)
    s = m.simulate(N).mean(1)
    output = estimate_white_noise(s)
    output["seed"] = seed
    output["t"] = t
    df.append(output)
df = pd.DataFrame(df)
df.to_pickle("graph0_test_sc_vs_resolution.pkl")

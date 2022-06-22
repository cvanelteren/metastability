import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy

from tqdm import tqdm

from experiment import small_tree


g = nx.krackhardt_kite_graph()
beta = 0.5732374683235916

# g = small_tree()
# beta = 0.9001444450539172
n_steps = 10e6
sample_size = 1

df = []


def get_success(s, n):
    before = np.sign(s[: n // 2].mean(1) - 0.5)
    a = set(before)

    after = np.sign(s[n // 2 + 1 :].mean(1) - 0.5)
    b = set(after)

    o = a.intersection(b)
    x3 = False
    if o == {0.0} or len(o) == 0:
        # if all([o == {0.0}, len(a) >= 1 and len(b)]) >= 1 or o == {}:
        x3 = True

    if x3:
        return 1
    return 0


def get_isi(seed, nudges={}, max_t=50_000, n=100):
    m = models.Potts(
        g,
        t=1 / beta,
        sampleSize=sample_size,
        seed=seed,
    )
    m.nudges = nudges
    m.states = 0

    tmp = np.zeros((max_t, len(g)))
    tips = np.array([])
    p = {}

    buffer = np.zeros((n, m.nNodes))

    success, num_tips, counter = 0, 0, 0
    detected_tipping = False
    for step in tqdm(range(int(n_steps))):
        buffer[-1] = m.updateState(m.sampleNodes(1)[0])
        if step < max_t:
            tmp[step] = buffer[-1]

        if np.mean(buffer[-1]) == 0.5:
            detected_tipping = True
            counter = 0
            num_tips += 1
            s = tuple(buffer[-1])
            p[s] = p.get(s, 0) + 1
            tips = np.append(tips, step)

        if counter == n // 2 - 1 and detected_tipping:
            assert buffer[n // 2].mean() == 0.5
            success += get_success(buffer, n)
            # print(f"{success=}", end="\r")
            detected_tipping = False

        counter += 1
        counter = counter % n
        buffer = np.roll(buffer, -1, axis=0)
    p = {x: y / sum(p.values()) for x, y in p.items()}
    isi = np.diff(tips)
    return (
        isi,
        tmp[:max_t],
        p,
        success,
        num_tips,
    )


def run(x):
    seed, nudge = x
    df = []
    isi, system, p, success, num_tips = get_isi(seed)

    row = dict(
        label="control",
        isi=isi,
        seed=seed,
        system=system,
        nudge=nudge,
        p=p,
        success=success,
        num_tips=num_tips,
    )
    df.append(row)
    print(f"{success=} {num_tips=} {success/num_tips} {row['label']} {row['nudge']}")

    for node in tqdm(g.nodes()):
        nudges = {node: nudge}
        isi, system, p, sucess, num_tips = get_isi(seed, nudges)
        print(f"{success=}")
        row = dict(
            label=node,
            isi=isi,
            seed=seed,
            system=system,
            nudge=nudge,
            p=p,
            success=success,
            num_tips=num_tips,
        )
        df.append(row)
    return df


import multiprocessing as mp
from itertools import product

seeds = (
    0,
    12,
    123,
    1234,
    12345,
    123456,
    # 1234567,
)
nudges = np.linspace(0, 10, 10)
nudges = np.array([np.inf])
combs = product(seeds, nudges)

with mp.Pool(mp.cpu_count() - 1) as p:
    for i in p.imap_unordered(
        run,
        combs,
    ):
        for j in i:
            df.append(j)

df = pd.DataFrame(df)
# df.to_pickle(f"small_tree_isi_{beta=}.pickle")
df.to_pickle(f"./kite_isi_{beta=}.pkl")
print(df)

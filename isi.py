import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from tqdm import tqdm
from experiment import small_tree
from exact import *
from exact_utils import get_p_gibbs, match_temp_stc
import multiprocessing as mp
from itertools import product
from figure4 import make_windows


def get_success(s, n) -> int:
    """
    Check if the metastable transition flipped sign

    return -1 if  transition is from 0->1
    returns 0 if no valid transition was found
    returns 1 if transition is from 1 -> 0
    """
    before = np.sign(s[: n // 2].mean(1) - 0.5)
    a = set(before)

    after = np.sign(s[n // 2 + 1 :].mean(1) - 0.5)
    b = set(after)

    o = a.intersection(b)
    x3 = False
    if o == {0.0} or len(o) == 0:
        x3 = True

    if x3:
        if -1 in a:
            return -1
        elif 1 in a:
            return 1
    return 0


def get_isi(g, beta, seed, sample_size=1, nudges={}, max_t=10_000, n=500):
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

    success_0_to_1, success_1_to_0, num_tips, counter = 0, 0, 0, 0
    detected_tipping = False
    for step in range(int(n_steps)):
        # for step in tqdm(range(int(n_steps)), position = mp.current_process()._identity[0] + 1):
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

            success = get_success(buffer, n)
            if success == -1:
                success_0_to_1 += 1
            elif success == 1:
                success_1_to_0 += 1

            # print(f"{success=}", end="\r")
            detected_tipping = False

        counter += 1
        counter = counter % n
        buffer = np.roll(buffer, -1, axis=0)
    p = {x: y / sum(p.values()) for x, y in p.items()}
    isi = np.diff(tips)
    return (
        isi,
        tmp,
        p,
        success_0_to_1,
        success_1_to_0,
        num_tips,
    )


def estimate_white_noise(macrostate, label, tipping=0.5):
    from scipy.stats import sem

    output = {}
    # print(row.label, np.where(macrostate > tipping)[0].size / macrostate.size)
    for idx, op in enumerate((np.greater, np.less)):
        where = np.where(op(macrostate, tipping))[0]
        # less_n, less_w, less_t
        other_name = op.__name__ + "_n"  # fraction of time spent spent
        name = (
            op.__name__ + "_w"
        )  # second moment : past me thought it was white noise hence w
        num_tips = (
            op.__name__ + "_t"
        )  # number of winodws, could be used to infer frequency

        output[other_name] = where.size / macrostate.size
        # output[other_name] = (macrostate.size - len(where)) / (macrostate.size)
        # output[op.__name__ + "_w"] = (tmp**2).sum()

        output[name] = 0
        # output[other_name] = 0
        windows = make_windows(where)
        output[num_tips] = len(windows)
        for window in windows:
            d = macrostate[where[window]]
            # d = sem(d, nan_policy="omit")
            a = 1
            if op == np.greater:
                d = abs(1 - d)
            if label != "control" and op == np.greater:
                a = 4 / 5
            d = np.sum(d**2) / (a**2 * where.size)
            output[name] += d  # / macrostate.size
            # output[other_name] += len(window) / len(windows)
    return output


def run(x):
    g, beta, seed, nudge = x
    df = []
    to_run = [({}, "control")]
    [to_run.append(({node: nudge}, node)) for node in g.nodes()]
    for (nudges, node) in to_run:
        isi, system, p, success_0_to_1, success_1_to_0, num_tips = get_isi(
            g, beta, seed, nudges=nudges
        )
        tmp = estimate_white_noise(system.mean(1), node)
        row = dict(
            label=node,
            isi=isi,
            seed=seed,
            system=system[:10_000],
            nudge=nudge,
            p=p,
            success_0_to_1=success_0_to_1,
            success_1_to_0=success_1_to_0,
            num_tips=num_tips,
            id=g.id,
            beta=beta,
        )
        for k, v in tmp.items():
            row[k] = v
        # print(f"{success=} {num_tips=} {success/num_tips} {row['label']} {row['nudge']}", end = "\r")
        df.append(row)
    return df


e_func = ising
n_steps = 1e5
sample_size = 1

seeds = (
    0,
    12,
    123,
    1234,
    12345,
    123456,
    1234567,
    10,
    11,
    12,
    13,
    14,
)
nudges = np.linspace(0, 10, 10)
nudges = np.array([np.inf])
betas = []
graphs = pd.read_pickle("graphs_seed=0.pkl")
# graphs = [nx.krackhardt_kite_graph()]
def setup_sim(g, id):
    t = match_temp_stc(g, e_func=e_func)
    g.id = id
    return g, 1 / t


with mp.Pool(mp.cpu_count() - 1) as p:
    for id, g in enumerate(tqdm(graphs, position=0)):
        print(id)
        df = []
        g, best_beta = setup_sim(g, id)
        beta = 1 / np.linspace(0, 10, 20)
        gs = [g]
        combs = product(gs, beta, seeds, nudges)

        for i in p.imap_unordered(run, combs):
            for j in i:
                df.append(j)
        df = pd.DataFrame(df)
        df["best_beta"] = best_beta
        df.to_pickle(f"./kite_isi_{g.id=}_ts.pkl")
        break

# df.to_pickle(f"small_tree_isi_{beta=}.pickle")
# df.to_pickle(f"./kite_isi_{beta=}.pkl")

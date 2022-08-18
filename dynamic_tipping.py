import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy


""" write an experiment where  we obtain the distribution of
states as a function of the tipping point.

"""


from exact import to_binary


def bin_data(buffer: np.ndarray, p: np.ndarray, Z: int, center=0):
    t, n = buffer.shape
    for ti in range(t):
        s = tuple(buffer[ti])
        T = -(t - ti - 1)
        # T = -(t - ti - 1 - center)
        # p[T][s] = p[T].get(default, 0) + 1 / Z

        # estimate joint distribution
        idx = to_binary(buffer[ti])
        for node, s in enumerate(buffer[0]):
            p[abs(T), idx, node, int(s)] += 1 / Z


def get_tipping(m, n: int, N: int) -> dict:
    buffer = np.zeros((n, m.nNodes))

    num_tips = 0
    p = np.zeros((n, 2**10, 10, 2))
    # p = {-ti: {} for ti in range(n)}
    # m.states =
    # assert np.mean(m.states) == 0
    m.simulate(1e6)
    while num_tips < N:
        if buffer[-1].mean() == 0.5:

            # if np.all(
            #     np.sign(buffer[: n // 2 - 1].mean(1) - 0.5)
            #     != np.sign(buffer[n // 2 + 1 :].mean(1) - 0.5)
            # ):

            # /               print(
            #                    buffer[: n // 2 - 1].mean(1) - 0.5,
            #                    buffer[n // 2 + 1 :].mean(1) - 0.5,
            #                    buffer[n // 2 + 1 :].mean(1),
            #                    buffer[: n // 2 - 1].mean(1),
            #                )
            #                assert 0
            # only bin from below
            # idx = np.where(buffer.mean(1) > 0.5)
            # if idx:
            # buffer[idx] = 1 - buffer[idx]
            if np.all(np.sign(buffer[:-1].mean() - 0.5) < 0):
                bin_data(buffer, p, N)
                num_tips += 1
                print(f"Found {num_tips}", end="\r")
            # tmp = np.sign(buffer[:-1, :].mean(1) - 0.5)
            # # print(tmp.mean(), np.all(tmp == 1) or np.all(tmp == -1))
            # if tmp.mean() >= 1:
            #     bin_data(buffer, p, N)
            #     num_tips += 1
            #     print(f"Found {num_tips}", end="\r")
            # elif tmp.mean() <= -1:
            #     buffer = abs(buffer - 1)
            #     bin_data(buffer, p, N)
            #     num_tips += 1
            #     print(f"Found {num_tips}", end="\r")

        buffer[:] = m.simulate(n)

        # m.simulate(n)
        # for ti in range(n - 1, 0, -1):
        # buffer[ti - 1] = buffer[ti]
        # s = np.array(m.updateState(m.sampleNodes(1)[0]))
        # buffer[-1, :] = s
    return p


g = nx.krackhardt_kite_graph()
remove = False
if remove:
    g.remove_edge(7, 8)
    g.remove_edge(8, 9)

# fig, ax = plt.subplots()
# nx.draw(g, ax=ax)
# plt.show(block=1)
# fig.show()

t = 1 / 0.5732374683235916
# t *= 0.8
seed = 12345
m = models.Potts(g, t=t, seed=seed, sampleSize=1)

# fig, ax = plt.subplots()
# ax.plot(np.array(m.simulate(3e5)).mean(1))
# fig.show()
# plt.show(block=1)

print(m.t)
N = 1000_000
N = 10000
w = 30
p = get_tipping(m, w, N)
data = [dict(p=p, t=t, g=g, N=N, w=w)]
df = pd.DataFrame(data)
df.to_pickle(f"dynamic_tipping_kite_single_side_{w=}_{N=}_{remove=}.pkl")

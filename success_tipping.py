import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy
from tqdm import tqdm
from tqdm.notebook import tqdm


def get_success(row, n=100):
    s = np.stack(row.system).mean(1)
    idxs = np.where(s == 0.5)[0]
    success = 0
    N = 0
    for idx in idxs:
        N += 1
        if idx < n or idx > len(s) - n:
            continue

        before = np.sign(s[idx - n : idx] - 0.5)
        a = set(before)

        # if np.mean(before) < 0:
        #     x1 = np.all(before <= 0)
        # elif np.mean(before) > 0:
        #     x1 = np.all(before >= 0)
        # else:

        after = np.sign(s[idx + 1 : idx + n] - 0.5)
        b = set(after)

        # if np.mean(after) < 0:
        #     x2 = np.all(after <= 0)
        # else:
        #     x2 = np.all(after >= 0)

        # x2 = np.all(after == after[0])
        o = a.intersection(b)
        x3 = False
        if all([o == {0.0}, len(a) >= 1 and len(b)]) >= 1 or o == {}:
            x3 = True

        if len(a) != 3 and len(b) != 3 and x3:
            success += 1
    # print(success, N, idxs.size, row.label, row.nudge)
    return success, N


beta = 0.5732374683235916
fp = f"./kite_isi_{beta=}.pkl"
df = pd.read_pickle(fp)
print(df)

df["success"] = df.apply(get_success, axis=1)
df.to_pickle(f"{fp}_tmp.pkl")

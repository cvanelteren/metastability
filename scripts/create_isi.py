import numpy as np, networkx as nx, pandas as pd
from plexsim.models import Potts
from pathlib import Path
from imi.signal import wait_tipping

base = Path("data")


def do_isi(data):
    assert "file" in data

    # if "tips" in data:
    # print("tips already done")
    # return
    m = Potts(**data["model_settings"])
    n_window = 1000
    n_tipping = 100
    M = int(1e5)
    bins = np.linspace(0, 1.05, 20)
    # m.sampleSize = m.nNodes
    print(m.sampleSize)
    snapshots, tips = wait_tipping(
        m, bins, n_window, n_tipping, allowance=1, n_equilibrate=M
    )
    data["tips"] = tips
    pd.to_pickle(data, base / data["file"])
    return data


if __name__ == "__main__":
    df = pd.read_pickle("./data/database.pkl")
    for file in df.file.unique():
        data = pd.read_pickle(base / file)
        data["file"] = file
        do_isi(data)

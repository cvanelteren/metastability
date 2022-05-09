from pathlib import Path
from utils import do_fit, do_halfimi, get_rmse
from tqdm import tqdm
from plexsim.models import Potts
import pandas as pd, numpy as np

if __name__ == "__main__":
    df = []
    system = 0
    base = Path("./data")
    for fp in tqdm(base.iterdir()):
        if fp.name.endswith("0.8.pkl"):
            data = pd.read_pickle(fp)
            data["file"] = fp.name
            row = dict(system=system, states=np.asarray([i for i in data["df"].bin]))
            df.append(row)
            system += 1
    pd.DataFrame(df).to_pickle(base / "state_dist.pkl")

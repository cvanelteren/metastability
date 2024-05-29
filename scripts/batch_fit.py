from pathlib import Path
from utils import do_fit, do_halfimi, get_rmse
from tqdm import tqdm
from plexsim.models import Potts
import pandas as pd, numpy as np

f = lambda x, a, b, c, d, g: a * np.exp(-b * x) + c * np.exp(-d * x) + g
# f = lambda x, a, b, c, d: a * np.exp(-b * x) + c * np.exp(-d * x)
# f = (
#     lambda x, a, b, c, d, e, f, g: a * np.exp(-b * (x - c))
#     + d * np.exp(-e * (x - f))
#     + g
# )

# f = lambda x, a, b, c: a * np.exp(-b * x) + c

from utils import compute_abs_mi_decay

redo = True
if __name__ == "__main__":
    df = []
    system = 0
    base = Path("./data")
    for fp in tqdm(base.iterdir()):
        if fp.name.endswith("0.8.pkl"):
            print(f"Reading {fp=}")
            data = pd.read_pickle(fp)
            data["file"] = fp.name
            data = compute_abs_mi_decay(data)
            coeffs = do_fit(data, f=f, redo=redo)
            half_imi = do_halfimi(data, f=f, redo=redo)
            mses = get_rmse(data, f=f)
            m = Potts(**data["model_settings"])
            for node, idx in m.adj.mapping.items():
                deg = m.graph.degree(node)
                for k, v in half_imi.items():
                    node_coeffs = v["coeffs"][idx]
                    node_halfimi = v["halfimi"][:, idx]
                    mag = round(float(k), 2)
                    row = dict(
                        system=system,
                        node=node,
                        deg=deg,
                        coeff=node_coeffs,
                        imi=node_halfimi[0],
                        half=node_halfimi[2],
                        asymp=node_halfimi[1],
                        mag=mag,
                        mse=mses[idx],
                        file=fp.name,
                    )
                    df.append(row)
            system += 1
    pd.DataFrame(df).to_pickle(base / "database.pkl")

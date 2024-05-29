import proplot as plt, cmasher as cmr, pandas as pd, numpy as np, os, sys, networkx as nx, warnings
from plexsim import models
from imi import infcy
from pathlib import Path
import re

f = lambda x, a, b, g: a * np.exp(-b * x) + g
ff = lambda x, a, b, c, d, g: a * np.exp(-b * x) + c * np.exp(-d * x) + g
fff = (
    lambda x, a, b, c, d, e, f, g: a * np.exp(-b * x)
    + c * np.exp(-d * x)
    + e * np.exp(-f * x)
    + g
)
ffff = (
    lambda x, a, b, c, d, e, f, h, i, g: a * np.exp(-b * x)
    + c * np.exp(-d * x)
    + e * np.exp(-f * x)
    + h * np.exp(-i * x)
    + g
)
from exact_utils import fit_curve
from tqdm import tqdm

tmp = []
base = Path("./data")
for fp in base.iterdir():
    if (
        fp.name.startswith("exact_recursive_tree_4_beta=0.567")
        and "order" not in fp.name
    ):
        df = pd.read_pickle(fp)
        n_samples = int(re.findall("T=\d+", fp.name)[0].split("=")[1])
        for idx, fi in tqdm(enumerate((f, ff, fff, ffff))):
            name = base / (fp.name.rstrip(".pkl") + f"_order={idx + 1}.pkl")
            ndf = fit_curve(df, fi, offset=True)
            pd.to_pickle(ndf, name)

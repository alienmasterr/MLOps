from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--month", required=True)
ap.add_argument("--mode", choices=["null_text", "bad_range"], required=True)
a = ap.parse_args()

df = pd.read_parquet(Path("data/raw") / f"{a.month}.parquet")
rng = np.random.default_rng(0)

if a.mode == "null_text":
    idx = rng.choice(len(df), size=int(0.30 * len(df)), replace=False)
    df.loc[df.index[idx], "comment_text"] = None
else:
    idx = rng.choice(len(df), size=int(0.05 * len(df)), replace=False)
    df.loc[df.index[idx], "toxicity"] = 1.7

out = Path("data/broken")
out.mkdir(parents=True, exist_ok=True)
df.to_parquet(out / f"{a.month}.parquet", index=False)
print(f"wrote data/broken/{a.month}.parquet ({a.mode}, {len(idx)} rows broken)")
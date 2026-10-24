"""
python scripts/sample_data.py --src /path/to/all_data.csv --out data/raw
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEEP = [
    "id", "comment_text", "created_date", "publication_id", "parent_id", "rating", "toxicity",
    "male", "female", "black", "white", "muslim", "jewish", "christian",
    "homosexual_gay_or_lesbian", "psychiatric_or_mental_illness",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--start", default="2016-03")
    ap.add_argument("--end", default="2017-10")
    ap.add_argument("--per-month", type=int, default=25_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    keep: dict[str, pd.DataFrame] = {}

    for chunk in pd.read_csv(Path(args.src).expanduser(), usecols=KEEP, chunksize=200_000):
        ts = pd.to_datetime(chunk["created_date"], format="ISO8601", utc=True)
        chunk = chunk.assign(created_date=ts, _month=ts.dt.strftime("%Y-%m"), _r=rng.random(len(chunk)))
        chunk = chunk[(chunk["_month"] >= args.start) & (chunk["_month"] <= args.end)]
        for month, part in chunk.groupby("_month"):
            prev = keep.get(month)
            merged = part if prev is None else pd.concat([prev, part])
            keep[month] = merged.nsmallest(args.per_month, "_r")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("month    rows  toxic_share")
    for month, part in sorted(keep.items()):
        part = part.drop(columns=["_month", "_r"]).sort_values("id").reset_index(drop=True)
        part.to_parquet(out / f"{month}.parquet", index=False)
        print(f"{month}  {len(part):6d}  {(part['toxicity'] >= 0.5).mean():.3f}")


if __name__ == "__main__":
    main()
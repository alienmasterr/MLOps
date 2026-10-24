from __future__ import annotations

import argparse
import logging

import pandas as pd

from mlops import core

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="2016-03")
    ap.add_argument("--to", dest="end", default="2017-05")
    ap.add_argument("--raw-dir", default=None)
    a = ap.parse_args()
    engine = core.get_engine()
    for m in pd.period_range(a.start, a.end, freq="M").strftime("%Y-%m"):
        df = core.extract(a.raw_dir, m)
        core.validate_schema(df)
        clean, stats = core.clean(df, m)
        core.load(engine, clean, m)
        core.build_features(engine, m)
        core.quality_check(engine, m, stats)


if __name__ == "__main__":
    main()
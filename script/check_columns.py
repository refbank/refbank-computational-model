#!/usr/bin/env python3
"""Quick check: print column names for trials, messages, and choices tables."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import redivis

redivis.authenticate()
ds = redivis.user("mcfrank").dataset("refbank", version="current")

for name, tag in [
    ("trials",   "trials:zkj2"),
    ("messages", "messages:2q18"),
    ("choices",  "choices:s1zj"),
]:
    df = ds.query(f"SELECT * FROM `{tag}` LIMIT 1").to_pandas_dataframe()
    print(f"\n{name}: {list(df.columns)}")

#!/usr/bin/env python
"""Print a compact summary of an ASE DB using the project's row.data schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from material_feature_fusion.data import summarize_database


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datapath")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    summary = summarize_database(args.datapath, limit=args.limit)
    print(summary)


if __name__ == "__main__":
    main()

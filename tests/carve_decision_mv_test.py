# -*- coding: utf-8 -*-
"""
Carve rows with missing decision (`dec` / last column) out as held-out test data.

Training set  = rows with known decision (used for imputation / fitting)
Test set      = rows with missing decision (evaluated after training)

Writes:
  <stem>_train.txt
  <stem>_test.txt
for each listed source file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nmilib import carve_and_save_train_test
DEFAULT_SOURCES = ("data2.txt", "data3.txt", "data4.txt", "data5.txt")
NA_VALUES = ["", "?"]


def carve_source(src_name: str, decision_col: str = "dec") -> dict:
    src = ROOT / src_name
    if not src.exists():
        raise FileNotFoundError(src)
    stem = src.stem
    df = pd.read_csv(src, na_values=NA_VALUES)
    if decision_col not in df.columns:
        decision_col = df.columns[-1]

    train_path = ROOT / f"{stem}_train.txt"
    test_path = ROOT / f"{stem}_test.txt"
    train, test = carve_and_save_train_test(
        df, train_path, test_path, decision_col=decision_col
    )
    return {
        "source": src_name,
        "decision_col": decision_col,
        "n_total": len(df),
        "n_train": len(train),
        "n_test": len(test),
        "train_file": train_path.name,
        "test_file": test_path.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        nargs="*",
        default=list(DEFAULT_SOURCES),
        help="Source .txt files to split (default: data2..data5)",
    )
    parser.add_argument("--decision-col", default="dec")
    args = parser.parse_args()

    for name in args.sources:
        info = carve_source(name, decision_col=args.decision_col)
        print(
            f"{info['source']}: total={info['n_total']} → "
            f"train={info['n_train']} ({info['train_file']}), "
            f"test={info['n_test']} ({info['test_file']}) "
            f"[decision={info['decision_col']}]"
        )


if __name__ == "__main__":
    main()

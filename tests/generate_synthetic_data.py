# -*- coding: utf-8 -*-
"""
Generate master (complete) and test (with '?') datasets for data3/4/5.

Master files contain all true values.
Test files are copies with '?' missing values in attributes and/or `dec`.
Total missing-value cells stay strictly below 25% of all cells; rows that
contain any '?' also stay at most 25% of rows.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPECS = (
    ("data3", 2),
    ("data4", 3),
    ("data5", 4),
)


def _write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    lines = [",".join(headers)]
    lines.extend(",".join(row) for row in rows)
    path.write_text("\n".join(lines) + "\n")


def generate_master(
    n_attrs: int,
    n_rows: int = 100,
    seed: int = 42,
) -> tuple[list[str], list[list[str]]]:
    """Build a complete dataset (no missing values)."""
    rng = random.Random(seed + n_attrs)
    cats = ["yes", "no"]
    decs = ["+", "-"]
    headers = [f"at{i}" for i in range(1, n_attrs + 1)] + ["dec"]

    rows: list[list[str]] = []
    for _ in range(n_rows):
        row: list[str] = []
        for i in range(1, n_attrs + 1):
            if i == 1:
                row.append(rng.choice(cats))
            else:
                row.append(f"{rng.uniform(8.0, 18.0):.1f}")
        row.append(rng.choice(decs))
        rows.append(row)
    return headers, rows


def inject_missing_values(
    rows: list[list[str]],
    n_attrs: int,
    max_mv_frac: float = 0.25,
    seed: int = 42,
) -> tuple[list[list[str]], dict]:
    """
    Return a copy of rows with '?' injected in attributes and/or `dec`.

    Constraints:
    - Fraction of rows with any '?' <= max_mv_frac (and < 0.25 if max is 0.25)
    - Fraction of all cells that are '?' < max_mv_frac
    - At least one `dec` value is missing
    - At least one attribute value is missing
    """
    rng = random.Random(seed + 1000 + n_attrs)
    n = len(rows)
    n_cols = n_attrs + 1  # attributes + dec
    total_cells = n * n_cols

    # Strictly less than 25% missing cells and at most 25% MV rows
    max_mv_cells = max(1, int(total_cells * max_mv_frac) - 1)  # < 25%
    max_mv_rows = min(int(n * max_mv_frac), n)
    if max_mv_rows < 1:
        max_mv_rows = 1

    # Leave enough complete donor rows (>= 50% complete)
    max_mv_rows = min(max_mv_rows, n // 2)

    mv_row_indices = sorted(rng.sample(range(n), max_mv_rows))
    out = [list(r) for r in rows]
    mv_cells = 0
    dec_mv = 0
    attr_mv = 0

    for i, r in enumerate(mv_row_indices):
        # Decide what to mask in this row: attrs only, dec only, or both
        mode = rng.choice(["attr", "attr", "dec", "both"])  # bias toward attrs
        if i == 0:
            mode = "both"  # guarantee both kinds appear
        elif i == 1 and dec_mv == 0:
            mode = "dec"

        if mode in ("attr", "both") and mv_cells < max_mv_cells:
            n_mask = rng.randint(1, min(2, n_attrs))
            # don't exceed remaining cell budget
            n_mask = min(n_mask, max_mv_cells - mv_cells)
            if n_mask > 0:
                cols = rng.sample(range(n_attrs), n_mask)
                for c in cols:
                    if out[r][c] != "?":
                        out[r][c] = "?"
                        mv_cells += 1
                        attr_mv += 1

        if mode in ("dec", "both") and mv_cells < max_mv_cells:
            if out[r][-1] != "?":
                out[r][-1] = "?"
                mv_cells += 1
                dec_mv += 1

        # If nothing was masked (budget edge), force one attr MV
        if all(out[r][c] != "?" for c in range(n_cols)) and mv_cells < max_mv_cells:
            c = rng.randrange(n_attrs)
            out[r][c] = "?"
            mv_cells += 1
            attr_mv += 1

    # Ensure at least one dec MV if still none
    if dec_mv == 0 and mv_cells < max_mv_cells:
        r = mv_row_indices[-1]
        if out[r][-1] != "?":
            out[r][-1] = "?"
            mv_cells += 1
            dec_mv += 1

    # Drop empty MV rows from count (shouldn't happen)
    actual_mv_rows = [r for r in range(n) if any(v == "?" for v in out[r])]
    stats = {
        "mv_rows": len(actual_mv_rows),
        "mv_row_frac": len(actual_mv_rows) / n,
        "mv_cells": mv_cells,
        "mv_cell_frac": mv_cells / total_cells,
        "attr_mv": attr_mv,
        "dec_mv": dec_mv,
        "total_cells": total_cells,
    }
    return out, stats


def generate_pair(
    stem: str,
    n_attrs: int,
    n_rows: int = 100,
    seed: int = 42,
    max_mv_frac: float = 0.25,
) -> dict:
    headers, master_rows = generate_master(n_attrs, n_rows=n_rows, seed=seed)
    test_rows, stats = inject_missing_values(
        master_rows, n_attrs, max_mv_frac=max_mv_frac, seed=seed
    )

    master_path = ROOT / f"{stem}_master.txt"
    test_path = ROOT / f"{stem}.txt"
    _write_csv(master_path, headers, master_rows)
    _write_csv(test_path, headers, test_rows)

    return {
        "stem": stem,
        "n_attrs": n_attrs,
        "n_rows": n_rows,
        "master": master_path.name,
        "test": test_path.name,
        **stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-mv-frac",
        type=float,
        default=0.25,
        help="Upper bound for MV row/cell fraction (must be <= 0.25; cells stay < this).",
    )
    args = parser.parse_args()
    if args.max_mv_frac > 0.25:
        raise SystemExit("max-mv-frac must be <= 0.25")

    for stem, n_attrs in SPECS:
        info = generate_pair(
            stem,
            n_attrs,
            n_rows=args.rows,
            seed=args.seed,
            max_mv_frac=args.max_mv_frac,
        )
        print(
            f"{info['master']} + {info['test']}: "
            f"{info['n_rows']} rows, {info['n_attrs']} attrs, "
            f"MV rows={info['mv_rows']} ({info['mv_row_frac']:.1%}), "
            f"MV cells={info['mv_cells']}/{info['total_cells']} "
            f"({info['mv_cell_frac']:.1%}), "
            f"attr_mv={info['attr_mv']}, dec_mv={info['dec_mv']}"
        )

    # Carve missing-decision rows as held-out test files
    from carve_decision_mv_test import carve_source

    print("\nCarving missing-decision rows as test data:")
    for stem, _n_attrs in SPECS:
        info = carve_source(f"{stem}.txt", decision_col="dec")
        print(
            f"  {info['source']}: train={info['n_train']} → {info['train_file']}, "
            f"test={info['n_test']} → {info['test_file']}"
        )


if __name__ == "__main__":
    main()

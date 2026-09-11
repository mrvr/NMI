# -*- coding: utf-8 -*-
"""
System test: imputation accuracy + timing on dengue.csv

Pipeline
--------
1. Load dengue.csv (header in first row).
2. Build a temporary complete master (no missing values).
3. Create a test set by randomly masking attribute cells
   (never mask the decision column ``Dengue``).
4. Run vectorized non_parametric_imputation and compare to the master.
5. Report accuracy and wall-clock / complexity timing.

Clinical setup
--------------
Features: Fever, Headache, JointPain, Bleeding
Decision: Dengue (1 = positive, 0 = negative)

``Name`` is dropped (patient identifier, not used for proximity).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nmilib import non_parametric_imputation

ROOT = Path(__file__).resolve().parents[1]
DENGUE_CSV = ROOT / "dengue.csv"
DECISION_COL = "Dengue"
ID_COL = "Name"
FEATURE_COLS = ["Fever", "Headache", "JointPain", "Bleeding"]

SAMPLE_ROWS = 10_000
SCALE_RUNS = (10_000, 20_000, 50_000)
MV_CELL_FRAC = 0.12
RNG_SEED = 42


def load_dengue(nrows: int | None = None) -> pd.DataFrame:
    assert DENGUE_CSV.exists(), f"Missing dengue data file: {DENGUE_CSV}"
    return pd.read_csv(DENGUE_CSV, nrows=nrows)


def build_temporary_master(
    df: pd.DataFrame,
    *,
    sample_rows: int = SAMPLE_ROWS,
    seed: int = RNG_SEED,
) -> pd.DataFrame:
    """
    Temporary master: complete rows only, no missing values, fixed sample size.

    Keeps Fever/Headache/JointPain/Bleeding + Dengue; drops Name.
    """
    work = df.copy()
    if ID_COL in work.columns:
        work = work.drop(columns=[ID_COL])

    assert work.columns[-1] == DECISION_COL, (
        f"Expected last column '{DECISION_COL}', got '{work.columns[-1]}'"
    )
    for col in FEATURE_COLS:
        assert col in work.columns, f"Missing feature column {col}"

    work = work.dropna().reset_index(drop=True)
    assert len(work) > 0, "No complete rows available for master"
    assert not work.isna().any().any(), "Master must have no missing values"

    n = min(sample_rows, len(work))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(work), size=n, replace=False)
    master = work.iloc[np.sort(idx)].reset_index(drop=True)
    # Keep feature order + decision last
    master = master[[*FEATURE_COLS, DECISION_COL]]
    assert not master.isna().any().any()
    return master


def inject_random_missing(
    master: pd.DataFrame,
    *,
    mv_frac: float = MV_CELL_FRAC,
    seed: int = RNG_SEED,
    decision_col: str = DECISION_COL,
) -> tuple[pd.DataFrame, list[tuple[int, str]]]:
    """Randomly remove values from feature columns only (never decision_col)."""
    test = master.copy()
    feature_cols = [c for c in test.columns if c != decision_col]
    assert feature_cols, "Need at least one attribute column to mask"

    rng = np.random.default_rng(seed)
    n_rows = len(test)
    n_cols = len(feature_cols)
    n_cells = n_rows * n_cols
    n_mask = max(1, int(round(mv_frac * n_cells)))
    n_mask = min(n_mask, n_cells)

    flat = rng.choice(n_cells, size=n_mask, replace=False)
    masked: list[tuple[int, str]] = []
    for pos in flat:
        r, c = divmod(int(pos), n_cols)
        col = feature_cols[c]
        test.at[r, col] = np.nan
        masked.append((r, col))

    assert test[decision_col].notna().all()
    assert test[decision_col].equals(master[decision_col])
    return test, masked


def values_equal(imputed, truth) -> bool:
    if pd.isna(imputed) or pd.isna(truth):
        return False
    try:
        return bool(np.isclose(float(imputed), float(truth), rtol=0.0, atol=0.0))
    except (TypeError, ValueError):
        return str(imputed) == str(truth)


def compare_imputed_to_master(
    test_df: pd.DataFrame,
    master: pd.DataFrame,
    imputed: pd.DataFrame,
    decision_col: str = DECISION_COL,
) -> dict:
    """Compare only cells that were missing in the test set."""
    feature_cols = [c for c in master.columns if c != decision_col]
    comparisons = []
    for idx in test_df.index:
        for col in feature_cols:
            if not pd.isna(test_df.at[idx, col]):
                continue
            truth = master.at[idx, col]
            got = imputed.at[idx, col]
            ok = values_equal(got, truth)
            comparisons.append(
                {
                    "row": int(idx),
                    "column": col,
                    "master": truth,
                    "imputed": got,
                    "match": ok,
                }
            )

    n = len(comparisons)
    n_match = sum(1 for c in comparisons if c["match"])
    n_miss = n - n_match
    accuracy = (n_match / n) if n else 1.0

    by_col: dict[str, dict[str, int | float]] = {}
    for col in feature_cols:
        col_rows = [c for c in comparisons if c["column"] == col]
        cn = len(col_rows)
        cm = sum(1 for c in col_rows if c["match"])
        by_col[col] = {
            "compared": cn,
            "match": cm,
            "mismatch": cn - cm,
            "accuracy": (cm / cn) if cn else 1.0,
        }

    return {
        "n_compared": n,
        "n_match": n_match,
        "n_mismatch": n_miss,
        "accuracy": accuracy,
        "by_column": by_col,
        "comparisons": comparisons,
    }


def print_accuracy_report(
    results: dict,
    *,
    n_master: int,
    n_masked: int,
    feature_cols: list[str],
    timing: dict | None = None,
    detail_limit: int = 40,
) -> None:
    print("\n" + "=" * 72)
    print("DENGUE SYSTEM TEST — vectorized imputation accuracy + timing")
    print("=" * 72)
    print(f"Source file                     : {DENGUE_CSV.name}")
    print(f"Temporary master rows           : {n_master}")
    print(f"Feature columns                 : {feature_cols}")
    print(f"Decision column (never masked)  : {DECISION_COL}")
    print(f"Masked attribute cells          : {n_masked}")
    print(f"Cells compared                  : {results['n_compared']}")
    print(f"Exact matches                   : {results['n_match']}")
    print(f"Mismatches                      : {results['n_mismatch']}")
    print(f"Overall accuracy                : {results['accuracy']:.1%}")
    if timing:
        t = timing["n_incomplete"]
        m = timing["n_donors"]
        f = timing["n_features"]
        print("-" * 72)
        print(f"Backend                         : {timing.get('backend')}")
        print(f"Time complexity                 : {timing.get('complexity')}")
        print(f"  T (incomplete targets)        : {t}")
        print(f"  M (complete donors)           : {m}")
        print(f"  F (features)                  : {f}")
        print(f"  chunk_size                    : {timing.get('chunk_size')}")
        print(f"Imputation wall time            : {timing['seconds']:.4f} s")
        if timing.get("seconds_total") is not None:
            print(f"Total (incl. prep)              : {timing['seconds_total']:.4f} s")
        ops = t * m * f
        if timing["seconds"] > 0:
            print(f"Effective throughput            : {ops / timing['seconds']:.2e} (T*M*F)/s")
    print("-" * 72)
    print(f"{'column':<12}  {'compared':>8}  {'match':>6}  {'miss':>6}  accuracy")
    print("-" * 72)
    for col, stats in results["by_column"].items():
        print(
            f"{col:<12}  {stats['compared']:>8}  {stats['match']:>6}  "
            f"{stats['mismatch']:>6}  {stats['accuracy']:.1%}"
        )
    comps = results["comparisons"]
    if comps and detail_limit > 0:
        print("-" * 72)
        print(f"Sample cell comparisons (first {min(detail_limit, len(comps))}):")
        print(f"{'row':>4}  {'col':<12}  {'master':>8}  {'imputed':>8}  result")
        print("-" * 72)
        for c in comps[:detail_limit]:
            flag = "MATCH" if c["match"] else "MISS "
            print(
                f"{c['row']:>4}  {c['column']:<12}  {str(c['master']):>8}  "
                f"{str(c['imputed']):>8}  {flag}"
            )
        if len(comps) > detail_limit:
            print(f"... ({len(comps) - detail_limit} more comparisons omitted)")
    print("=" * 72 + "\n")


def run_dengue_accuracy_trial(
    *,
    sample_rows: int = SAMPLE_ROWS,
    mv_frac: float = MV_CELL_FRAC,
    seed: int = RNG_SEED,
    tmp_dir: Path | None = None,
    verbose: bool = False,
    detail_limit: int = 40,
) -> dict:
    """Full system-test trial with timing. Optionally writes temporary CSVs."""
    # Load a pool large enough to draw the requested master sample.
    raw = load_dengue(nrows=max(sample_rows * 3, sample_rows))
    master = build_temporary_master(raw, sample_rows=sample_rows, seed=seed)
    assert len(master) == sample_rows, (
        f"Need {sample_rows} complete rows, got {len(master)} "
        f"(increase dengue.csv pool / nrows)"
    )
    test_df, masked = inject_random_missing(
        master, mv_frac=mv_frac, seed=seed, decision_col=DECISION_COL
    )
    feature_cols = list(FEATURE_COLS)

    if tmp_dir is not None:
        tmp_dir = Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        master.to_csv(tmp_dir / "dengue_master_tmp.csv", index=False)
        test_df.to_csv(tmp_dir / "dengue_test_tmp.csv", index=False)

    imputed, timing = non_parametric_imputation(
        test_df,
        decision_col=DECISION_COL,
        feature_cols=feature_cols,
        verbose=verbose,
        return_timing=True,
    )

    assert not imputed[feature_cols].isna().any().any()
    pd.testing.assert_series_equal(
        imputed[DECISION_COL], master[DECISION_COL], check_names=False
    )

    results = compare_imputed_to_master(test_df, master, imputed, DECISION_COL)
    print_accuracy_report(
        results,
        n_master=len(master),
        n_masked=len(masked),
        feature_cols=feature_cols,
        timing=timing,
        detail_limit=detail_limit,
    )
    results.update(
        {
            "master": master,
            "test_df": test_df,
            "imputed": imputed,
            "masked_cells": masked,
            "feature_cols": feature_cols,
            "timing": timing,
            "sample_rows": sample_rows,
        }
    )
    return results


def print_scale_summary(runs: list[dict]) -> None:
    print("\n" + "=" * 88)
    print("DENGUE SCALE SUMMARY — accuracy + time complexity")
    print("=" * 88)
    print(
        f"{'run':>4}  {'rows':>7}  {'T':>6}  {'M':>6}  {'F':>2}  "
        f"{'time_s':>10}  {'accuracy':>9}  backend"
    )
    print("-" * 88)
    for i, r in enumerate(runs, start=1):
        tm = r["timing"]
        print(
            f"{i:>4}  {r['sample_rows']:>7}  {tm['n_incomplete']:>6}  "
            f"{tm['n_donors']:>6}  {tm['n_features']:>2}  "
            f"{tm['seconds']:>10.4f}  {r['accuracy']:>8.1%}  {tm['backend']}"
        )
    print("-" * 88)
    print("Complexity model: O(T * M * F) with NumPy-broadcast distances / z-scores,")
    print("chunked over incomplete targets (bounded memory). Dengue is never masked.")
    print("=" * 88 + "\n")


@pytest.mark.system
class TestDengueImputationAccuracy:
    def test_load_dengue_header_and_decision_column(self):
        df = load_dengue(nrows=5)
        assert list(df.columns) == [
            "Name",
            "Fever",
            "Headache",
            "JointPain",
            "Bleeding",
            "Dengue",
        ]
        assert df.columns[-1] == DECISION_COL

    def test_temporary_master_has_no_missing_values(self, tmp_path):
        raw = load_dengue(nrows=2_000)
        master = build_temporary_master(raw, sample_rows=1_000)
        assert not master.isna().any().any()
        assert list(master.columns) == [*FEATURE_COLS, DECISION_COL]
        assert ID_COL not in master.columns
        path = tmp_path / "dengue_master_tmp.csv"
        master.to_csv(path, index=False)
        reloaded = pd.read_csv(path)
        assert not reloaded.isna().any().any()
        assert len(reloaded) == len(master)

    def test_random_mask_never_touches_dengue(self):
        raw = load_dengue(nrows=2_000)
        master = build_temporary_master(raw, sample_rows=1_000)
        test_df, masked = inject_random_missing(master)
        assert test_df[DECISION_COL].notna().all()
        assert test_df[DECISION_COL].equals(master[DECISION_COL])
        assert len(masked) > 0
        assert all(col != DECISION_COL for _, col in masked)
        for col in FEATURE_COLS:
            keep = test_df[col].notna()
            pd.testing.assert_series_equal(
                test_df.loc[keep, col].astype(float),
                master.loc[keep, col].astype(float),
                check_names=False,
            )

    def test_imputation_accuracy_against_temporary_master(self, tmp_path):
        results = run_dengue_accuracy_trial(
            sample_rows=SAMPLE_ROWS, tmp_dir=tmp_path, detail_limit=20
        )
        assert (tmp_path / "dengue_master_tmp.csv").exists()
        assert (tmp_path / "dengue_test_tmp.csv").exists()
        assert results["n_compared"] > 0
        assert results["n_compared"] == results["n_match"] + results["n_mismatch"]
        assert 0.0 <= results["accuracy"] <= 1.0
        assert results["timing"]["seconds"] is not None
        assert results["timing"]["backend"] == "vectorized-categorical"
        pd.testing.assert_series_equal(
            results["imputed"][DECISION_COL],
            results["master"][DECISION_COL],
            check_names=False,
        )
        print(
            f"dengue system: {results['n_match']}/{results['n_compared']} "
            f"matched ({results['accuracy']:.1%}); "
            f"time={results['timing']['seconds']:.4f}s"
        )

    def test_scale_runs_10000_20000_50000(self):
        """Run 1: 10k, Run 2: 20k, Run 3: 50k — report accuracy + time."""
        runs = []
        for n_rows in SCALE_RUNS:
            results = run_dengue_accuracy_trial(
                sample_rows=n_rows,
                seed=RNG_SEED,
                detail_limit=0,
            )
            assert results["sample_rows"] == n_rows
            assert results["timing"]["backend"] == "vectorized-categorical"
            assert results["n_compared"] > 0
            runs.append(results)
        print_scale_summary(runs)
        assert [r["sample_rows"] for r in runs] == list(SCALE_RUNS)


if __name__ == "__main__":
    runs = []
    for n_rows in SCALE_RUNS:
        runs.append(
            run_dengue_accuracy_trial(sample_rows=n_rows, detail_limit=0)
        )
    print_scale_summary(runs)

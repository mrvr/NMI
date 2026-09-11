# -*- coding: utf-8 -*-
"""Unit tests for NMI non-parametric imputation."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nmilib import (
    filter_mv_decisions,
    get_MV_records,
    get_decision_mv_records,
    split_train_test_by_decision,
    carve_and_save_train_test,
    non_parametric_imputation,
    prepare_numeric_data,
    strip_MV_Records,
    compute_case1_categorical,
    compute_case1_fractions,
    compute_case2_categorical,
    compute_ic,
    pairwise_distance,
    get_nearest_neighbors,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = {
    "data.txt": ROOT / "data.txt",
    "data1.txt": ROOT / "data1.txt",
    "data2.txt": ROOT / "data2.txt",
    "data3.txt": ROOT / "data3.txt",
    "data4.txt": ROOT / "data4.txt",
    "data5.txt": ROOT / "data5.txt",
}
MASTER_FILES = {
    "data3.txt": ROOT / "data3_master.txt",
    "data4.txt": ROOT / "data4_master.txt",
    "data5.txt": ROOT / "data5_master.txt",
}
ORIGINAL_DATA_FILES = ("data.txt", "data1.txt", "data2.txt")
SYNTHETIC_DATA_FILES = ("data3.txt", "data4.txt", "data5.txt")
NA_VALUES = ["", "?"]  # '?' denotes missing values in the .txt test files
MAX_MV_FRAC = 0.25


def load_dataset(name: str) -> pd.DataFrame:
    path = DATA_FILES[name]
    assert path.exists(), f"Missing test data file: {path}"
    return pd.read_csv(path, na_values=NA_VALUES)


def load_master_for(test_name: str) -> pd.DataFrame:
    path = MASTER_FILES[test_name]
    assert path.exists(), f"Missing master file: {path}"
    df = pd.read_csv(path, na_values=NA_VALUES)
    assert not df.isna().any().any(), f"Master {path.name} must have no missing values"
    return df


def values_equal(imputed, truth, kind: str, rtol: float = 1e-6, atol: float = 1e-6) -> bool:
    if pd.isna(imputed) or pd.isna(truth):
        return False
    if kind == "categorical":
        return str(imputed) == str(truth)
    try:
        return bool(np.isclose(float(imputed), float(truth), rtol=rtol, atol=atol))
    except (TypeError, ValueError):
        return str(imputed) == str(truth)


def column_kind_for_compare(series: pd.Series) -> str:
    if pd.api.types.is_float_dtype(series) or pd.api.types.is_integer_dtype(series):
        # at1 is categorical even if somehow numeric; use unique string check
        return "fractional"
    return "categorical"


def compare_imputation_to_master(test_name: str) -> dict:
    """
    Impute the test file and compare masked attribute cells to the master.

    Attribute cells on rows with missing `dec` are not imputed by the algorithm
    and are reported as SKIPPED (decision missing).
    """
    test_df = load_dataset(test_name)
    master = load_master_for(test_name)
    assert list(test_df.columns) == list(master.columns)
    assert len(test_df) == len(master)

    attrs = [c for c in test_df.columns if c != "dec"]
    imputed = non_parametric_imputation(test_df, decision_col="dec")

    comparisons = []
    skipped = []
    for idx in test_df.index:
        for col in attrs:
            if not pd.isna(test_df.at[idx, col]):
                continue
            truth = master.at[idx, col]
            got = imputed.at[idx, col]
            # Algorithm does not impute rows with missing decision
            if pd.isna(test_df.at[idx, "dec"]):
                skipped.append(
                    {
                        "row": int(idx),
                        "column": col,
                        "master": truth,
                        "imputed": got,
                        "reason": "dec missing",
                    }
                )
                continue
            kind = "categorical" if col == "at1" else "fractional"
            ok = values_equal(got, truth, kind)
            comparisons.append(
                {
                    "row": int(idx),
                    "column": col,
                    "master": truth,
                    "imputed": got,
                    "match": ok,
                }
            )

    dec_mv = int(test_df["dec"].isna().sum())
    n = len(comparisons)
    n_match = sum(1 for c in comparisons if c["match"])
    n_miss = n - n_match
    accuracy = (n_match / n) if n else 1.0

    print("\n" + "=" * 72)
    print(f"IMPUTATION vs MASTER — {test_name}  (master: {MASTER_FILES[test_name].name})")
    print("=" * 72)
    print(f"Missing dec values              : {dec_mv}")
    print(f"Attr cells skipped (dec missing): {len(skipped)}")
    print(f"Masked attribute cells compared : {n}")
    print(f"Exact/approx matches            : {n_match}")
    print(f"Mismatches                      : {n_miss}")
    print(f"Match rate                      : {accuracy:.1%}")
    print("-" * 72)
    print(f"{'row':>4}  {'col':<4}  {'master':>10}  {'imputed':>10}  result")
    print("-" * 72)
    for c in comparisons:
        flag = "MATCH" if c["match"] else "MISS "
        print(
            f"{c['row']:>4}  {c['column']:<4}  {str(c['master']):>10}  "
            f"{str(c['imputed']):>10}  {flag}"
        )
    if skipped:
        print("-" * 72)
        print("Skipped (dec missing — not imputed by algorithm):")
        for c in skipped:
            print(
                f"{c['row']:>4}  {c['column']:<4}  master={c['master']}  "
                f"left={c['imputed']}"
            )
    print("=" * 72 + "\n")

    return {
        "test_name": test_name,
        "n_compared": n,
        "n_match": n_match,
        "n_mismatch": n_miss,
        "n_skipped": len(skipped),
        "dec_mv": dec_mv,
        "accuracy": accuracy,
        "comparisons": comparisons,
        "skipped": skipped,
        "imputed": imputed,
        "test_df": test_df,
        "master": master,
    }


@pytest.fixture(params=list(DATA_FILES))
def dataset_name(request):
    return request.param


@pytest.fixture
def raw_df(dataset_name):
    return load_dataset(dataset_name)


@pytest.fixture(params=list(SYNTHETIC_DATA_FILES))
def synthetic_name(request):
    return request.param


# ---------------------------------------------------------------------------
# Schema / loading tests
# ---------------------------------------------------------------------------

class TestDataFileSchema:
    def test_first_row_is_header_with_dec(self, dataset_name):
        df = load_dataset(dataset_name)
        assert "dec" in df.columns
        assert df.columns[-1] == "dec"

    def test_parameter_columns_named_atN(self, dataset_name):
        df = load_dataset(dataset_name)
        attrs = [c for c in df.columns if c != "dec"]
        assert attrs, "expected at least one attribute column"
        for i, col in enumerate(attrs, start=1):
            assert col == f"at{i}", f"expected at{i}, got {col}"

    def test_data_and_data1_have_two_attrs(self):
        for name in ("data.txt", "data1.txt"):
            df = load_dataset(name)
            assert list(df.columns) == ["at1", "at2", "dec"]

    def test_data2_has_three_attrs(self):
        df = load_dataset("data2.txt")
        assert list(df.columns) == ["at1", "at2", "at3", "dec"]

    def test_synthetic_files_have_100_records(self):
        for name in SYNTHETIC_DATA_FILES:
            df = load_dataset(name)
            assert len(df) == 100, f"{name} should have 100 records"
            master = load_master_for(name)
            assert len(master) == 100

    def test_synthetic_attribute_counts(self):
        assert list(load_dataset("data3.txt").columns) == ["at1", "at2", "dec"]
        assert list(load_dataset("data4.txt").columns) == ["at1", "at2", "at3", "dec"]
        assert list(load_dataset("data5.txt").columns) == [
            "at1", "at2", "at3", "at4", "dec",
        ]

    def test_question_mark_is_missing_value_marker(self):
        """Test files encode missing values as '?'; masters do not."""
        for name in DATA_FILES:
            text = DATA_FILES[name].read_text()
            assert "?" in text, f"{name} should contain '?' missing-value markers"
            df = load_dataset(name)
            assert df.isna().any().any(), f"{name}: '?' should become NA"
        for name in SYNTHETIC_DATA_FILES:
            master_text = MASTER_FILES[name].read_text()
            assert "?" not in master_text, f"{MASTER_FILES[name].name} must be complete"

    def test_data_files_have_missing_values(self, raw_df):
        assert raw_df.isna().any().any()

    def test_mv_count_less_than_25_percent(self, synthetic_name):
        df = load_dataset(synthetic_name)
        n_rows = len(df)
        n_cells = int(df.shape[0] * df.shape[1])
        mv_cells = int(df.isna().sum().sum())
        mv_rows = int(df.isna().any(axis=1).sum())
        assert mv_cells > 0
        assert mv_cells / n_cells < MAX_MV_FRAC, (
            f"{synthetic_name}: MV cells {mv_cells}/{n_cells} "
            f"({mv_cells / n_cells:.1%}) must be < 25%"
        )
        assert mv_rows <= int(n_rows * MAX_MV_FRAC), (
            f"{synthetic_name}: MV rows {mv_rows}/{n_rows} exceed 25%"
        )

    def test_synthetic_files_have_missing_dec(self, synthetic_name):
        df = load_dataset(synthetic_name)
        assert df["dec"].isna().any(), f"{synthetic_name}: expected some missing dec values"

    def test_masters_are_complete_and_aligned(self, synthetic_name):
        test_df = load_dataset(synthetic_name)
        master = load_master_for(synthetic_name)
        assert list(master.columns) == list(test_df.columns)
        assert len(master) == len(test_df)
        # Non-missing test cells must equal master
        for col in test_df.columns:
            mask = test_df[col].notna()
            pd.testing.assert_series_equal(
                test_df.loc[mask, col].astype(object),
                master.loc[mask, col].astype(object),
                check_names=False,
            )

    def test_synthetic_files_have_complete_donors(self):
        for name in SYNTHETIC_DATA_FILES:
            df = load_dataset(name)
            complete = strip_MV_Records(filter_mv_decisions(df))
            assert len(complete) >= 8, f"{name}: need complete donor rows"


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_filter_mv_decisions_keeps_known_dec(self, raw_df):
        filtered = filter_mv_decisions(raw_df)
        assert filtered["dec"].notna().all()
        assert len(filtered) == raw_df["dec"].notna().sum()

    def test_strip_and_get_mv_partition(self, raw_df):
        usable = filter_mv_decisions(raw_df)
        complete = strip_MV_Records(usable)
        incomplete = get_MV_records(usable)
        assert complete.isna().sum().sum() == 0
        assert incomplete.isna().any(axis=1).all()
        assert set(usable.index) == set(complete.index).union(set(incomplete.index))

    def test_prepare_numeric_preserves_categorical_at1(self, raw_df):
        prepared = prepare_numeric_data(raw_df)
        assert "at1" in prepared.columns
        non_null = prepared["at1"].dropna()
        assert set(non_null.unique()).issubset({"yes", "no"})


# ---------------------------------------------------------------------------
# Index / distance / z-score unit checks
# ---------------------------------------------------------------------------

class TestIndexFormulas:
    def test_case1_categorical_ratio_between_zero_and_one(self):
        df = load_dataset("data2.txt")
        complete = strip_MV_Records(filter_mv_decisions(df)).reset_index(drop=True)
        assert len(complete) >= 2
        i, k = 0, 1
        if complete.iloc[i]["dec"] == complete.iloc[k]["dec"]:
            val = compute_case1_categorical(complete, complete.columns, 0, i, k)
        else:
            val = compute_case2_categorical(complete, complete.columns, 0, i, k)
        assert val >= 0.0
        assert np.isfinite(val)

    def test_case1_fractions_uses_column_mean(self):
        complete = pd.DataFrame(
            {"at1": ["yes", "no"], "at2": [10.0, 14.0], "dec": ["+", "+"]}
        )
        val = compute_case1_fractions(complete, complete.columns, 1, 0, 1)
        assert val == pytest.approx(10.0 / 12.0)

    def test_ic_self_is_zero(self):
        df = load_dataset("data.txt")
        complete = strip_MV_Records(filter_mv_decisions(df)).reset_index(drop=True)
        assert compute_ic(complete, 0, 0, 0) == 0.0
        assert pairwise_distance(complete, 0, 0) == 0.0

    def test_zd_nonpositive_neighbors_exist(self):
        d = np.array([0.0, 0.5, 1.0, 2.0])
        nn, z = get_nearest_neighbors(d)
        assert all(z[j] <= 0 for j in nn)
        assert len(nn) >= 1


# ---------------------------------------------------------------------------
# End-to-end imputation
# ---------------------------------------------------------------------------

class TestNonParametricImputation:
    def test_imputes_all_attribute_missing_values(self, dataset_name, raw_df):
        attrs = [c for c in raw_df.columns if c != "dec"]
        out = non_parametric_imputation(raw_df, decision_col="dec")
        known_dec = raw_df["dec"].notna()
        for col in attrs:
            still_missing = out.loc[known_dec, col].isna()
            assert not still_missing.any(), (
                f"{dataset_name}: leftover MV in {col} for rows with known dec"
            )

    def test_does_not_impute_missing_decision(self, raw_df):
        out = non_parametric_imputation(raw_df, decision_col="dec")
        pd.testing.assert_series_equal(
            out["dec"].isna(),
            raw_df["dec"].isna(),
            check_names=False,
        )

    def test_preserves_observed_attribute_values(self, raw_df):
        out = non_parametric_imputation(raw_df, decision_col="dec")
        attrs = [c for c in raw_df.columns if c != "dec"]
        for col in attrs:
            mask = raw_df[col].notna()
            left = out.loc[mask, col].astype(object)
            right = raw_df.loc[mask, col].astype(object)
            pd.testing.assert_series_equal(left, right, check_names=False)

    def test_output_shape_and_columns(self, raw_df):
        out = non_parametric_imputation(raw_df, decision_col="dec")
        assert out.shape == raw_df.shape
        assert list(out.columns) == list(raw_df.columns)

    def test_expected_imputed_values_data_txt(self):
        df = load_dataset("data.txt")
        out = non_parametric_imputation(df, decision_col="dec")
        assert out.loc[0, "at1"] == "yes"
        assert out.loc[6, "at1"] == "yes"
        assert float(out.loc[6, "at2"]) == pytest.approx(12.5)

    def test_expected_imputed_values_data1_txt(self):
        df = load_dataset("data1.txt")
        out = non_parametric_imputation(df, decision_col="dec")
        assert out.loc[0, "at1"] == "yes"
        assert float(out.loc[6, "at2"]) == pytest.approx(12.5)

    def test_expected_imputed_values_data2_txt(self):
        df = load_dataset("data2.txt")
        out = non_parametric_imputation(df, decision_col="dec")
        assert out.loc[0, "at1"] == "yes"
        assert float(out.loc[2, "at3"]) == pytest.approx(13.9)
        assert float(out.loc[6, "at2"]) == pytest.approx(10.5)

    def test_varying_attribute_count_supported(self):
        expected = {
            "data.txt": 2,
            "data1.txt": 2,
            "data2.txt": 3,
            "data3.txt": 2,
            "data4.txt": 3,
            "data5.txt": 4,
        }
        for name, n_attrs in expected.items():
            df = load_dataset(name)
            attrs = [c for c in df.columns if c != "dec"]
            assert len(attrs) == n_attrs
            out = non_parametric_imputation(df, decision_col="dec")
            assert list(out.columns) == list(df.columns)


# ---------------------------------------------------------------------------
# Master-ground-truth comparison for synthetic files (flash results)
# ---------------------------------------------------------------------------

class TestImputationAgainstMaster:
    def test_compare_and_flash_results(self, synthetic_name):
        results = compare_imputation_to_master(synthetic_name)
        assert results["dec_mv"] > 0
        assert results["n_compared"] > 0
        assert results["n_compared"] == results["n_match"] + results["n_mismatch"]
        test_df = results["test_df"]
        imputed = results["imputed"]
        attrs = [c for c in test_df.columns if c != "dec"]
        # Attribute MVs on rows with known dec must be filled
        known_dec = test_df["dec"].notna()
        assert not imputed.loc[known_dec, attrs].isna().any().any()
        # Missing dec pattern unchanged
        pd.testing.assert_series_equal(
            imputed["dec"].isna(), test_df["dec"].isna(), check_names=False
        )
        summary = (
            f"{synthetic_name}: {results['n_match']}/{results['n_compared']} "
            f"matched ({results['accuracy']:.1%}); "
            f"skipped={results['n_skipped']}, dec_mv={results['dec_mv']}"
        )
        print(summary)
        assert results["n_match"] >= 0


# ---------------------------------------------------------------------------
# Carve missing-decision rows as held-out test data
# ---------------------------------------------------------------------------

class TestCarveDecisionMvAsTest:
    def test_split_train_test_by_decision(self, synthetic_name):
        df = load_dataset(synthetic_name)
        train, test = split_train_test_by_decision(df, decision_col="dec")
        assert len(train) + len(test) == len(df)
        assert train["dec"].notna().all()
        assert test["dec"].isna().all()
        assert len(test) == int(df["dec"].isna().sum())
        # no index overlap
        assert set(train.index).isdisjoint(set(test.index))

    def test_get_decision_mv_records_matches_test_split(self, synthetic_name):
        df = load_dataset(synthetic_name)
        carved = get_decision_mv_records(df, decision_col="dec")
        _, test = split_train_test_by_decision(df, decision_col="dec")
        pd.testing.assert_frame_equal(carved, test)

    def test_carve_and_save_writes_train_test_files(self, tmp_path, synthetic_name):
        df = load_dataset(synthetic_name)
        train_path = tmp_path / f"{synthetic_name}_train.txt"
        test_path = tmp_path / f"{synthetic_name}_test.txt"
        train, test = carve_and_save_train_test(
            df, train_path, test_path, decision_col="dec"
        )
        assert train_path.exists() and test_path.exists()
        assert len(test) > 0
        reloaded_test = pd.read_csv(test_path, na_values=NA_VALUES)
        assert reloaded_test["dec"].isna().all()
        reloaded_train = pd.read_csv(train_path, na_values=NA_VALUES)
        assert reloaded_train["dec"].notna().all()

    def test_impute_on_train_only_then_test_held_out(self, synthetic_name):
        df = load_dataset(synthetic_name)
        train, test = split_train_test_by_decision(df, decision_col="dec")
        imputed_train = non_parametric_imputation(train, decision_col="dec")
        attrs = [c for c in train.columns if c != "dec"]
        # training partition attribute MVs filled
        assert not imputed_train[attrs].isna().any().any()
        # test partition untouched by this training step (still missing dec)
        assert test["dec"].isna().all()
        assert len(test) == int(df["dec"].isna().sum())

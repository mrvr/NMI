# -*- coding: utf-8 -*-
"""
Cross-check the IEEE paper illustrative example against nmilib.

Example S (from paper):
  R1 = (?, 12.0, positive)   ← impute at1
  R2 = (yes, 10.5, positive)
  R3 = (no, 14.0, positive)
  R4 = (no, 13.0, negative)

Paper reports:
  A# ≈ 12.17
  distances of R1 to {R2,R3,R4} ≈ {0.86, 0.99, 1.14}
  z-scores ≈ {-0.57, -0.57, 1.154}
  nearest (z ≤ 0): R2, R3
  imputed value for missing at1: yes
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nmilib import (
    non_parametric_imputation,
    pairwise_distance,
    compute_zd,
    get_nearest_neighbors,
)


def paper_example_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "at1": [np.nan, "yes", "no", "no"],
            "at2": [12.0, 10.5, 14.0, 13.0],
            "dec": ["positive", "positive", "positive", "negative"],
        }
    )


PAPER_DISTANCES = np.array([0.86, 0.99, 1.14])
PAPER_Z = np.array([-0.57, -0.57, 1.154])
PAPER_A_HASH = 12.17
PAPER_IMPUTED = "yes"
PAPER_NEAREST = {0, 1}  # R2, R3 among donors R2,R3,R4


class TestPaperIllustrativeExample:
    def test_final_imputation_matches_paper(self):
        S = paper_example_frame()
        out = non_parametric_imputation(S, decision_col="dec")
        assert out.loc[0, "at1"] == PAPER_IMPUTED
        # observed cells unchanged
        assert out.loc[0, "at2"] == 12.0
        assert out.loc[0, "dec"] == "positive"

    def test_distances_close_to_paper(self):
        S = paper_example_frame()
        ours = np.array([pairwise_distance(S, 0, k) for k in (1, 2, 3)])
        # R2/R3 should be within ~5%; R4 may differ more (Case II / A# definition)
        assert ours[0] == pytest.approx(PAPER_DISTANCES[0], rel=0.05, abs=0.05)
        assert ours[1] == pytest.approx(PAPER_DISTANCES[1], rel=0.05, abs=0.05)

    def test_report_cross_check(self, capsys):
        """Flash a comparison table of paper vs implementation."""
        S = paper_example_frame()
        out = non_parametric_imputation(S, decision_col="dec")
        dists = np.array([pairwise_distance(S, 0, k) for k in (1, 2, 3)])
        z = compute_zd(dists)
        nn, _ = get_nearest_neighbors(dists)
        a_hash_overall = float(S["at2"].mean())
        a_hash_pos = float(S.loc[S["dec"] == "positive", "at2"].mean())

        print("\n" + "=" * 72)
        print("PAPER ILLUSTRATIVE EXAMPLE — cross-check")
        print("=" * 72)
        print(S.to_string())
        print("-" * 72)
        print(f"{'metric':<28} {'paper':>12} {'ours':>12} {'status':>10}")
        print("-" * 72)

        rows = [
            ("A# (overall mean)", PAPER_A_HASH, a_hash_overall, abs(a_hash_overall - PAPER_A_HASH) < 0.3),
            ("A# (positive-class mean)", PAPER_A_HASH, a_hash_pos, abs(a_hash_pos - PAPER_A_HASH) < 0.05),
            ("d(R1,R2)", PAPER_DISTANCES[0], dists[0], abs(dists[0] - PAPER_DISTANCES[0]) < 0.05),
            ("d(R1,R3)", PAPER_DISTANCES[1], dists[1], abs(dists[1] - PAPER_DISTANCES[1]) < 0.05),
            ("d(R1,R4)", PAPER_DISTANCES[2], dists[2], abs(dists[2] - PAPER_DISTANCES[2]) < 0.08),
            ("z(R2)", PAPER_Z[0], z[0], abs(z[0] - PAPER_Z[0]) < 0.15),
            ("z(R3)", PAPER_Z[1], z[1], abs(z[1] - PAPER_Z[1]) < 0.15),
            ("z(R4)", PAPER_Z[2], z[2], abs(z[2] - PAPER_Z[2]) < 0.25),
            ("NN includes R2", True, 0 in nn, (0 in nn) is True),
            ("NN includes R3", True, 1 in nn, (1 in nn) is True),
            ("NN excludes R4", True, 2 not in nn, (2 not in nn) is True),
            ("imputed at1", PAPER_IMPUTED, out.loc[0, "at1"], out.loc[0, "at1"] == PAPER_IMPUTED),
        ]
        n_ok = 0
        for name, paper, ours, ok in rows:
            status = "MATCH" if ok else "DIFF"
            if ok:
                n_ok += 1
            paper_s = f"{paper:.4f}" if isinstance(paper, (float, np.floating)) else str(paper)
            ours_s = f"{ours:.4f}" if isinstance(ours, (float, np.floating, int)) and not isinstance(ours, bool) else str(ours)
            print(f"{name:<28} {paper_s:>12} {ours_s:>12} {status:>10}")

        # prettier numeric formatting
        print("-" * 72)
        print("Distances detail:")
        for i, lab in enumerate(["R2", "R3", "R4"]):
            rel = abs(dists[i] - PAPER_DISTANCES[i]) / PAPER_DISTANCES[i] * 100
            print(
                f"  {lab}: ours={dists[i]:.4f}  paper={PAPER_DISTANCES[i]:.2f}  "
                f"rel_err={rel:.2f}%  z_ours={z[i]:.4f}  z_paper={PAPER_Z[i]}"
            )
        print("-" * 72)
        print(f"Final imputation: ours={out.loc[0,'at1']!r}  paper={PAPER_IMPUTED!r}")
        print(f"Checks matched: {n_ok}/{len(rows)}")
        print("=" * 72 + "\n")

        # Critical accuracy: final answer must match
        assert out.loc[0, "at1"] == PAPER_IMPUTED

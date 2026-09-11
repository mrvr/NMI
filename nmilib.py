# -*- coding: utf-8 -*-
"""
Source code for New Imputation Method (non-parametric indexing measure).

Implements the proximity index I_Cl(Ri, Rk), distance d_ik, z-score
nearest-neighbour selection, and mode/mean imputation from the attached method.
"""
import time

import pandas as pd
import numpy as np
from pandas import Categorical

# Known categorical / binary decision-attribute values
CATEGORICAL_TYPES = Categorical(
    [True, False, '+', '-', 'yes', 'YES', 'Yes', 'no', 'NO', 'No']
)
CATEGORICAL_VALUES = set(CATEGORICAL_TYPES)


def _safe_ratio_min(a, b):
    """min(a/b, b/a) with zero-safe handling."""
    a = float(a)
    b = float(b)
    if a == 0.0 and b == 0.0:
        return 0.0
    if a == 0.0 or b == 0.0:
        return 0.0
    return min(a / b, b / a)


def _safe_ratio_max(a, b):
    """max(a/b, b/a) with zero-safe handling."""
    a = float(a)
    b = float(b)
    if a == 0.0 and b == 0.0:
        return 0.0
    if a == 0.0 or b == 0.0:
        return 0.0
    return max(a / b, b / a)


def _column_kind(series):
    """
    Classify an attribute column.
    Returns 'categorical' (nominal or integer) or 'fractional' (real/float).
    """
    if pd.api.types.is_float_dtype(series):
        # Treat binary 0/1 floats still as categorical if all values are categorical-like
        vals = series.dropna().unique()
        if len(vals) and set(vals).issubset({0.0, 1.0, True, False}):
            return 'categorical'
        return 'fractional'
    if not pd.api.types.is_numeric_dtype(series):
        return 'categorical'
    if series.dropna().isin(list(CATEGORICAL_VALUES)).any():
        return 'categorical'
    # integer / boolean numeric -> Case item (i)
    return 'categorical'


def _column_is_categorical(M, col_label):
    """Nominal/categorical/integer attributes use counting-based indices."""
    return _column_kind(M[col_label]) == 'categorical'


def filter_mv_decisions(data):
    """Keep rows whose decision attribute (last column) is present."""
    data1 = pd.DataFrame(data)
    return data1[data1.iloc[:, -1].notnull()]


def get_decision_mv_records(data, decision_col=None):
    """
    Carve out rows whose decision attribute is missing.

    These rows are held out as **test data** after training / imputation
    on rows with a known decision.
    """
    df = pd.DataFrame(data)
    if decision_col is None:
        decision_col = df.columns[-1]
    return df[df[decision_col].isnull()].copy()


def split_train_test_by_decision(data, decision_col=None):
    """
    Split S into training and test sets by decision availability.

    - train: rows with known decision (used for imputation / model fitting)
    - test:  rows with missing decision (held out for evaluation after training)

    Returns
    -------
    train : DataFrame
    test : DataFrame
    """
    df = pd.DataFrame(data).copy()
    if decision_col is None:
        decision_col = df.columns[-1]
    known = df[decision_col].notnull()
    train = df.loc[known].copy()
    test = df.loc[~known].copy()
    return train, test


def dataframe_to_qmark_csv(df, path):
    """Write a DataFrame using '?' for missing values (NMI .txt convention)."""
    out = pd.DataFrame(df).copy()
    out = out.where(out.notna(), other="?")
    out.to_csv(path, index=False)


def carve_and_save_train_test(
    data,
    train_path,
    test_path,
    decision_col=None,
):
    """
    Carve rows with missing decision as test data, remaining as train,
    and save both using '?' for missing cells.
    """
    train, test = split_train_test_by_decision(data, decision_col=decision_col)
    dataframe_to_qmark_csv(train, train_path)
    dataframe_to_qmark_csv(test, test_path)
    return train, test


def strip_MV_Records(Samples):
    """Return only completely observed rows."""
    return pd.DataFrame(Samples).dropna()


def get_MV_records(Samples):
    """Return rows that have at least one missing value."""
    temp = pd.DataFrame(Samples)
    return temp[temp.isnull().any(axis=1)]


def prepare_numeric_data(datatable):
    """Coerce columns to numeric where possible."""
    datatable = pd.DataFrame(datatable).copy()
    for col in datatable.columns:
        try:
            datatable[col] = pd.to_numeric(datatable[col])
        except (ValueError, TypeError):
            pass
    return datatable


def compute_case1_categorical(A, column_labels, mv_col, i, k, non_mv_row_index=None):
    """
    Case I(i): same decision class, categorical/nominal/integer.
    I = min(γ_pi / γ_qk, γ_qk / γ_pi) for i != k, else 0.
    """
    A = pd.DataFrame(A)
    C = column_labels
    if i == k:
        return 0.0

    dec_col = C[-1]
    col_name = C[mv_col]
    class_of_i = A.iloc[i][dec_col]

    A_same = strip_MV_Records(A)
    A_same = A_same[A_same[dec_col] == class_of_i]
    if A_same.empty:
        return 0.0

    val_i = A.iloc[i][col_name]
    val_k = A.iloc[k][col_name]
    gamma_pi = int((A_same[col_name] == val_i).sum())
    gamma_qk = int((A_same[col_name] == val_k).sum())
    return _safe_ratio_min(gamma_pi, gamma_qk)


def compute_case1_fractions(A, ci, col, i, k, non_mv_row_index=None):
    """
    Case I(ii): same decision class, fractional/real attribute.
    I = min(A_il / A#, A_kl / A#) for i != k, else 0.

    A# = mean of the *l-th* column entries (column Cl), excluding missing
    values in that column. (Not the decision column; not a fixed "first" column.)
    """
    A = pd.DataFrame(A)
    C = ci
    if i == k:
        return 0.0

    col_name = C[col]
    series = A[col_name].dropna().astype(float)
    if series.empty:
        return 0.0
    a_hash = float(series.mean())
    if a_hash == 0.0:
        return 0.0

    val_i = float(A.iloc[i][col_name])
    val_k = float(A.iloc[k][col_name])
    return min(val_i / a_hash, val_k / a_hash)


def compute_case2_categorical(A, ci, col, i, k, non_mv_row_index=None):
    """
    Case II(i): different decision classes, categorical/nominal/integer.
    I = max(β_r / δ_s, δ_s / β_r) for i != k, else 0.
    """
    A = pd.DataFrame(A)
    C = ci
    if i == k:
        return 0.0

    dec_col = C[-1]
    col_name = C[col]
    dec_i = A.iloc[i][dec_col]
    dec_k = A.iloc[k][dec_col]

    A_no_mv = strip_MV_Records(A)
    P = A_no_mv[A_no_mv[dec_col] == dec_i]
    Q = A_no_mv[A_no_mv[dec_col] == dec_k]

    val_i = A.iloc[i][col_name]
    val_k = A.iloc[k][col_name]
    beta_r = int((P[col_name] == val_i).sum())
    delta_s = int((Q[col_name] == val_k).sum())
    return _safe_ratio_max(beta_r, delta_s)


def compute_case2_fractions(A, ci, col, i, k, non_mv_row_index=None):
    """
    Case II(ii): different decision classes, fractional/real attribute.
    I = max(A_il / Λ, A_kl / Λ) for i != k, else 0,
    where Λ = min(P#, Q#).

    P# and Q# are the means of the *l-th* column (Cl) over decision classes
    of Ri and Rk respectively, excluding rows with MV in column l.

    Note: some paper text says "average of the first column entries"; that is
    treated as a wording error. Numerators use A_il, A_kl (column l), and MV
    exclusion is also defined on the l-th column, so the averages must be over
    column l — not column 1.
    """
    A = pd.DataFrame(A)
    C = ci
    if i == k:
        return 0.0

    dec_col = C[-1]
    col_name = C[col]  # l-th attribute column Cl (not the first column)
    dec_i = A.iloc[i][dec_col]
    dec_k = A.iloc[k][dec_col]

    # Means of column l within each decision class, excluding MV in column l
    P_vals = A.loc[A[dec_col] == dec_i, col_name].dropna().astype(float)
    Q_vals = A.loc[A[dec_col] == dec_k, col_name].dropna().astype(float)
    if P_vals.empty or Q_vals.empty:
        return 0.0

    p_hash = float(P_vals.mean())
    q_hash = float(Q_vals.mean())
    lam = min(p_hash, q_hash)
    if lam == 0.0:
        return 0.0

    val_i = float(A.iloc[i][col_name])
    val_k = float(A.iloc[k][col_name])
    return max(val_i / lam, val_k / lam)


def compute_ic(A, col_idx, i, k):
    """Compute I_Cl(Ri, Rk) for one attribute column."""
    A = pd.DataFrame(A)
    C = A.columns
    if i == k:
        return 0.0

    dec_col = C[-1]
    col_name = C[col_idx]
    # Skip if either side missing in this attribute
    if pd.isnull(A.iloc[i][col_name]) or pd.isnull(A.iloc[k][col_name]):
        return 0.0
    if pd.isnull(A.iloc[i][dec_col]) or pd.isnull(A.iloc[k][dec_col]):
        return 0.0

    same_class = A.iloc[i][dec_col] == A.iloc[k][dec_col]
    kind = _column_kind(A[col_name])

    if same_class:
        if kind == 'categorical':
            return compute_case1_categorical(A, C, col_idx, i, k)
        return compute_case1_fractions(A, C, col_idx, i, k)
    if kind == 'categorical':
        return compute_case2_categorical(A, C, col_idx, i, k)
    return compute_case2_fractions(A, C, col_idx, i, k)


def pairwise_distance(A, i, k, feature_cols=None):
    """
    d_ik = sqrt(sum_l I_Cl(Ri, Rk)^2) over attribute columns 1..n-1
    (or a provided feature column list). Only columns observed in both rows.
    """
    A = pd.DataFrame(A)
    C = list(A.columns)
    if feature_cols is None:
        feature_cols = C[:-1]

    total = 0.0
    for col_name in feature_cols:
        col_idx = C.index(col_name)
        ic = compute_ic(A, col_idx, i, k)
        total += ic * ic
    return float(np.sqrt(total))


def compute_distance_matrix(A, row_positions=None, feature_cols=None):
    """
    Build distance matrix D among the given row positions (iloc indices).
    Diagonal is 0.
    """
    A = pd.DataFrame(A)
    if row_positions is None:
        row_positions = list(range(len(A)))
    m = len(row_positions)
    D = np.zeros((m, m), dtype=float)
    for a, i in enumerate(row_positions):
        for b, k in enumerate(row_positions):
            if a == b:
                D[a, b] = 0.0
            elif b < a:
                D[a, b] = D[b, a]
            else:
                D[a, b] = pairwise_distance(A, i, k, feature_cols=feature_cols)
    return D


def compute_zd(distances):
    """
    z(d_j) = (d_j - mean(d)) / sqrt(1/(m-1) * sum (d_j - mean)^2)
    Returns z-scores for one row's distance vector.
    """
    d = np.asarray(distances, dtype=float)
    if d.ndim == 1:
        return compute_zd_matrix(d.reshape(1, -1))[0]
    return compute_zd_matrix(d)


def compute_zd_matrix(distances):
    """
    Vectorized z-scores for a batch of distance rows.

    Parameters
    ----------
    distances : array (t, m)
        Distances from t targets to m donors.

    Returns
    -------
    array (t, m)
    """
    d = np.asarray(distances, dtype=float)
    if d.ndim != 2:
        raise ValueError("compute_zd_matrix expects a 2-D distance array")
    t, m = d.shape
    if m < 2:
        return np.zeros_like(d)
    mean_d = d.mean(axis=1, keepdims=True)
    var = np.sum((d - mean_d) ** 2, axis=1, keepdims=True) / (m - 1)
    z = np.zeros_like(d)
    positive = var[:, 0] > 0.0
    if np.any(positive):
        z[positive] = (d[positive] - mean_d[positive]) / np.sqrt(var[positive])
    return z


def nearest_neighbor_mask(distances):
    """
    Boolean NN mask (t, m): donors with z(d) <= 0.

    Rows with no non-positive z fall back to the single closest donor.
    """
    d = np.asarray(distances, dtype=float)
    if d.ndim == 1:
        d = d.reshape(1, -1)
    z = compute_zd_matrix(d)
    nn = z <= 0.0
    empty = ~nn.any(axis=1)
    if np.any(empty):
        rows = np.flatnonzero(empty)
        cols = np.argmin(d[empty], axis=1)
        nn[rows, cols] = True
    return nn


def get_nearest_neighbors(distances, exclude_self=True):
    """Records with z(d) <= 0 are nearest neighbours."""
    d = np.asarray(distances, dtype=float)
    z = compute_zd(d)
    nn_mask = nearest_neighbor_mask(d.reshape(1, -1))[0]
    nn = list(np.flatnonzero(nn_mask))
    if exclude_self and 0 in nn and len(distances) > 0:
        # self is not identified by index 0 here; caller passes distances
        # from target to donors only, so no self to exclude.
        pass
    if not nn:
        nn = [int(np.argmin(d))]
    return nn, z


def print_distance_scores(target_label, distances, z_scores, nn_indices, donor_labels=None):
    """Print distance / z-score table for reference during imputation."""
    distances = np.asarray(distances, dtype=float)
    z_scores = np.asarray(z_scores, dtype=float)
    nn_set = set(nn_indices)
    print(f"\n--- Distance scores for target {target_label} ---")
    print(f"{'donor':>10}  {'d_ik':>12}  {'z(d)':>10}  {'NN (z<=0)':>10}")
    for j, (d, z) in enumerate(zip(distances, z_scores)):
        label = donor_labels[j] if donor_labels is not None else j
        flag = "yes" if j in nn_set else ""
        print(f"{label!s:>10}  {d:12.6f}  {z:10.4f}  {flag:>10}")
    print(f"Nearest neighbours: {[donor_labels[j] if donor_labels is not None else j for j in nn_indices]}")


def _impute_from_neighbors(donor_values, kind):
    """Mode for categorical/integer; mean for fractional/real."""
    vals = pd.Series(donor_values).dropna()
    if vals.empty:
        return np.nan
    if kind == 'categorical':
        return vals.mode().iloc[0]
    return float(vals.astype(float).mean())


def _stable_uniques(values):
    """Unique non-null values with a deterministic order for tie-breaking."""
    vals = pd.unique(pd.Series(values).dropna())
    try:
        return list(np.sort(np.asarray(vals)))
    except TypeError:
        return sorted(vals, key=lambda x: (str(type(x)), str(x)))


def _precompute_categorical_counts(donors, feature_cols, decision_col):
    """
    Frequency tables among complete donors for Case I / II categorical indices.

    Incomplete targets are excluded from strip_MV_Records(temp) in the paper
    path, so counts depend only on the complete donor set.
    """
    counts = {}
    for col in feature_cols:
        counts[col] = {}
        for cls, grp in donors.groupby(decision_col, sort=False):
            counts[col][cls] = grp[col].value_counts(dropna=True).to_dict()
    return counts


def _ic_categorical_from_counts(counts, col, dec_i, dec_k, val_i, val_k):
    """Case I / II categorical I_Cl using precomputed donor counts."""
    if dec_i == dec_k:
        gamma_pi = float(counts[col].get(dec_i, {}).get(val_i, 0))
        gamma_qk = float(counts[col].get(dec_i, {}).get(val_k, 0))
        return _safe_ratio_min(gamma_pi, gamma_qk)
    beta_r = float(counts[col].get(dec_i, {}).get(val_i, 0))
    delta_s = float(counts[col].get(dec_k, {}).get(val_k, 0))
    return _safe_ratio_max(beta_r, delta_s)


def _build_categorical_ic_tensor(counts, col, dec_levels, val_levels):
    """
    Dense I_Cl lookup: ic[dec_i, dec_k, val_i, val_k] for one attribute.
    """
    n_dec = len(dec_levels)
    n_val = len(val_levels)
    ic = np.zeros((n_dec, n_dec, n_val, n_val), dtype=float)
    for di, dec_i in enumerate(dec_levels):
        for dk, dec_k in enumerate(dec_levels):
            for vi, val_i in enumerate(val_levels):
                for vk, val_k in enumerate(val_levels):
                    ic[di, dk, vi, vk] = _ic_categorical_from_counts(
                        counts, col, dec_i, dec_k, val_i, val_k
                    )
    return ic


def _codes_from_levels(values, levels, *, missing_code=-1):
    """Map values to integer codes; nulls -> missing_code."""
    mapping = {lev: i for i, lev in enumerate(levels)}
    out = np.full(len(values), missing_code, dtype=np.int32)
    arr = np.asarray(values, dtype=object)
    for i, v in enumerate(arr):
        if pd.isnull(v):
            continue
        # Exact key first; also try numeric-normalized key for 0 vs 0.0
        if v in mapping:
            out[i] = mapping[v]
            continue
        matched = False
        for lev, code in mapping.items():
            try:
                if float(v) == float(lev):
                    out[i] = code
                    matched = True
                    break
            except (TypeError, ValueError):
                if v == lev:
                    out[i] = code
                    matched = True
                    break
        if not matched:
            # Unseen value contributes 0 counts via explicit extension not available;
            # treat as missing for IC (no contribution).
            out[i] = missing_code
    return out


def _batched_categorical_distance_matrix(
    target_dec_codes,
    target_feat_codes,
    donor_dec_codes,
    donor_feat_codes,
    ic_tensors,
    observed_mask,
):
    """
    Vectorized d_ik for a target chunk against all donors.

    Parameters
    ----------
    target_dec_codes : (t,)
    target_feat_codes : (t, F)  (-1 = missing attribute)
    donor_dec_codes : (m,)
    donor_feat_codes : (m, F)
    ic_tensors : list of (C, C, V_f, V_f) arrays
    observed_mask : (t, F) bool — True where target attribute is observed

    Returns
    -------
    distances : (t, m)
    """
    t = target_dec_codes.shape[0]
    m = donor_dec_codes.shape[0]
    d2 = np.zeros((t, m), dtype=float)
    for f, ic in enumerate(ic_tensors):
        obs = observed_mask[:, f]
        if not np.any(obs):
            continue
        td = target_dec_codes[obs][:, None]
        tv = target_feat_codes[obs, f][:, None]
        # Skip any residual missing codes
        valid = tv[:, 0] >= 0
        if not np.any(valid):
            continue
        td = td[valid]
        tv = tv[valid]
        rows = np.flatnonzero(obs)[valid]
        ic_block = ic[td, donor_dec_codes[None, :], tv, donor_feat_codes[None, :, f]]
        d2[rows] += np.square(ic_block)
    return np.sqrt(d2)


def _modes_from_nn_mask(nn_mask, donor_values):
    """
    Vectorized categorical mode over NN sets.

    Tie-break matches pandas Series.mode().iloc[0] (smallest / first sorted mode).
    """
    t, m = nn_mask.shape
    donor_values = np.asarray(donor_values, dtype=object)
    levels = _stable_uniques(donor_values)
    if not levels:
        return np.array([np.nan] * t, dtype=object)

    best_count = np.full(t, -1, dtype=np.int64)
    best_val = np.empty(t, dtype=object)
    best_val[:] = np.nan
    for v in levels:
        counts = np.sum(nn_mask & (donor_values == v), axis=1)
        take = counts > best_count
        best_val[take] = v
        best_count[take] = counts[take]
    return best_val


def _means_from_nn_mask(nn_mask, donor_values):
    """Vectorized mean over NN sets for fractional attributes."""
    vals = np.asarray(donor_values, dtype=float)
    masked = np.where(nn_mask, vals[None, :], np.nan)
    with np.errstate(all='ignore'):
        return np.nanmean(masked, axis=1)


def _impute_categorical_batch(
    incomplete,
    donors,
    feature_cols,
    decision_col,
    cat_counts,
    *,
    chunk_size=512,
    verbose=False,
):
    """
    Fully vectorized categorical imputation over target chunks.

    Complexity: O(T * M * F) arithmetic via NumPy broadcasting, memory
    bounded by processing targets in chunks of size ``chunk_size``.
    """
    result_updates = {}  # (orig_idx, col) -> value
    dec_levels = _stable_uniques(donors[decision_col])
    if not dec_levels:
        return result_updates

    donor_dec_codes = _codes_from_levels(donors[decision_col].to_numpy(), dec_levels)
    ic_tensors = []
    val_levels_per_col = []
    donor_feat_codes = np.empty((len(donors), len(feature_cols)), dtype=np.int32)
    for f, col in enumerate(feature_cols):
        val_levels = _stable_uniques(donors[col])
        val_levels_per_col.append(val_levels)
        ic_tensors.append(
            _build_categorical_ic_tensor(cat_counts, col, dec_levels, val_levels)
        )
        donor_feat_codes[:, f] = _codes_from_levels(donors[col].to_numpy(), val_levels)

    target_index = list(incomplete.index)
    n_targets = len(target_index)
    target_dec_codes = _codes_from_levels(
        incomplete[decision_col].to_numpy(), dec_levels
    )
    target_feat_codes = np.empty((n_targets, len(feature_cols)), dtype=np.int32)
    observed_mask = np.zeros((n_targets, len(feature_cols)), dtype=bool)
    missing_mask = np.zeros((n_targets, len(feature_cols)), dtype=bool)
    for f, col in enumerate(feature_cols):
        raw = incomplete[col].to_numpy()
        target_feat_codes[:, f] = _codes_from_levels(raw, val_levels_per_col[f])
        observed_mask[:, f] = pd.notnull(raw)
        missing_mask[:, f] = pd.isnull(raw)

    col_kinds = {c: _column_kind(donors[c]) for c in feature_cols}
    donor_value_arrays = {
        c: donors[c].to_numpy() for c in feature_cols
    }

    for start in range(0, n_targets, chunk_size):
        end = min(start + chunk_size, n_targets)
        dist = _batched_categorical_distance_matrix(
            target_dec_codes[start:end],
            target_feat_codes[start:end],
            donor_dec_codes,
            donor_feat_codes,
            ic_tensors,
            observed_mask[start:end],
        )
        nn = nearest_neighbor_mask(dist)
        z = compute_zd_matrix(dist) if verbose else None

        for local_i, orig_idx in enumerate(target_index[start:end]):
            if verbose:
                nn_idx = list(np.flatnonzero(nn[local_i]))
                print_distance_scores(
                    target_label=orig_idx,
                    distances=dist[local_i],
                    z_scores=z[local_i],
                    nn_indices=nn_idx,
                    donor_labels=list(range(len(donors))),
                )

        for f, col in enumerate(feature_cols):
            miss_local = missing_mask[start:end, f]
            if not np.any(miss_local):
                continue
            if col_kinds[col] == 'categorical':
                modes = _modes_from_nn_mask(nn[miss_local], donor_value_arrays[col])
                miss_rows = np.flatnonzero(miss_local)
                for k, local_i in enumerate(miss_rows):
                    oi = target_index[start + int(local_i)]
                    result_updates[(oi, col)] = modes[k]
                    if verbose:
                        print(
                            f"  imputed {col}[{oi}] = {modes[k]!r}  (categorical)"
                        )
            else:
                means = _means_from_nn_mask(nn[miss_local], donor_value_arrays[col])
                miss_rows = np.flatnonzero(miss_local)
                for k, local_i in enumerate(miss_rows):
                    oi = target_index[start + int(local_i)]
                    result_updates[(oi, col)] = float(means[k])
                    if verbose:
                        print(
                            f"  imputed {col}[{oi}] = {means[k]!r}  (fractional)"
                        )

    return result_updates


def _distances_target_to_donors(target_row, donors, feature_cols, decision_col, cat_counts=None):
    """
    Distances from one target row to every donor.

    Uses precomputed categorical counts when all compared attributes are
    categorical (typical for binary clinical indicators). Falls back to the
    general pairwise_distance path otherwise.
    """
    n_donors = len(donors)
    if cat_counts is not None and all(_column_is_categorical(donors, c) for c in feature_cols):
        # Single-row batch path for API compatibility / tests
        dec_levels = _stable_uniques(donors[decision_col])
        donor_dec_codes = _codes_from_levels(donors[decision_col].to_numpy(), dec_levels)
        target_dec_codes = _codes_from_levels(
            np.array([target_row[decision_col]], dtype=object), dec_levels
        )
        ic_tensors = []
        donor_feat = np.empty((n_donors, len(feature_cols)), dtype=np.int32)
        target_feat = np.empty((1, len(feature_cols)), dtype=np.int32)
        observed = np.zeros((1, len(feature_cols)), dtype=bool)
        for f, col in enumerate(feature_cols):
            val_levels = _stable_uniques(donors[col])
            ic_tensors.append(
                _build_categorical_ic_tensor(cat_counts, col, dec_levels, val_levels)
            )
            donor_feat[:, f] = _codes_from_levels(donors[col].to_numpy(), val_levels)
            target_feat[0, f] = _codes_from_levels(
                np.array([target_row[col]], dtype=object), val_levels
            )[0]
            observed[0, f] = not pd.isnull(target_row[col])
        return _batched_categorical_distance_matrix(
            target_dec_codes, target_feat, donor_dec_codes, donor_feat, ic_tensors, observed
        )[0]

    # General (fractional / mixed) path
    temp = pd.concat(
        [
            pd.DataFrame([target_row[feature_cols + [decision_col]]]),
            donors[feature_cols + [decision_col]],
        ],
        ignore_index=True,
    )
    return np.asarray(
        [
            pairwise_distance(temp, 0, donor_iloc, feature_cols=feature_cols)
            for donor_iloc in range(1, len(temp))
        ],
        dtype=float,
    )


def non_parametric_imputation(
    data,
    decision_col=None,
    feature_cols=None,
    verbose=False,
    *,
    chunk_size=512,
    return_timing=False,
):
    """
    Non-parametric missing-value imputation.

    Parameters
    ----------
    data : DataFrame
        Dataset S. Last column is the decision attribute unless decision_col
        is given.
    decision_col : str, optional
        Name of the physician diagnosis / class column.
    feature_cols : list of str, optional
        Attribute columns used for proximity (defaults to all except decision).
    verbose : bool, optional
        If True, print distance scores (and z-scores / NN) for each target row.
    chunk_size : int, optional
        Target-row chunk size for vectorized distance matrices (memory bound).
    return_timing : bool, optional
        If True, return ``(imputed_df, timing_dict)`` instead of only the frame.

    Returns
    -------
    DataFrame or (DataFrame, dict)
        Copy of data with missing attribute values imputed.
        Rows with missing decision are left unchanged (not imputed).
    """
    t_wall0 = time.perf_counter()
    df = prepare_numeric_data(pd.DataFrame(data).copy())

    if decision_col is None:
        decision_col = df.columns[-1]
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != decision_col]

    # Work only on rows with observed decision (paper assumes diagnosis present)
    work = filter_mv_decisions(df[[*feature_cols, decision_col]])
    # Reorder so decision is last (case helpers expect this)
    work = work[[*feature_cols, decision_col]].copy()

    complete = strip_MV_Records(work)
    incomplete = get_MV_records(work)

    timing = {
        "n_incomplete": int(len(incomplete)),
        "n_donors": int(len(complete)),
        "n_features": int(len(feature_cols)),
        "complexity": "O(T*M*F) vectorized (chunked targets)",
        "chunk_size": int(chunk_size),
        "backend": None,
        "seconds": None,
    }

    if incomplete.empty:
        timing["seconds"] = time.perf_counter() - t_wall0
        timing["backend"] = "noop"
        return (df, timing) if return_timing else df

    if complete.empty:
        raise ValueError("No complete records available to use as imputation donors.")

    donors = complete.reset_index(drop=True)
    result = df.copy()

    use_cat_fast = all(_column_is_categorical(donors, c) for c in feature_cols)
    cat_counts = (
        _precompute_categorical_counts(donors, feature_cols, decision_col)
        if use_cat_fast
        else None
    )

    if verbose:
        print(
            f"\nImputation: {len(incomplete)} incomplete row(s), "
            f"{len(donors)} complete donor(s); "
            f"backend={'vectorized-categorical' if use_cat_fast else 'sequential'}"
        )

    t_imp0 = time.perf_counter()
    if use_cat_fast:
        timing["backend"] = "vectorized-categorical"
        updates = _impute_categorical_batch(
            incomplete,
            donors,
            feature_cols,
            decision_col,
            cat_counts,
            chunk_size=chunk_size,
            verbose=verbose,
        )
        for (orig_idx, col), value in updates.items():
            result.at[orig_idx, col] = value
    else:
        timing["backend"] = "sequential"
        timing["complexity"] = "O(T*M*F) sequential pairwise"
        col_kinds = {c: _column_kind(donors[c]) for c in feature_cols}
        for orig_idx, row in incomplete.iterrows():
            distances = _distances_target_to_donors(
                row, donors, feature_cols, decision_col, cat_counts=None
            )
            nn_local, z_scores = get_nearest_neighbors(distances)
            if verbose:
                print_distance_scores(
                    target_label=orig_idx,
                    distances=distances,
                    z_scores=z_scores,
                    nn_indices=nn_local,
                    donor_labels=list(donors.index.astype(object)),
                )
            donor_rows = donors.iloc[list(nn_local)]
            for col in feature_cols:
                if pd.isnull(row[col]):
                    kind = col_kinds[col]
                    imputed = _impute_from_neighbors(donor_rows[col], kind)
                    result.at[orig_idx, col] = imputed
                    if verbose:
                        print(f"  imputed {col}[{orig_idx}] = {imputed!r}  ({kind})")

    timing["seconds"] = time.perf_counter() - t_imp0
    timing["seconds_total"] = time.perf_counter() - t_wall0
    if verbose:
        print(
            f"Timing: {timing['seconds']:.4f}s imputing | "
            f"T={timing['n_incomplete']} M={timing['n_donors']} "
            f"F={timing['n_features']} | {timing['complexity']}"
        )

    return (result, timing) if return_timing else result


# --- Compatibility helpers used by older NMI.py sketches ---

def _normalize_mv_record(N, columns):
    """Accept a single MV record whether passed as a row or transposed Series."""
    N = pd.DataFrame(N)
    if len(N) == 1 and len(N.columns) == len(columns):
        return N.reset_index(drop=True)
    if N.shape[1] == 1 and len(N) == len(columns):
        return pd.DataFrame([N.iloc[:, 0].values], columns=list(columns))
    return N.iloc[[0]]


def get_indices(M, N):
    """IC matrices for each attribute missing in record N against complete records M."""
    M = pd.DataFrame(M)
    C = M.columns
    R = M.index
    N = _normalize_mv_record(N, C)
    non_mv_row_index = [R.get_loc(row) for row in R]

    mv_col_index = []
    for col_idx, col in enumerate(C):
        if col == C[-1]:
            break
        if pd.isnull(N.iloc[0, col_idx]):
            mv_col_index.append(col_idx)

    ICmaster = []
    for col_idx in mv_col_index:
        IC_i_list = []
        for i in non_mv_row_index:
            IC_k_list = [compute_ic(M, col_idx, i, k) for k in non_mv_row_index]
            IC_i_list.append(pd.Series(IC_k_list))
        ICmaster.append(pd.DataFrame(IC_i_list))
    return ICmaster


def compute_distance(IC):
    """
    Legacy helper: given a list of IC DataFrames (one per attribute),
    build pairwise distances d_ik = sqrt(sum_l IC_l[i,k]^2).
    """
    if not IC:
        return []
    mats = [pd.DataFrame(x).astype(float).values for x in IC]
    stacked = np.sqrt(sum(m ** 2 for m in mats))
    return stacked.tolist()


def compute_indices(M):
    """Compute IC matrices for every missing attribute location in M."""
    M = pd.DataFrame(M)
    C = M.columns
    R = M.index
    non_mv_row_index = [R.get_loc(row) for row in R if M.loc[row].notnull().all()]

    mv_col_index = []
    for col in C:
        if col == C[-1]:
            break
        for row in R:
            if pd.isnull(M.at[row, col]):
                mv_col_index.append(C.get_loc(col))
                break

    ICmaster = []
    for col_idx in mv_col_index:
        IC_i_list = []
        for i in non_mv_row_index:
            IC_k_list = [compute_ic(M, col_idx, i, k) for k in non_mv_row_index]
            IC_i_list.append(pd.Series(IC_k_list))
        ICmaster.append(pd.DataFrame(IC_i_list))
    return ICmaster

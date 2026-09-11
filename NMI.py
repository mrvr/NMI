# -*- coding: utf-8 -*-
"""
Source code for New Imputation Method

Workflow:
  1. Load data (`?` = missing)
  2. Carve rows with missing decision as held-out **test** data
  3. Train / impute on remaining rows (known decision)
  4. Test partition is available after training for evaluation
"""
import pandas as pd
from nmilib import (
    CATEGORICAL_TYPES,
    non_parametric_imputation,
    split_train_test_by_decision,
    carve_and_save_train_test,
)

NA = ["", "?"]

# --- Split data3: missing-dec rows → test; known-dec rows → train ---
datatable = pd.read_csv("data3.txt", na_values=NA)
train, test = split_train_test_by_decision(datatable, decision_col="dec")
carve_and_save_train_test(
    datatable,
    train_path="data3_train.txt",
    test_path="data3_test.txt",
    decision_col="dec",
)

print("CATEGORICAL_TYPES:", list(CATEGORICAL_TYPES))
print(f"\nTotal rows: {len(datatable)}")
print(f"Train (known dec): {len(train)}  → data3_train.txt")
print(f"Test  (missing dec): {len(test)} → data3_test.txt")
print("\nTest data (missing decision) carved out:\n", test.head(10))

# --- Training: impute attribute MVs on the train partition ---
print("\nTrain before imputation:\n", train.head(8))
train_imputed = non_parametric_imputation(train, decision_col="dec", verbose=True)
print("\nTrain after imputation:\n", train_imputed.head(8))

# --- After training, test partition is ready for evaluation ---
print(
    f"\nTraining done. Held-out test set has {len(test)} rows "
    f"with missing decision (not used during training)."
)

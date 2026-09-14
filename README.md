# NMI — Non-Parametric Missing-Value Imputation

Research prototype of a **non-parametric indexing** method for imputing missing attribute values, based on proximity to complete donor records within / across decision classes.

Reference method: V. Sree Hari Rao & M. Naresh Kumar (IEEE TITB / JBHI).

## What it does

1. Keep rows with a known **decision** (class / diagnosis) column  
2. Use **complete** rows as donors  
3. For each incomplete row, compute attribute-wise indices \(I_{C_l}\), distances \(d_{ik}\), and z-score nearest neighbours (\(z(d) \le 0\))  
4. Impute **mode** (categorical / integer) or **mean** (fractional / real)  
5. Rows with a missing decision are left unchanged (held out)

For pictorial formulas and flowcharts, see [`docs/imputation_flowchart.md`](docs/imputation_flowchart.md).

## Requirements

- Python 3.10+ (tested with 3.12)
- Dependencies in `requirements.txt`: `pandas`, `numpy`, `matplotlib`, `pytest`

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick start

```python
import pandas as pd
from nmilib import non_parametric_imputation

df = pd.read_csv("data.txt", na_values=["", "?"])
imputed = non_parametric_imputation(df, decision_col="dec")
print(imputed)
```

Demo script (carve train/test on `data3.txt`, impute train with distance prints):

```bash
python NMI.py
```

## Data format

| Kind | Example files | Notes |
|------|---------------|--------|
| Small examples | `data.txt`, `data1.txt`, `data2.txt` | Header row; `?` = missing; last column `dec` |
| Synthetic | `data3.txt`…`data5.txt` (+ `*_master.txt`) | Masters are complete ground truth |
| Dengue system | `dengue.csv` | Features: `Fever`, `Headache`, `JointPain`, `Bleeding`; decision: `Dengue` |

- Attribute columns come before the decision column.  
- Decision column defaults to the **last** column if not passed as `decision_col`.  
- For dengue runs, `Name` is an ID and is not used as a proximity feature.

## Main API (`nmilib.py`)

| Function | Purpose |
|----------|---------|
| `non_parametric_imputation(data, decision_col=..., feature_cols=..., verbose=False, return_timing=False)` | Impute missing attributes |
| `split_train_test_by_decision(...)` | Train = known decision; test = missing decision |
| `carve_and_save_train_test(...)` | Same split, write CSVs with `?` for MVs |

Categorical clinical / binary features use a **vectorized** distance + NN path (complexity \(O(T \times M \times F)\): incomplete targets × donors × features).

## Tests

```bash
# Unit + paper example
python -m pytest -q tests/test_nmi_imputation.py tests/test_paper_example.py

# Dengue accuracy + timing (10k / 20k / 50k rows)
python -m pytest -s tests/test_dengue_system.py::TestDengueImputationAccuracy::test_scale_runs_10000_20000_50000
```

## Project layout

```
NMI.py                 # Demo entry script
nmilib.py              # Imputation library
tests/                 # Unit, paper, and dengue system tests
docs/                  # Flowcharts and formula images
data*.txt              # Small / synthetic datasets
dengue.csv             # Large dengue sample for system tests
requirements.txt
pytest.ini
```

## License / status

Research / prototype code. Not packaged for production use.

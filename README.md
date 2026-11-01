# ml-end-to-end

![CI](https://github.com/usmanali-hub/ml-end-to-end/actions/workflows/ci.yml/badge.svg)

A supervised-learning project on telecom customer churn, built to be reproducible from the first
commit: a validated data loader, a documented dataset, one fixed train/test split, and honest
baselines that every later model has to beat. Models come next; this stage sets the bar they are
measured against.

The data is IBM's public Telco Customer Churn sample (7,043 customers, 26.5% churned).
See the [data card](docs/data_card.md) for columns, quality notes and limitations.

## Baselines

5-fold cross-validation on the training split (5,634 rows). The test split (1,409 rows) has not been used.
Primary metric: **PR-AUC**, because only 1 in 4 customers churns and accuracy rewards a model that
never flags anyone.

| baseline | PR-AUC | ROC-AUC | accuracy | recall | precision |
|---|---|---|---|---|---|
| `majority` (nobody churns) | 0.265 | 0.500 | 0.735 | 0.000 | 0.000 |
| `random_prior` (weighted coin) | 0.279 | 0.506 | 0.606 | 0.274 | 0.265 |
| `month_to_month` (contract rule) | 0.416 | 0.739 | 0.655 | 0.887 | 0.428 |
| `short_tenure` (first year rule) | 0.479 | 0.735 | 0.718 | 0.550 | 0.473 |
| `logistic_reference` (logistic regression, all features) | **0.661** | 0.846 | 0.802 | 0.543 | 0.653 |

Full table with standard deviations: [`reports/baseline.md`](reports/baseline.md).

What this says:

- The "never churns" model is 73.5% accurate and finds no churners, so accuracy cannot be the goal.
- Two one-line business rules already reach ROC-AUC 0.74. Contract type and tenure carry a lot of signal, and a model that cannot beat a rule written on a whiteboard is not earning its complexity.
- A plain logistic regression on all features reaches PR-AUC 0.661, far above the rules. **That is the bar for later models**: a PR-AUC clearly above 0.661, meaning by more than the fold-to-fold spread (about 0.02).
- PR-AUC does not depend on the 12-month cutoff in `short_tenure`: it ranks customers by score, and the cutoff only changes precision and recall (a test checks this). So the number is not tuned by that choice.

## Does the model lean on `tenure`?

The data card warns that `tenure` partly encodes the outcome. [`reports/tenure_dependence.md`](reports/tenure_dependence.md)
measures it with the same logistic regression:

| inputs | PR-AUC |
|---|---|
| all features | 0.661 |
| without `tenure` | 0.651 |
| without `tenure` and `TotalCharges` | 0.612 |

Dropping both time columns costs 7.5% of PR-AUC and the model stays well above every rule baseline, so it
does not rest on them alone. This bounds the effect rather than removing it: `MonthlyCharges`, `Contract`
and the add-on columns may also reflect how long a customer has stayed.

## Run it

```bash
pip install -e ".[dev]"
python -m mlproject check      # load, clean and validate the raw file
python -m mlproject all        # regenerate docs/data_card.md and reports/baseline.*
pytest
```

## Layout

```
data/raw/telco_churn.csv     original file, untouched (source and licence: data/NOTICE.md)
src/mlproject/data.py        load, type fixes, schema validation
src/mlproject/split.py       stratified 80/20 split, seed 42
src/mlproject/metrics.py     PR-AUC, ROC-AUC, accuracy, precision, recall, F1
src/mlproject/baseline.py    the five baselines, cross-validation and the tenure ablation
src/mlproject/card.py        generates the data card from the data
docs/data_card.md            generated
reports/baseline.md|json     generated
reports/tenure_dependence.md generated
```

## Design decisions

- **Missing values stay missing.** 11 new customers have a blank `TotalCharges`. The loader converts them to NaN and nothing more; filling them with a statistic from the full dataset would leak test information into training, so imputation will live inside the model pipeline.
- **One split, fixed once.** Stratified so train and test have the same churn rate. Choices are made with cross-validation on the training rows; the test rows are for the final model only.
- **Generated docs are tested.** The data card and baseline report are produced by code, and a test (and CI) fails if the committed copies are out of date.
- **Quoted numbers are asserted.** The data-quality claims (11 blank rows, all with tenure 0; 1,526 customers without internet) are checked by tests against the data.

## Limits

This is a snapshot with no dates, so `tenure` partly reflects the outcome for customers who already left. The data is also a public tutorial sample: `TotalCharges` is almost exactly `MonthlyCharges` x `tenure`, which real billing rarely is. It is a good dataset for demonstrating a workflow and a poor basis for claims about real customers, so scores here show the method, not what a model would achieve on a real telecom. The data card lists these and other caveats.

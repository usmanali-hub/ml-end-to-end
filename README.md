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

## What the exploration found

[`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) explores the training rows only (the test rows stay untouched) and checks each pattern against its most likely alternative explanation. In short:

- **Contract type is the strongest signal**: month-to-month customers churn at 42.7%, one-year at 11.1%, two-year at 2.9%.
- **Risk is front-loaded**: customers in their first six months are 21% of the base but 41% of churners.
- **Fibre churn is not just a contract effect**: the gap against DSL persists within every contract type (55.1% vs 31.8% on month-to-month).
- **Monthly charges mostly proxy service type**: churn rises with the bill overall, but falls with the bill within DSL and is flat within fibre, so the coefficient should not be read as a price effect.
- **Streaming add-ons add little beyond "has internet"**, and gender carries no signal (26.6% vs 26.4%); senior citizens churn at 41.1% vs 23.7%, which the later fairness checks will look at.

Every number in the notebook's prose is computed by `mlproject.eda` and asserted in the tests; CI re-executes the notebook. These are associations in one sample, not causes.

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

## Preprocessing

One recipe turns the 19 raw columns into 45 numeric columns (median imputation and standardisation for numbers, one-hot
encoding for categories). It is the first step of every model pipeline, so each statistic it learns is computed from the
rows that model is being fitted on, and a test checks that. Five variants were compared on identical folds:

| recipe | PR-AUC | change vs reference | folds better |
|---|---|---|---|
| reference | 0.661 | - | - |
| no scaling | 0.662 | +0.000 | tied |
| missing-value indicator | 0.661 | +0.000 | tied |
| log of `TotalCharges` | 0.673 | +0.011 | 5 of 5 |
| `tenure` in 6 bins | 0.666 | +0.005 | 4 of 5 |

Scaling leaves the score unchanged here but stops logistic regression converging, and a missing-value flag adds
nothing (8 imputed rows). The log transform is a small but consistent gain, so later models start from it. Binning
`tenure` is within noise. Details: [`reports/preprocessing.md`](reports/preprocessing.md).

## Linear and logistic regression

All models start from the recommended preprocessing recipe and are compared on the same cross-validation folds as the
baselines (training split only, test split untouched).

| model | PR-AUC | recall | precision | Brier |
|---|---|---|---|---|
| linear regression on the 0/1 outcome | 0.667 | 0.490 | 0.682 | 0.136 |
| logistic regression, L2, C = 0.03 (chosen) | **0.674** | 0.531 | 0.680 | 0.133 |
| logistic regression, L1, C = 0.1 | 0.673 | 0.530 | 0.674 | 0.133 |
| logistic regression, balanced class weights | 0.673 | 0.795 | 0.530 | 0.161 |

- **The penalty barely matters** once it is not extreme: PR-AUC is flat from C = 0.03 to 100. L1 matches L2 with about half the coefficients non-zero.
- **Plain linear regression ranks customers almost as well** (0.006 behind) but is not a probability: 17% of customers get a "probability" below 0 or above 1.
- **Reweighting the classes changes the trade-off, not the ranking**: recall rises from 0.53 to 0.80 and precision falls, with worse-calibrated probabilities. Moving the decision threshold does the same without distorting the probabilities.
- **Gain over the earlier reference is small** (0.661 to 0.674, mostly the log transform from the preprocessing step): a linear model on these inputs has about reached its limit, which is what the next models test.

Sweep, odds ratios and caveats: [`reports/linear_models.md`](reports/linear_models.md).

## Run it

```bash
pip install -e ".[dev]"
python -m mlproject check      # load, clean and validate the raw file
python -m mlproject all        # regenerate docs/data_card.md and the reports
pytest
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb   # re-run the EDA
```

## Layout

```
data/raw/telco_churn.csv     original file, untouched (source and licence: data/NOTICE.md)
src/mlproject/data.py        load, type fixes, schema validation
src/mlproject/split.py       stratified 80/20 split, seed 42
src/mlproject/metrics.py     PR-AUC, ROC-AUC, accuracy, precision, recall, F1
src/mlproject/preprocess.py  the preprocessing recipe (leakage-safe ColumnTransformer) and its variants
src/mlproject/models.py      linear and logistic models, penalty sweep, odds ratios
src/mlproject/baseline.py    the five baselines, cross-validation, tenure ablation, recipe comparison
src/mlproject/card.py        generates the data card from the data
src/mlproject/eda.py         tested exploration helpers (rates with intervals, Cramér's V)
notebooks/01_eda.ipynb       executed exploratory analysis with findings
docs/data_card.md            generated
reports/baseline.md|json     generated
reports/tenure_dependence.md generated
reports/preprocessing.md     generated
reports/linear_models.md      generated
```

## Design decisions

- **Missing values stay missing.** 11 new customers have a blank `TotalCharges`. The loader converts them to NaN and nothing more; filling them with a statistic from the full dataset would leak test information into training, so imputation lives inside the model pipeline (`preprocess.py`).
- **One split, fixed once.** Stratified so train and test have the same churn rate. Choices are made with cross-validation on the training rows; the test rows are for the final model only.
- **Generated docs are tested.** The data card and baseline report are produced by code, and a test (and CI) fails if the committed copies are out of date.
- **Quoted numbers are asserted.** The data-quality claims (11 blank rows, all with tenure 0; 1,526 customers without internet) are checked by tests against the data.

## Limits

This is a snapshot with no dates, so `tenure` partly reflects the outcome for customers who already left. The data is also a public tutorial sample: `TotalCharges` is almost exactly `MonthlyCharges` x `tenure`, which real billing rarely is. It is a good dataset for demonstrating a workflow and a poor basis for claims about real customers, so scores here show the method, not what a model would achieve on a real telecom. The data card lists these and other caveats.

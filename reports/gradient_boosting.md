# Gradient boosting

LightGBM grows shallow trees one after another, each fitted to the errors of the ensemble so far. Fixed settings: at least 40 rows per leaf, 80% of rows and 80% of columns per tree, one thread (so results reproduce). Inputs are imputed and one-hot encoded, fitted on training rows only. 5-fold stratified cross-validation on the training split (5,634 rows), the same folds as every earlier model. The test split has not been used.

## 1. Rounds: boosting keeps overfitting if you let it

Tree size and learning rate as chosen below (4 leaves, rate 0.05). `fit` is PR-AUC on the rows the booster was fitted to; `validation` is on held-out rows.

| rounds | fit PR-AUC | validation PR-AUC |
|---|---|---|
| 10 | 0.622 | 0.611 ± 0.013 |
| 25 | 0.652 | 0.638 ± 0.013 |
| 50 | 0.670 | 0.654 ± 0.017 |
| 100 | 0.691 | 0.669 ± 0.021 |
| 150 | 0.699 | 0.672 ± 0.022 |
| 200 | 0.706 | 0.672 ± 0.022 |
| 300 | 0.718 | 0.670 ± 0.022 |
| 400 | 0.728 | 0.668 ± 0.022 |
| 600 | 0.744 | 0.665 ± 0.021 |

- Validation PR-AUC peaks around 150 rounds and then declines slowly, while the fit score keeps rising (0.744 at 600 rounds against 0.665 on held-out rows). The number of rounds is a regularisation setting, not something to maximise.

## 2. Tree size and learning rate

For each setting, the round count with the best validation score:

| leaves | learning rate | best rounds | fit PR-AUC | validation PR-AUC |
|---|---|---|---|---|
| 4 | 0.02 | 600 | 0.712 | 0.672 ± 0.022 |
| 4 | 0.05 | 150 | 0.699 | 0.672 ± 0.022 |
| 4 | 0.1 | 100 | 0.706 | 0.671 ± 0.022 |
| 8 | 0.02 | 200 | 0.710 | 0.670 ± 0.025 |
| 8 | 0.05 | 100 | 0.718 | 0.671 ± 0.023 |
| 8 | 0.1 | 50 | 0.715 | 0.668 ± 0.022 |
| 16 | 0.02 | 150 | 0.734 | 0.669 ± 0.025 |
| 16 | 0.05 | 50 | 0.726 | 0.668 ± 0.024 |
| 16 | 0.1 | 25 | 0.724 | 0.662 ± 0.027 |

- Smaller learning rates need more rounds for the same score, and bigger trees peak earlier and lower: the best setting is within 0.010 of the worst, so none of these knobs is decisive.
- Chosen: **4 leaves, rate 0.05, 150 rounds**: within 0.001 of the best cell, preferring the simplest (fewest leaves, then fewest rounds).

## 3. Against the earlier models (differences from `logistic_l2_chosen` on the same folds; `tied` means within 0.002)

| model | fit PR-AUC | validation PR-AUC | Δ vs logistic | folds better | recall | precision | Brier |
|---|---|---|---|---|---|---|---|
| `logistic_l2_chosen` | 0.678 | 0.674 ± 0.018 | - | - | 0.531 | 0.680 | 0.133 |
| `random_forest_chosen` | 0.708 | 0.666 ± 0.022 | -0.008 ± 0.007 | 1 of 5 | 0.486 | 0.681 | 0.135 |
| `lightgbm_default` | 0.895 | 0.644 ± 0.018 | -0.030 ± 0.008 | 0 of 5 | 0.514 | 0.633 | 0.140 |
| `lightgbm_chosen` | 0.699 | 0.672 ± 0.022 | -0.002 ± 0.007 | tied | 0.536 | 0.664 | 0.133 |
| `sklearn_hist_same_settings` | 0.698 | 0.669 ± 0.024 | -0.005 ± 0.008 | 1 of 5 | 0.522 | 0.670 | 0.134 |
| `xgboost_same_settings` | 0.699 | 0.672 ± 0.022 | -0.002 ± 0.006 | tied | 0.527 | 0.665 | 0.133 |
| `lightgbm_nested_tuning` | - | 0.670 ± 0.022 | -0.003 ± 0.007 | 1 of 5 | - | - | - |

- **Tuned boosting ties logistic regression** (-0.002 ± 0.007, better in 3 of 5 folds) and is +0.006 above the tuned forest. Boosting is often the strongest learner on tabular data, but here it finds little the linear model missed: the signal is mostly additive and there are only 5,634 rows.
- **Defaults hurt**: LightGBM's own defaults (31 leaves, no sampling) score 0.644, 0.028 below the tuned version, and 0.030 below logistic regression.
- Two other implementations with the same tree size, rate and rounds land in the same place: scikit-learn's boosting scores 0.669 (-0.003 from LightGBM) and XGBoost 0.672 (-0.000). XGBoost's leaf limit works differently from LightGBM's, so its settings are the closest equivalent, not identical.

## 4. How optimistic is the tuned score?

The tuned row above picked its settings on the same folds it is scored on. `lightgbm_nested_tuning` repeats the whole search (same grid, same rule) inside each outer fold, using an inner 3-fold split of that fold's fitting rows only, and scores the outer fold once.

| outer fold | chosen leaves | rate | rounds | PR-AUC on the outer fold |
|---|---|---|---|---|
| 0 | 4 | 0.02 | 300 | 0.657 |
| 1 | 4 | 0.1 | 100 | 0.639 |
| 2 | 4 | 0.05 | 150 | 0.693 |
| 3 | 8 | 0.02 | 200 | 0.683 |
| 4 | 4 | 0.1 | 100 | 0.679 |

- The nested estimate is 0.670, 0.002 below the tuned row (0.672): that gap is the optimism from tuning on the reported folds.
- The procedure chose 4 different settings across 5 folds, and the best and worst cells of the grid differ by only 0.010, so no single setting is clearly best.

## What this does not show

- All scores are cross-validation estimates on the training split; the test split is still untouched and will be used once, for the final model.
- One dataset of 7,043 customers. On much larger or more interaction-heavy data boosting often pulls ahead of a linear model.
- Grid is small and the other regularisation settings are fixed, so a different search could move the tuned score by a few thousandths.
- 1 of 9 settings peak at the largest round count tried (600), so a longer run might score a little higher.

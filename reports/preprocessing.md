# Preprocessing

One recipe turns the 19 raw columns into a numeric matrix, and it is the first step of every model pipeline, so each statistic it learns comes from the rows that model is fitted on. The test split has not been used.

- **Numeric columns** (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`): median imputation, then standardisation. Only `TotalCharges` has gaps (8 blank values in the training split, all brand-new customers).
- **Categorical columns** (15): one-hot encoding. A category that never appeared in training becomes all zeros rather than an error.
- **Output:** 45 columns, no missing values.

## Does the recipe matter? Five variants, same logistic regression, 5-fold CV on 5,634 training rows

`Δ vs reference` is the PR-AUC difference from the reference recipe on the *same* folds (mean ± standard deviation over folds), which is far less noisy than comparing two averages.

| recipe | PR-AUC | Δ vs reference | folds better | solver converged |
|---|---|---|---|---|
| `reference` | 0.661 ± 0.022 | - | - | yes |
| `no_scaling` | 0.662 ± 0.02 | +0.000 ± 0.00 | tied | no |
| `missing_indicator` | 0.661 ± 0.022 | +0.000 ± 0.000 | tied | yes |
| `log_total_charges` | 0.673 ± 0.018 | +0.011 ± 0.006 | 5 of 5 | yes |
| `tenure_bins_6` | 0.666 ± 0.018 | +0.005 ± 0.010 | 4 of 5 | yes |

What the comparison shows:

- **Scaling does not change the score here but does change whether the solver finishes.** Without it the PR-AUC is the same to within 0.001, yet logistic regression hits its 2,000-iteration limit without converging. Scale-sensitive models (KNN, SVM, neural networks) are a different story.
- **A missing-value indicator adds nothing**: only 8 training rows were imputed.
- **Log-transforming `TotalCharges` helps a little, consistently**: +0.011 PR-AUC, better in 5 of 5 folds. It is a small gain, about 1.7% of the reference score.
- **Binning `tenure` is inconclusive**: a small average gain with a spread as large as the gain, so it is not adopted.

Later models start from `log_total_charges`. The baseline table keeps the plain reference recipe so the bar it sets does not move.

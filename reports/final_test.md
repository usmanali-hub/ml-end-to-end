# Final evaluation on the held-out test split

Each model was fitted on the whole training split (5,634 rows) and scored once on the untouched test split (1,409 rows, churn rate 26.5%). The models and the primary model (`logistic_l2_chosen`) were fixed from cross-validation before any test score was seen; this report did not choose between them. The 95% interval is a bootstrap over test rows.

| model | cross-validation PR-AUC | test PR-AUC | 95% interval | test ROC-AUC | test Brier |
|---|---|---|---|---|---|
| `logistic_l2_chosen` (primary) | 0.674 | 0.665 | 0.613 to 0.714 | 0.849 | 0.134 |
| `random_forest_chosen` | 0.666 | 0.662 | 0.610 to 0.709 | 0.846 | 0.135 |
| `lightgbm_chosen` | 0.672 | 0.663 | 0.611 to 0.711 | 0.847 | 0.135 |
| (always predict the base rate) | - | 0.265 | - | 0.500 | - |

Differences between models smaller than the interval are not evidence that one is better.

## Decisions by `logistic_l2_chosen`

The cutoff 0.33 maximised F1 on out-of-fold *training* scores; it was not tuned on the test split.

| cutoff | flagged | true positives | false positives | false negatives | precision | recall | accuracy |
|---|---|---|---|---|---|---|---|
| 0.50 | 290 | 195 | 95 | 179 | 0.672 | 0.521 | 0.806 |
| 0.33 | 493 | 275 | 218 | 99 | 0.558 | 0.735 | 0.775 |

The test split has now been used. Any further change to models or settings must be reported as made after seeing it.

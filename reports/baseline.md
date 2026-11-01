# Baseline results

5-fold stratified cross-validation on the training split (5,634 rows, seed 42), mean ± standard deviation over folds.
The churn rate is 26.5%, so a model with no ranking ability scores a PR-AUC near 0.265 and a ROC-AUC near 0.5.
The test split has not been used.

| baseline | pr_auc | roc_auc | accuracy | balanced_accuracy | recall | precision | f1 |
|---|---|---|---|---|---|---|---|
| `majority` | 0.265 ± 0.000 | 0.500 ± 0.000 | 0.735 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| `random_prior` | 0.279 ± 0.009 | 0.506 ± 0.016 | 0.606 ± 0.012 | 0.500 ± 0.016 | 0.274 ± 0.023 | 0.265 ± 0.022 | 0.269 ± 0.022 |
| `month_to_month` | 0.416 ± 0.003 | 0.739 ± 0.004 | 0.655 ± 0.006 | 0.729 ± 0.003 | 0.887 ± 0.013 | 0.428 ± 0.004 | 0.577 ± 0.002 |
| `short_tenure` | 0.479 ± 0.023 | 0.735 ± 0.008 | 0.718 ± 0.009 | 0.664 ± 0.014 | 0.550 ± 0.026 | 0.473 ± 0.015 | 0.509 ± 0.019 |
| `logistic_reference` | 0.661 ± 0.022 | 0.846 ± 0.014 | 0.802 ± 0.013 | 0.719 ± 0.022 | 0.543 ± 0.046 | 0.653 ± 0.028 | 0.592 ± 0.033 |

**Primary metric: PR-AUC.** Best baseline: `logistic_reference` at 0.661. A model is only worth keeping if it clearly beats that, by more than the fold-to-fold spread above.

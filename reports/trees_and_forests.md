# Decision trees and random forests

Trees are invariant to scaling, so the inputs are only imputed and one-hot encoded (fitted on training rows only). 5-fold stratified cross-validation on the training split (5,634 rows), the same folds as the baselines and the linear models. The test split has not been used.

## 1. A single tree: depth is the whole story

`fit` is the PR-AUC on the rows the tree was grown from; `validation` is on held-out rows. The gap between them is overfitting.

| max depth | leaves | fit PR-AUC | validation PR-AUC |
|---|---|---|---|
| 1 | 2 | 0.409 | 0.409 ± 0.002 |
| 2 | 4 | 0.494 | 0.494 ± 0.014 |
| 3 | 8 | 0.572 | 0.563 ± 0.017 |
| 4 | 16 | 0.617 | 0.601 ± 0.020 |
| 5 | 32 | 0.652 | 0.614 ± 0.026 |
| 6 | 59 | 0.686 | 0.615 ± 0.042 |
| 8 | 171 | 0.780 | 0.551 ± 0.017 |
| 10 | 379 | 0.887 | 0.467 ± 0.015 |
| 15 | 931 | 0.994 | 0.392 ± 0.015 |
| none | 1100 | 1.000 | 0.379 ± 0.013 |

- Validation PR-AUC peaks at depth 6 and then falls. Chosen: **depth 5**, the shallowest tree within 0.001 of the best.
- A tree allowed to grow without limit fits the training rows perfectly (1.000) and scores 0.379 on held-out rows, **worse than the one-line month-to-month contract rule (0.416)**. A model that memorises is not better than a rule.

## 2. A forest: tuning the trees

150 trees per cell. `leaf` is the minimum rows in a leaf; `max_features` is the share of inputs each split may look at.

| min leaf | max_features | fit PR-AUC | validation PR-AUC |
|---|---|---|---|
| 5 | sqrt | 0.836 | 0.657 |
| 5 | 0.3 | 0.881 | 0.658 |
| 5 | 0.5 | 0.904 | 0.652 |
| 10 | sqrt | 0.768 | 0.663 |
| 10 | 0.3 | 0.800 | 0.662 |
| 10 | 0.5 | 0.818 | 0.664 |
| 20 | sqrt | 0.724 | 0.663 |
| 20 | 0.3 | 0.744 | 0.666 |
| 20 | 0.5 | 0.755 | 0.666 |
| 40 | sqrt | 0.697 | 0.662 |
| 40 | 0.3 | 0.707 | 0.666 |
| 40 | 0.5 | 0.713 | 0.664 |

- Leaf size matters most: with leaf = 5 the forest overfits a little; from leaf = 20 upward it is flat. How many inputs a split may see barely matters.
- Chosen: **leaf = 40, max_features = 0.3**: within 0.001 of the best cell, preferring the most regularised setting.

More trees reduce variance with diminishing returns:

| trees | validation PR-AUC |
|---|---|
| 10 | 0.661 |
| 25 | 0.663 |
| 50 | 0.666 |
| 100 | 0.666 |
| 200 | 0.666 |
| 400 | 0.666 |

Beyond about 50 trees the score is flat, so 400 is more than needed and costs only time.

## 3. Against the logistic model (differences are from `logistic_l2_chosen` on the same folds; `tied` means within 0.002)

| model | fit PR-AUC | validation PR-AUC | Δ vs logistic | folds better | recall | precision | Brier |
|---|---|---|---|---|---|---|---|
| `logistic_l2_chosen` | 0.678 | 0.674 ± 0.018 | - | - | 0.531 | 0.680 | 0.133 |
| `decision_tree_chosen` | 0.652 | 0.614 ± 0.026 | -0.060 ± 0.014 | 0 of 5 | 0.528 | 0.635 | 0.141 |
| `decision_tree_unlimited` | 1.000 | 0.379 ± 0.013 | -0.295 ± 0.011 | 0 of 5 | 0.504 | 0.489 | 0.271 |
| `random_forest_default` | 1.000 | 0.615 ± 0.032 | -0.059 ± 0.020 | 0 of 5 | 0.487 | 0.626 | 0.146 |
| `random_forest_chosen` | 0.708 | 0.666 ± 0.022 | -0.008 ± 0.007 | 1 of 5 | 0.486 | 0.681 | 0.135 |

- **Averaging many trees repairs most of a single tree's weakness**: the tuned forest is +0.052 PR-AUC above the tuned tree. Tuning matters too: the default forest (leaf = 1) overfits to a fit PR-AUC of 1.000 and scores 0.615.
- **The forest does not beat logistic regression here** (-0.008 ± 0.007, better in 1 of 5 folds). The churn signal is mostly additive (contract, tenure, internet service), which is what a linear model captures; the extra flexibility of trees has little left to find in 5,634 rows. A more complex model is not automatically a better one.

## 4. What the forest leans on, and why two importance measures disagree

A pure-noise column (`random_noise`, drawn at random and unrelated to churn) was added as a yardstick. Impurity importance is computed while the trees grow; permutation importance is the validation PR-AUC lost when a column is shuffled. Fitted on 80% of the training split, evaluated on the other 20%.

| column | permutation drop (± sd over 10 shuffles) | permutation rank | impurity importance | impurity rank |
|---|---|---|---|---|
| `Contract` | +0.0649 ± 0.0133 | 1 | 0.2326 | 1 |
| `tenure` | +0.0621 ± 0.0170 | 2 | 0.1666 | 2 |
| `InternetService` | +0.0502 ± 0.0114 | 3 | 0.1054 | 4 |
| `TotalCharges` | +0.0223 ± 0.0115 | 4 | 0.0951 | 6 |
| `MonthlyCharges` | +0.0107 ± 0.0093 | 5 | 0.0627 | 7 |
| `OnlineSecurity` | +0.0093 ± 0.0109 | 6 | 0.1077 | 3 |
| `MultipleLines` | +0.0093 ± 0.0047 | 7 | 0.0128 | 11 |
| `TechSupport` | +0.0084 ± 0.0078 | 8 | 0.0955 | 5 |
| `PaymentMethod` | +0.0040 ± 0.0025 | 9 | 0.0363 | 8 |
| `PaperlessBilling` | +0.0028 ± 0.0052 | 10 | 0.0088 | 13 |
| `random_noise` | -0.0013 ± 0.0012 | 19 | 0.0128 | 10 |

- **Impurity importance rewards noise.** The meaningless column gets impurity rank 10 of 20 and outranks 10 real columns, because a continuous column offers many places to split and so many chances to fit chance patterns. Its permutation importance is -0.0013, about zero, as it should be.
- **The top of the ranking agrees with the exploration and the logistic model**: contract, tenure and internet service lead. Below the top few the permutation drops are smaller than their spread, so the order there should not be trusted.
- `tenure`, `TotalCharges` and `MonthlyCharges` overlap (the linear-model report shows `TotalCharges` is almost `tenure` times `MonthlyCharges`), so shuffling one leaves the others to compensate and each looks less important than the group is.

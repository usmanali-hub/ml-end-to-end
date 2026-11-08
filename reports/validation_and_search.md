# Cross-validation noise and hyperparameter search

Both analyses use only the training split (5,634 rows); the test split has not been used. Question 1: how much of a gap between two models is just which rows landed in which fold? Question 2: does picking the best of many tried settings inflate the score we report for it?

## 1. Fold-assignment noise: the same comparison under 4 different 5-fold splits

`repeat sd` is the spread of the mean score across the repeats (different fold assignments, same data). `fold SE` is the standard error of a single 5-fold mean, from the fold-to-fold spread. `vs logistic` is the model minus `logistic_l2_chosen` within each repeat.

| model | mean PR-AUC | repeat sd | fold SE | vs logistic (mean) | vs logistic (range over repeats) | repeats ahead of logistic |
|---|---|---|---|---|---|---|
| `logistic_l2_chosen` | 0.674 | 0.001 | 0.011 | - | - | - |
| `random_forest_chosen` | 0.665 | 0.003 | 0.011 | -0.009 | 0.008 | 0 of 4 |
| `lightgbm_chosen` | 0.671 | 0.002 | 0.010 | -0.003 | 0.006 | 1 of 4 |
| `lightgbm_default` | 0.650 | 0.004 | 0.011 | -0.024 | 0.010 | 0 of 4 |

- Reshuffling the folds moves a model's mean by about 0.004 (the largest sd across models), but a single 5-fold mean has a standard error of about 0.011. Reshuffling does not remove that: every repeat sees the same 5,634 customers, so it measures split noise, not sampling noise.
- Tuned boosting is -0.003 from logistic regression, inside one standard error, so their order is not established by this data. The tuned forest is -0.009 behind. LightGBM's untuned defaults are -0.024, well outside the noise measured here and behind logistic regression in every repeat.
- Earlier selections used a tolerance of 0.001, tighter than the noise: where a simpler model was chosen, it was chosen as a tie-break on simplicity, not on evidence of being better.

## 2. Search optimism: random search of increasing size, nested inside each outer fold

24 LightGBM settings were drawn at random (leaves, learning rate, rounds, minimum leaf rows, row and column sampling, L2 penalty). For each outer fold, every setting was scored by an inner 3-fold split of that fold's fitting rows; a search of size n tries the first n settings and takes the best. The winner is then scored once on the outer fold, which played no part in choosing it.

| settings tried | best inner score | winner on outer fold | inner minus outer (± sd over folds) | average setting on outer fold | best setting in hindsight | distinct winners |
|---|---|---|---|---|---|---|
| 1 | 0.670 | 0.670 | +0.000 ± 0.029 | 0.670 | 0.670 | 1 |
| 2 | 0.670 | 0.670 | +0.000 ± 0.029 | 0.654 | 0.670 | 1 |
| 4 | 0.671 | 0.668 | +0.002 ± 0.028 | 0.661 | 0.673 | 2 |
| 8 | 0.671 | 0.668 | +0.002 ± 0.028 | 0.663 | 0.674 | 2 |
| 16 | 0.671 | 0.671 | +0.000 ± 0.029 | 0.663 | 0.675 | 2 |
| 24 | 0.671 | 0.671 | +0.000 ± 0.029 | 0.662 | 0.675 | 2 |

- **No optimism large enough to measure here.** Inner minus outer is +0.000 for the full search, but its fold-to-fold sd is 0.029, so five outer folds cannot resolve anything smaller than roughly 0.013. The earlier nested check on the boosting grid found a similar small gap.
- **Searching helped a little.** The winner of 24 settings scores 0.671 on outer folds, +0.009 above an average setting (0.662), and 0.004 below the best setting chosen with knowledge of the outer fold (0.675). The very first setting drawn scores 0.670 on its own, so the extra tries added only +0.001 over that draw.
- The space is narrow (4 to 24 leaves, rates 0.01 to 0.2, at least 10 rows per leaf, an L2 penalty from 0.1 to 10): an average setting and the best one in hindsight differ by only 0.013. Optimism is a bigger risk when settings differ a lot, so the same search was repeated in a much wider space below.

### The same search in a wide space

Leaves 8 to 64, learning rate 0.02 to 0.3, as few as 1 row per leaf, 30% to 100% of rows and columns per tree, and an L2 penalty down to 0.001: unregularised settings are allowed, so many are poor and the settings differ a lot.

| settings tried | best inner score | winner on outer fold | inner minus outer (± sd over folds) | average setting on outer fold | best setting in hindsight | distinct winners |
|---|---|---|---|---|---|---|
| 1 | 0.663 | 0.666 | -0.003 ± 0.029 | 0.666 | 0.666 | 1 |
| 2 | 0.663 | 0.666 | -0.003 ± 0.029 | 0.621 | 0.666 | 1 |
| 4 | 0.663 | 0.666 | -0.003 ± 0.029 | 0.632 | 0.666 | 1 |
| 8 | 0.663 | 0.666 | -0.003 ± 0.029 | 0.630 | 0.666 | 1 |
| 16 | 0.667 | 0.671 | -0.003 ± 0.029 | 0.634 | 0.671 | 1 |
| 24 | 0.669 | 0.672 | -0.003 ± 0.024 | 0.638 | 0.676 | 4 |

- **Selecting mattered much more here**: the winner of 24 settings scores 0.672, +0.034 above an average setting (0.638), and within 0.004 of the best setting in hindsight. In the narrow space an average setting was only 0.013 from the best.
- **Inner minus outer is still about zero** (-0.003 ± 0.024), so no optimism shows up. This is not proof it is absent: inner scores come from models fitted on about 53% of the training rows (a third of the outer fitting rows held out) while the outer score comes from models fitted on 80%, so inner scores run low, and that bias works against the optimism being measured. A cleaner test would need more rows or repeated outer folds.

## What this does not show

- One dataset of 7,043 customers; the standard errors above would shrink with more rows.
- Repeated CV re-splits the same data, so it cannot estimate how a model would do on a different sample of customers.
- Only boosting was searched, in two spaces, with one random draw of settings in each.
- The search measure compares inner scores (models fitted on about 53% of the training rows) with outer scores (80%), so it cannot cleanly separate optimism from the effect of training-set size.
- The test split is still untouched.

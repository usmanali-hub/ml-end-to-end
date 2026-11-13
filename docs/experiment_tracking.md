# Experiment tracking

The reports in `reports/` are the conclusions. Tracking is the ledger behind them: one [MLflow](https://mlflow.org)
run per candidate model, so a number can be traced to the exact settings, data and code that produced it.

```
pip install -e ".[tracking]"
python -m mlproject track      # cross-validate the four chosen models and log one run each
python -m mlproject runs       # print the latest run of each model as a table
```

Runs go to a local SQLite store, `.mlflow/mlflow.db`, which git ignores; nothing needs a server. To browse it in
MLflow's web interface, install the full package (`pip install mlflow`) and run
`mlflow ui --backend-store-uri sqlite:///.mlflow/mlflow.db`; the skinny package used here has no interface.

## What each run records

| kind | contents |
|---|---|
| parameters | the model family and its chosen settings (for example `C`, `num_leaves`, `hidden`), plus `seed`, `cv_folds`, `n_train` and `positive_rate` |
| tags | `git_commit` (with `-dirty` if the working tree has uncommitted changes), `data_sha256` (a fingerprint of the raw CSV), `primary_metric`, and a note that the test split is untouched |
| metrics | for PR-AUC, ROC-AUC, Brier score, recall and precision: every fold as a step, and the mean and standard deviation (`cv_pr_auc_mean`, `cv_pr_auc_std`, ...) |
| artifact | `params.json`, the same parameters as a file |

Only cross-validation on the training split is logged. The 20% test split is not used by tracking; `python -m mlproject
final` remains the single place it is scored.

## Snapshot

Output of `python -m mlproject runs` after one `track`, on the author's machine. It matches the reports because both come
from the same 5 folds. The neural network row can differ in the third decimal on another processor.

| model | PR-AUC | fold sd | ROC-AUC | Brier |
|---|---|---|---|---|
| `logistic_l2_chosen` | 0.674 | 0.018 | 0.850 | 0.133 |
| `mlp_chosen` | 0.672 | 0.019 | 0.850 | 0.134 |
| `lightgbm_chosen` | 0.672 | 0.022 | 0.850 | 0.133 |
| `random_forest_chosen` | 0.666 | 0.022 | 0.848 | 0.135 |

Reading it: the first three are separated by less than the fold standard deviation (0.018 to 0.022), which is the same
tie the earlier reports describe. A table of means cannot show that on its own, which is why the per-fold values are
kept as steps.

## What tracking does and does not give

* It makes a result *reproducible to inspect*: the parameters, data fingerprint and commit identify the inputs. It does
  not by itself make a run reproducible; the training code and library versions still have to match.
* Re-running `track` adds new runs and keeps the old ones; `runs` shows only the latest run per model.
* `log_runs` refuses data that contains any customer from the test split (matched by `customerID`), and refuses data
  with no ID column because it cannot check. That stops tracked results being computed on the test rows by accident; it
  cannot stop someone logging a test score they computed elsewhere.
* The comparison table is selection by mean PR-AUC, which these models do not separate. Prefer the simplest.

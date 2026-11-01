# How much does the model lean on tenure?

The data card warns that `tenure` partly encodes the outcome (a churned customer's tenure is how long they stayed). This measures it: the same logistic regression, 5-fold cross-validation on the training split (5,634 rows), with the time-related columns removed. The test split has not been used.

| inputs | PR-AUC | ROC-AUC | recall | precision |
|---|---|---|---|---|
| `all_features` | 0.661 ± 0.022 | 0.846 ± 0.014 | 0.543 ± 0.046 | 0.653 ± 0.028 |
| `without_tenure` | 0.651 ± 0.022 | 0.840 ± 0.012 | 0.528 ± 0.043 | 0.645 ± 0.030 |
| `without_tenure_and_total_charges` | 0.612 ± 0.025 | 0.832 ± 0.011 | 0.523 ± 0.044 | 0.629 ± 0.024 |

Removing `tenure` and `TotalCharges` changes PR-AUC by -0.050 (-7.5%). The model keeps 92.5% of its PR-AUC without them and still scores well above the best rule baseline, so it does not depend on the time columns alone. This only bounds the effect: `MonthlyCharges`, `Contract` and the add-on columns may also reflect how long a customer has stayed.

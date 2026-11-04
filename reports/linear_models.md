# Linear and logistic regression

Every model starts from the `log_total_charges` preprocessing recipe, so scaling, imputation and encoding are fitted on training rows only. 5-fold stratified cross-validation on the training split (5,634 rows), the same folds as the baselines; all choices below are made on those folds. The test split has not been used.

## 1. How strong should the penalty be?

Smaller `C` means a stronger penalty. L2 shrinks every coefficient; L1 pushes some to exactly zero. PR-AUC is the mean over folds.

| C | L2 PR-AUC | L1 PR-AUC | L1 non-zero coefficients (of 45) |
|---|---|---|---|
| 0.001 | 0.658 | 0.265 | 0 |
| 0.003 | 0.666 | 0.599 | 2 |
| 0.01 | 0.671 | 0.647 | 11 |
| 0.03 | 0.674 | 0.671 | 16 |
| 0.1 | 0.673 | 0.673 | 21 to 23 |
| 0.3 | 0.673 | 0.673 | 25 to 26 |
| 1 | 0.673 | 0.673 | 31 to 36 |
| 3 | 0.673 | 0.673 | 38 to 39 |
| 10 | 0.673 | 0.673 | 37 to 39 |
| 100 | 0.673 | 0.673 | 35 to 39 |

- **Only a strong penalty hurts.** From C = 0.03 upward the PR-AUC is flat (0.670 to 0.674), so the choice is not delicate. At C = 0.001 L1 removes every coefficient and falls to the churn rate.
- **L1 matches L2 with half the inputs.** At C = 0.1 it keeps 21 to 23 of 45 coefficients and loses nothing measurable. The exact count varies with the solver's visiting order because the one-hot columns are exactly redundant, so L1 can give a duplicate column's weight to any of its copies. The *model* does not change: across orderings no customer's predicted probability moves by more than 3.4e-06.
- **Chosen: L2 with C = 0.03**, the strongest penalty within 0.001 PR-AUC of the best setting. Settings that tie are not worth telling apart, and the more heavily penalised one is the safer pick. The tolerance is a judgement call, so here is what other tolerances would pick:

| tolerance | C picked | PR-AUC at that C |
|---|---|---|
| 0 | 0.03 | 0.674 |
| 0.001 | 0.03 | 0.674 |
| 0.002 | 0.03 | 0.674 |
| 0.005 | 0.01 | 0.671 |

Every tolerance lands on a PR-AUC within 0.002 of the others, so the rule decides which of several equivalent models to report, not whether the model is good.

## 2. Five ways to model the same data (differences are from `logistic_l2_chosen` on the same folds; `tied` means within 0.002)

| model | PR-AUC | Δ vs chosen | folds better | ROC-AUC | recall | precision | Brier |
|---|---|---|---|---|---|---|---|
| `linear_probability` | 0.667 ± 0.020 | -0.006 ± 0.003 | 0 of 5 | 0.847 | 0.490 | 0.682 | 0.136 |
| `logistic_default` | 0.673 ± 0.018 | -0.001 ± 0.002 | tied | 0.851 | 0.540 | 0.671 | 0.133 |
| `logistic_l2_chosen` | 0.674 ± 0.018 | - | - | 0.850 | 0.531 | 0.680 | 0.133 |
| `logistic_l1_chosen` | 0.673 ± 0.018 | -0.001 ± 0.003 | tied | 0.850 | 0.530 | 0.674 | 0.133 |
| `logistic_balanced` | 0.673 ± 0.018 | -0.001 ± 0.001 | tied | 0.850 | 0.795 | 0.530 | 0.161 |

Brier score is the mean squared error of the predicted probabilities (lower is better; a model that always says the churn rate scores 0.195).

- **A linear regression on the 0/1 outcome ranks customers almost as well as logistic regression** (about 0.006 PR-AUC behind), but it is not a probability: it puts 17.4% of customers below 0 or above 1.
- **Reweighting the classes does not improve the ranking.** PR-AUC is unchanged. It trades precision for recall by moving the 0.5 cutoff, and it makes the probabilities worse (higher Brier). Move the decision threshold instead of reweighting, when the goal is a different recall.

## 3. What the chosen model leans on

Largest coefficients as odds ratios. Numeric inputs are standardised, so their ratio is per one standard deviation; a one-hot input's ratio is for having that value versus not.

| input | coefficient | odds ratio |
|---|---|---|
| `TotalCharges` | -0.65 | 0.52 |
| `Contract_Month-to-month` | +0.56 | 1.76 |
| `Contract_Two year` | -0.51 | 0.60 |
| `InternetService_Fiber optic` | +0.40 | 1.49 |
| `MonthlyCharges` | +0.37 | 1.44 |
| `InternetService_DSL` | -0.30 | 0.74 |
| `tenure` | -0.28 | 0.76 |
| `PaymentMethod_Electronic check` | +0.26 | 1.30 |
| `OnlineSecurity_No` | +0.25 | 1.28 |
| `TechSupport_No` | +0.23 | 1.25 |

Contract type, internet service and payment method point the way the exploration found. `TotalCharges` has a *negative* coefficient even though customers who stay longer have paid more: it is almost exactly `tenure` times `MonthlyCharges` (correlation 0.999), so the three share credit and a single sign is not interpretable. Removing `TotalCharges` lowers PR-AUC from 0.674 to 0.660, so it does carry some information, and the `tenure` coefficient moves from -0.28 to -0.76 and `MonthlyCharges` from +0.37 to +0.23, which shows how much of each coefficient was an artefact of the overlap. Read these as a description of this fitted model, not as the effect of changing a customer's contract or bill.

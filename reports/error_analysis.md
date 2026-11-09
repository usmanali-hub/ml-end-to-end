# Error analysis and feature importance

Out-of-fold predictions on the training split (5,634 customers); the test split has not been used. Errors are for `logistic_l2_chosen` at the cutoff 0.33 that maximises F1 on those same out-of-fold scores (chosen by looking at the training errors, so it is a description, not a held-out estimate).

## 1. Who the model gets wrong

| group | customers | mean score | mean tenure (months) | mean monthly charge | month-to-month | fibre optic |
|---|---|---|---|---|---|---|
| caught churners | 1,085 | 0.606 | 12.5 | 78.5 | 99.2% | 80.5% |
| missed churners | 410 | 0.194 | 33.9 | 65.1 | 61.0% | 42.0% |
| wrongly flagged | 812 | 0.509 | 17.3 | 76.6 | 96.8% | 71.7% |
| correctly left alone | 3,327 | 0.104 | 42.5 | 57.6 | 29.8% | 25.7% |

410 churners are missed and 812 loyal customers are flagged. The customers the model wrongly flags look like the churners it catches (tenure 17 against 12 months, 97% against 99% on month-to-month contracts). The churners it misses look like loyal customers (tenure 34 months, 61% month-to-month, against 30% for the customers it correctly leaves alone). In both directions the inputs describe an ordinary customer. Boosting and a forest, tuned earlier, did no better than this model, which is consistent with the table not containing what would separate them; it does not prove that.

Month-to-month customers account for 1,036 of the 1,222 errors (85%) and are 55% of customers. Recall is 0.07 on one-year and 0.00 on two-year contracts, because almost no one on them is flagged; 169 churners sit there.

### By segment

`gap` is observed churn rate minus mean predicted probability; `z` divides it by its sampling standard error.

| segment | value | customers | churn rate | mean score | gap | z | precision | recall | error rate |
|---|---|---|---|---|---|---|---|---|---|
| Contract | Month-to-month | 3,102 | 0.427 | 0.421 | +0.007 | +0.8 | 0.578 | 0.811 | 0.334 |
| Contract | One year | 1,173 | 0.111 | 0.113 | -0.002 | -0.2 | 0.265 | 0.069 | 0.124 |
| Contract | Two year | 1,359 | 0.029 | 0.043 | -0.014 | -3.2 | 0.000 | 0.000 | 0.029 |
| InternetService | DSL | 1,937 | 0.187 | 0.193 | -0.006 | -0.7 | 0.477 | 0.572 | 0.197 |
| InternetService | Fiber optic | 2,483 | 0.421 | 0.415 | +0.006 | +0.6 | 0.600 | 0.835 | 0.304 |
| InternetService | No | 1,214 | 0.072 | 0.076 | -0.003 | -0.5 | 0.625 | 0.057 | 0.071 |
| PaymentMethod | Bank transfer (automatic) | 1,244 | 0.162 | 0.164 | -0.003 | -0.3 | 0.517 | 0.537 | 0.156 |
| PaymentMethod | Credit card (automatic) | 1,213 | 0.149 | 0.153 | -0.004 | -0.4 | 0.465 | 0.481 | 0.160 |
| PaymentMethod | Electronic check | 1,891 | 0.457 | 0.452 | +0.005 | +0.5 | 0.604 | 0.862 | 0.322 |
| PaymentMethod | Mailed check | 1,286 | 0.193 | 0.195 | -0.002 | -0.2 | 0.543 | 0.581 | 0.175 |
| PaperlessBilling | No | 2,303 | 0.160 | 0.164 | -0.003 | -0.5 | 0.487 | 0.518 | 0.165 |
| PaperlessBilling | Yes | 3,331 | 0.338 | 0.336 | +0.002 | +0.3 | 0.594 | 0.794 | 0.253 |
| Partner | No | 2,905 | 0.328 | 0.328 | 0.000 | -0.1 | 0.582 | 0.771 | 0.256 |
| Partner | Yes | 2,729 | 0.199 | 0.199 | +0.000 | +0.0 | 0.552 | 0.646 | 0.175 |
| Dependents | No | 3,955 | 0.314 | 0.313 | +0.001 | +0.1 | 0.579 | 0.764 | 0.248 |
| Dependents | Yes | 1,679 | 0.152 | 0.154 | -0.003 | -0.3 | 0.529 | 0.541 | 0.143 |
| SeniorCitizen | 0 | 4,714 | 0.237 | 0.237 | 0.000 | -0.0 | 0.562 | 0.688 | 0.201 |
| SeniorCitizen | 1 | 920 | 0.411 | 0.411 | +0.000 | +0.0 | 0.598 | 0.836 | 0.298 |
| TechSupport | No | 2,771 | 0.418 | 0.414 | +0.003 | +0.3 | 0.583 | 0.831 | 0.318 |
| TechSupport | No internet service | 1,214 | 0.072 | 0.076 | -0.003 | -0.5 | 0.625 | 0.057 | 0.071 |
| TechSupport | Yes | 1,649 | 0.152 | 0.155 | -0.003 | -0.4 | 0.492 | 0.476 | 0.154 |
| tenure | 0-12 months | 1,737 | 0.473 | 0.468 | +0.005 | +0.4 | 0.634 | 0.865 | 0.301 |
| tenure | 13-24 | 818 | 0.286 | 0.294 | -0.008 | -0.5 | 0.522 | 0.774 | 0.268 |
| tenure | 25-48 | 1,290 | 0.202 | 0.206 | -0.003 | -0.3 | 0.445 | 0.579 | 0.231 |
| tenure | 49+ | 1,789 | 0.099 | 0.099 | +0.000 | +0.1 | 0.472 | 0.236 | 0.102 |
| MonthlyCharges | Q1 (lowest) | 1,409 | 0.109 | 0.113 | -0.004 | -0.4 | 0.540 | 0.351 | 0.104 |
| MonthlyCharges | Q2 | 1,413 | 0.250 | 0.252 | -0.002 | -0.2 | 0.518 | 0.666 | 0.238 |
| MonthlyCharges | Q3 | 1,406 | 0.370 | 0.369 | +0.001 | +0.1 | 0.608 | 0.856 | 0.257 |
| MonthlyCharges | Q4 (highest) | 1,406 | 0.333 | 0.328 | +0.005 | +0.4 | 0.574 | 0.750 | 0.268 |

Segments with a calibration gap beyond 2 standard errors: 1 of 29 (Contract = Two year, z -3.2); with 29 segments examined and segments overlapping, about 1.3 would be expected by chance even for a perfectly calibrated model. Treat a flag as a question to ask, not a finding.

## 2. Do the two models fail on the same customers?

`lightgbm_chosen` (its own best-F1 cutoff) misses 415 churners against 410 for the logistic model; the sets of missed churners overlap by 84% (Jaccard) and the sets of false alarms by 81%. Their scores have rank correlation 0.983. Two models that make nearly the same mistakes will not fix each other, so averaging them is unlikely to add much.

## 3. Which inputs the models rely on

Per raw input column (one-hot groups kept together), mean ± standard deviation over the 5 folds of the fall in held-out PR-AUC. *Permutation* shuffles the column in the held-out fold (10 shuffles) without refitting; *drop-column* refits without it. Columns are ordered by the two models' permutation importance combined.

| column | logistic: permutation | logistic: drop-column | LightGBM: permutation | LightGBM: drop-column |
|---|---|---|---|---|
| Contract | 0.033 ± 0.003 | 0.006 ± 0.003 | 0.108 ± 0.009 | 0.006 ± 0.005 |
| TotalCharges | 0.119 ± 0.013 | 0.013 ± 0.003 | 0.012 ± 0.004 | 0.001 ± 0.002 |
| tenure | 0.023 ± 0.009 | 0.002 ± 0.002 | 0.108 ± 0.021 | 0.000 ± 0.002 |
| InternetService | 0.028 ± 0.005 | 0.002 ± 0.002 | 0.038 ± 0.007 | 0.005 ± 0.004 |
| MonthlyCharges | 0.032 ± 0.004 | 0.000 ± 0.001 | 0.019 ± 0.007 | 0.002 ± 0.002 |
| OnlineSecurity | 0.009 ± 0.005 | 0.002 ± 0.005 | 0.018 ± 0.008 | 0.002 ± 0.008 |
| PaperlessBilling | 0.011 ± 0.002 | 0.005 ± 0.004 | 0.011 ± 0.003 | 0.004 ± 0.005 |
| MultipleLines | 0.012 ± 0.006 | 0.007 ± 0.006 | 0.010 ± 0.006 | 0.007 ± 0.006 |
| TechSupport | 0.007 ± 0.003 | 0.002 ± 0.003 | 0.011 ± 0.004 | 0.004 ± 0.003 |
| PaymentMethod | 0.008 ± 0.004 | 0.002 ± 0.003 | 0.008 ± 0.003 | 0.001 ± 0.002 |
| StreamingMovies | 0.005 ± 0.004 | 0.001 ± 0.004 | 0.001 ± 0.003 | 0.002 ± 0.002 |
| StreamingTV | 0.004 ± 0.002 | 0.000 ± 0.003 | 0.001 ± 0.001 | 0.001 ± 0.001 |
| OnlineBackup | 0.002 ± 0.002 | 0.000 ± 0.002 | 0.002 ± 0.001 | 0.002 ± 0.001 |
| Dependents | 0.002 ± 0.002 | 0.001 ± 0.001 | 0.002 ± 0.001 | 0.002 ± 0.002 |
| SeniorCitizen | 0.001 ± 0.001 | 0.000 ± 0.002 | 0.001 ± 0.001 | 0.002 ± 0.004 |
| PhoneService | 0.001 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.001 | 0.002 ± 0.003 |
| DeviceProtection | 0.000 ± 0.001 | 0.000 ± 0.001 | 0.000 ± 0.000 | 0.003 ± 0.001 |
| Partner | 0.000 ± 0.000 | 0.000 ± 0.001 | 0.000 ± 0.000 | 0.001 ± 0.003 |
| gender | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.001 ± 0.002 |

The two measures disagree sharply where inputs overlap. For the logistic model, permuting `TotalCharges` costs 0.119 PR-AUC but refitting without it costs 0.013: as fitted, the model leans hard on it, but most of that reliance disappears when the model is refitted, because other columns can stand in for it. For LightGBM, `tenure` is 0.108 by permutation and 0.000 by drop-column. `TotalCharges` is roughly tenure times monthly charge, so the three columns carry overlapping information, and permuting one of them also creates customer profiles that cannot occur (a long tenure with a tiny total charge) which the model never saw. Permutation importance therefore overstates reliance on overlapping columns; drop-column importance understates it, because a substitute exists.

Rank agreement between the models' permutation importances is Spearman 0.91. `Contract` ranks 2 for the logistic model and 1 for LightGBM by permutation. For both models the least important columns include `gender`, `Partner`, `DeviceProtection` (logistic permutation importance no larger than 0.001 in absolute value).

## What this does not show

- Importance is reliance of a fitted model on a column, not a cause of churn. Customers choose their contract; offering a two-year contract to someone does not turn them into a two-year customer.
- Differences smaller than the fold standard deviation (several columns in the middle of the table) are ties; the ranks of those columns should not be read.
- Segment errors depend on the cutoff, which was chosen on the same scores. At a different cutoff the segments with the most errors would change.
- Everything is cross-validation on the training split of one tutorial dataset. The segment findings are descriptive of these 5,634 customers.

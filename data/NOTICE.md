# Data source and licence

`raw/telco_churn.csv` is the "Telco Customer Churn" sample dataset (7,043 customers, 21 columns),
downloaded unchanged from the IBM repository
<https://github.com/IBM/telco-customer-churn-on-icp4d> (`data/Telco-Customer-Churn.csv`), which is
published under the Apache License 2.0. The file is the original, byte for byte; every cleaning
step happens in code (`src/mlproject/data.py`) so the raw file stays auditable.

The dataset is a public sample of a fictional telecom company. It is used here for learning and
demonstration, not as evidence about any real customer base.

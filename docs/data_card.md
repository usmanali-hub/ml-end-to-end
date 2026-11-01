# Data card: Telco customer churn

## Summary

- **Source**: IBM's public "Telco Customer Churn" sample dataset; licence and provenance in [`data/NOTICE.md`](../data/NOTICE.md).
- **Size**: 7,043 customers, 21 columns after cleaning (one row per customer, `customerID` is unique).
- **Target**: `churned`. 1,869 of 7,043 customers churned (26.5%), so the classes are imbalanced.
- **Split**: stratified 80%/20%, seed 42: 5,634 training rows and 1,409 test rows, both at the same churn rate. The test rows are reserved for the final model.

## Intended use

Learning and demonstrating an end-to-end classification workflow: exploration, preprocessing, model comparison, calibration and deployment. It is a public sample of a telecom company, not evidence about any real customer base.

**Not suitable for**: decisions about real people. In particular `gender` and `SeniorCitizen` describe protected groups and should not drive retention, pricing or service decisions; later work checks model behaviour across them rather than assuming it is fair.

## Columns

| Column | Type | Values | Missing | Meaning |
|---|---|---|---|---|
| `customerID` | text | 7,043 unique | 0 | Unique customer identifier. Not a feature. |
| `gender` | category | Female, Male | 0 | Female or Male as recorded by the company. |
| `Partner` | category | No, Yes | 0 | Whether the customer has a partner. |
| `Dependents` | category | No, Yes | 0 | Whether the customer has dependents. |
| `PhoneService` | category | No, Yes | 0 | Whether the customer has phone service. |
| `MultipleLines` | category | No, No phone service, Yes | 0 | Multiple phone lines; 'No phone service' when there is no phone service. |
| `InternetService` | category | DSL, Fiber optic, No | 0 | Internet provider type: DSL, Fiber optic or none. |
| `OnlineSecurity` | category | No, No internet service, Yes | 0 | Online security add-on; 'No internet service' when there is no internet. |
| `OnlineBackup` | category | No, No internet service, Yes | 0 | Online backup add-on; same convention. |
| `DeviceProtection` | category | No, No internet service, Yes | 0 | Device protection add-on; same convention. |
| `TechSupport` | category | No, No internet service, Yes | 0 | Tech support add-on; same convention. |
| `StreamingTV` | category | No, No internet service, Yes | 0 | Streaming TV add-on; same convention. |
| `StreamingMovies` | category | No, No internet service, Yes | 0 | Streaming movies add-on; same convention. |
| `Contract` | category | Month-to-month, One year, Two year | 0 | Contract term: month-to-month, one year or two year. |
| `PaperlessBilling` | category | No, Yes | 0 | Whether the customer uses paperless billing. |
| `PaymentMethod` | category | Bank transfer (automatic), Credit card (automatic), Electronic check, Mailed check | 0 | How the customer pays. |
| `SeniorCitizen` | binary | 0, 1 | 0 | 1 if the customer is a senior citizen, else 0. |
| `tenure` | integer | 0 to 72 | 0 | Months the customer has been with the company. |
| `MonthlyCharges` | float | 18.25 to 118.75 | 0 | Current monthly charge. |
| `TotalCharges` | float | 18.8 to 8684.8 | 11 | Total charged over the customer's lifetime; blank in the raw file for brand-new customers. |
| `churned` | binary | 0, 1 | 0 | Target. 1 if the customer left the company (raw column 'Churn' = Yes), else 0. |

## Data quality notes

- **Blank `TotalCharges`**: 11 rows. All have `tenure` = 0 (yes) and 0 of them churned: they are customers who joined but have not been billed yet. The raw file stores them as blank text, which makes the whole column load as strings; the loader converts them to NaN and leaves imputation to the model pipeline, fitted on training data only.
- **Redundant categories**: 1,526 customers have no internet service, so six add-on columns say 'No internet service' for the same rows; 682 have no phone service, which `MultipleLines` repeats as 'No phone service'. Seven columns therefore restate two facts.
- **`TotalCharges` is almost `MonthlyCharges` x `tenure`**: the median gap is $28.65 (2.0% of the expected amount) and the largest $373.25. It carries little beyond tenure and price. The remaining gap is probably plan or price changes over time, which cannot be checked because the data has no dates.
- **Identical feature rows**: 40 customers share every feature value with another customer. With so many categorical columns this is plausible for different people, so they are kept.
- **Types**: `SeniorCitizen` is 0/1 while every other yes/no column is text 'Yes'/'No'.

## Limitations

- **A snapshot, not a history.** There are no dates. `tenure` for a churned customer is how long they stayed, while for an active customer it is how long they have stayed so far, so tenure partly encodes the outcome. Treat results as descriptive; a deployed model would need features known *before* the decision point.
- **One company, one moment.** Nothing here says how a model would behave on another region, product mix or year.
- **No cost information.** Whether a missed churner or a wasted retention offer is worse is a business decision this dataset cannot make, so threshold choices stay explicit and are not tuned to a made-up cost.

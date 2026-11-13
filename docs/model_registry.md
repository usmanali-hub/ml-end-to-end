# Saving, versioning and scoring

## What is saved

`python -m mlproject train` writes `models/<name>/<version>/`:

| file | contents |
|---|---|
| `model.joblib` | the fitted pipeline: preprocessing and logistic regression together, so scoring cannot skip a step |
| `metadata.json` | name, version, SHA-256, creation time (UTC), input columns, training rows, data fingerprint, git commit, decision threshold, library versions |

`models/` is ignored by git: models are build outputs, rebuilt from the data and the code with one command.

## Versions

The version is the first 12 hex characters of the SHA-256 of `model.joblib`. Two consequences:

* saving identical bytes twice is a no-op, and
* a different model can never reuse an old version number.

It is a content version, not a "v1, v2" counter, so it says nothing about which model is newer or better. `predict`
uses the newest by creation time unless `--version` is given. Creation time has one-second resolution, so two models
saved in the same second are ordered by version string, not by when they were made.

## Scoring checks

Before the model sees any row, `registry.prepare` checks the input and refuses the whole batch if anything is wrong,
reporting the file lines (the header is line 1) and how many rows were affected:

* every one of the 19 feature columns must be present (extra columns are ignored);
* numeric columns must parse; `tenure` must be a whole number of months, 0 or more; `MonthlyCharges` positive;
  `TotalCharges` not negative, and **may be blank** (new customers), which the pipeline imputes;
* `SeniorCitizen` is 0 or 1;
* every categorical value must be one the model was trained on.

Refusing is deliberate. Guessing a value or dropping a customer silently would produce a score that looks normal and is wrong.

## Load checks

Loading verifies the file hash against the metadata and that scikit-learn, NumPy, pandas and SciPy have the same
major.minor version as when the model was saved. Pickled models are not portable across versions, and the failure
otherwise shows up as strange scores rather than an error. The fix is to retrain.

## Limits

* **joblib files are pickles.** Loading one executes code from the file. The hash catches corruption and a mismatch with
  `metadata.json`, but anyone able to replace the model can replace the metadata too. Only load models you trained.
  A signed registry or a non-pickle format (for example ONNX) is how a production system would remove this risk.
* There is no registry server, no promotion stages and no rollback beyond `--version`.
* The saved threshold is whatever `--threshold` was when training (default 0.5). It is not tuned here; the cutoff
  analysis lives in the reports and the final test evaluation.
* No monitoring: nothing checks whether new customers still resemble the training data.
* The metadata records no performance numbers. The cross-validated results are in `reports/`, and the test split is
  untouched until `python -m mlproject final`.

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from mlproject import __main__ as cli
from mlproject import baseline, preprocess
from mlproject.data import CATEGORIES, NUMERIC, TARGET, load
from mlproject.preprocess import Options, build_preprocessor, feature_names, transform_frame
from mlproject.split import split

COLUMNS = [*NUMERIC, *CATEGORIES]


@pytest.fixture(scope="module")
def train():
    return split(load())[0]


@pytest.fixture(scope="module")
def fitted(train):
    return build_preprocessor().fit(train[COLUMNS])


def test_output_is_numeric_complete_and_the_expected_width(train, fitted):
    out = transform_frame(fitted, train[COLUMNS])
    assert not out.isna().any().any()
    assert out.shape == (len(train), len(NUMERIC) + sum(len(v) for v in CATEGORIES.values()))
    assert len(set(out.columns)) == out.shape[1]  # every output column has a unique name
    assert "cat__Contract_Month-to-month" in out.columns and "num__tenure" in out.columns


def test_numeric_columns_are_standardised_on_the_fitting_rows(train, fitted):
    out = transform_frame(fitted, train[COLUMNS])
    numeric = out[[f"num__{c}" for c in NUMERIC]]
    assert np.allclose(numeric.mean(), 0, atol=1e-9)
    assert np.allclose(numeric.std(ddof=0), 1, atol=1e-9)


def test_statistics_come_from_the_fitting_rows_only(train):
    fit_rows, other = train.iloc[:3000], train.iloc[3000:]
    prep = build_preprocessor().fit(fit_rows[COLUMNS])
    num = prep.named_transformers_["num"]
    assert num.named_steps["impute"].statistics_[NUMERIC.index("TotalCharges")] == pytest.approx(fit_rows["TotalCharges"].median())
    assert num.named_steps["scale"].mean_[NUMERIC.index("tenure")] == pytest.approx(fit_rows["tenure"].mean())
    before = num.named_steps["scale"].mean_.copy()
    prep.transform(other[COLUMNS])
    assert np.array_equal(num.named_steps["scale"].mean_, before)  # transforming new rows learns nothing
    refit_on_all = build_preprocessor().fit(train[COLUMNS]).named_transformers_["num"].named_steps["scale"].mean_
    assert not np.allclose(before, refit_on_all)  # fitting on more rows WOULD change what the model sees: that is the leak


def test_a_row_is_transformed_the_same_alone_or_in_a_batch(train, fitted):
    batch = fitted.transform(train[COLUMNS].iloc[:200])
    for i in (0, 57, 199):
        assert np.allclose(fitted.transform(train[COLUMNS].iloc[[i]]), batch[[i]])


def test_missing_total_charges_gets_the_training_median(train, fitted):
    row = train[COLUMNS].iloc[[0]].assign(TotalCharges=np.nan)
    scale = fitted.named_transformers_["num"].named_steps["scale"]
    j = NUMERIC.index("TotalCharges")
    expected = (train["TotalCharges"].median() - scale.mean_[j]) / scale.scale_[j]
    assert transform_frame(fitted, row)["num__TotalCharges"].iloc[0] == pytest.approx(expected)


def test_an_unseen_category_becomes_zeros_not_an_error(train, fitted):
    row = train[COLUMNS].iloc[[0]].assign(Contract="Lifetime")
    out = transform_frame(fitted, row)
    assert out.filter(like="cat__Contract_").to_numpy().sum() == 0
    assert out.filter(like="cat__gender_").to_numpy().sum() == 1  # the other columns are unaffected


def test_each_categorical_column_is_one_hot_per_row(train, fitted):
    out = transform_frame(fitted, train[COLUMNS])
    for col, levels in CATEGORIES.items():
        block = out.filter(like=f"cat__{col}_")
        assert block.shape[1] == len(levels) and (block.sum(axis=1) == 1).all()


def test_unfitted_clone_carries_no_learned_state(train, fitted):
    fresh = clone(fitted)
    with pytest.raises(NotFittedError):
        fresh.transform(train[COLUMNS].head(3))


def test_no_scaling_leaves_raw_units(train):
    out = transform_frame(build_preprocessor(options=Options(scale=False)).fit(train[COLUMNS]), train[COLUMNS])
    assert out["num__tenure"].equals(train["tenure"].astype(float))


def test_log_option_is_log1p_then_standardise(train):
    out = transform_frame(build_preprocessor(options=Options(log_total_charges=True)).fit(train[COLUMNS]), train[COLUMNS])
    col = np.log1p(train["TotalCharges"].fillna(train["TotalCharges"].median()))
    assert np.allclose(out["log__TotalCharges"], (col - col.mean()) / col.std(ddof=0))
    assert "num__TotalCharges" not in out.columns and "num__tenure" in out.columns


def test_missing_indicator_marks_exactly_the_imputed_rows(train):
    out = transform_frame(build_preprocessor(options=Options(missing_indicator=True)).fit(train[COLUMNS]), train[COLUMNS])
    flag = out.filter(like="missingindicator")
    assert flag.shape[1] == 1 and int(flag.sum().iloc[0]) == int(train["TotalCharges"].isna().sum())


def test_tenure_bins_add_a_one_hot_block(train):
    out = transform_frame(build_preprocessor(options=Options(tenure_bins=6)).fit(train[COLUMNS]), train[COLUMNS])
    block = out.filter(like="tenure_bins__")
    assert block.shape[1] == 6 and (block.sum(axis=1) == 1).all()


def test_dropping_a_column_removes_it_from_the_output(train):
    prep = build_preprocessor(numeric=[c for c in NUMERIC if c != "tenure"]).fit(train[COLUMNS])
    assert "num__tenure" not in feature_names(prep)


def test_preprocessing_inside_a_pipeline_matches_hand_written_per_fold_fitting(train):
    """The Pipeline must give exactly what fitting the preprocessor on each training fold by hand gives."""
    X, y = train[COLUMNS], train[TARGET]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
    via_pipeline = cross_val_score(
        make_pipeline(build_preprocessor(), LogisticRegression(max_iter=2000)), X, y, cv=folds, scoring="average_precision"
    )
    by_hand = []
    for fit_idx, val_idx in folds:
        prep = build_preprocessor().fit(X.iloc[fit_idx])
        model = LogisticRegression(max_iter=2000).fit(prep.transform(X.iloc[fit_idx]), y.iloc[fit_idx])
        by_hand.append(average_precision_score(y.iloc[val_idx], model.predict_proba(prep.transform(X.iloc[val_idx]))[:, 1]))
    assert np.allclose(via_pipeline, by_hand)


@pytest.fixture(scope="module")
def comparison(train) -> pd.DataFrame:
    return baseline.compare_variants(train)


def test_reference_recipe_reproduces_the_baseline_logistic_score(train, comparison):
    ref = baseline.summarize(baseline.cross_validate(train)).loc["logistic_reference", "pr_auc_mean"]
    assert comparison.loc["reference", "pr_auc_mean"] == pytest.approx(ref)


def test_variants_are_compared_on_identical_folds(comparison):
    assert list(comparison.index) == list(preprocess.VARIANTS)
    assert comparison.loc["reference", "delta_mean"] == 0 and comparison.loc["reference", "folds_better"] == 0


def test_scaling_does_not_change_the_score_but_changes_convergence(comparison):
    assert abs(comparison.loc["no_scaling", "delta_mean"]) < 0.002
    assert bool(comparison.loc["reference", "converged"]) and not bool(comparison.loc["no_scaling", "converged"])


def test_log_total_charges_helps_consistently_and_is_the_recommended_recipe(comparison):
    assert preprocess.RECOMMENDED == "log_total_charges" and preprocess.RECOMMENDED in preprocess.VARIANTS
    row = comparison.loc["log_total_charges"]
    assert row["delta_mean"] > 0.005 and row["folds_better"] >= 4
    assert comparison["pr_auc_mean"].idxmax() == "log_total_charges"


def test_test_split_is_untouched_by_the_preprocessing_report():
    text = cli.PREPROCESS_MD.read_text(encoding="utf-8")
    assert "has not been used" in text and "5,634" in text

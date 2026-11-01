import pandas as pd
import pytest

from mlproject import data
from mlproject.data import DataError, clean, load, load_raw, validate


@pytest.fixture(scope="module")
def df():
    return load()


def test_shape_and_target_rate(df):
    assert df.shape == (7043, 21)
    assert df[data.TARGET].sum() == 1869
    assert df[data.TARGET].mean() == pytest.approx(0.2654, abs=1e-4)


def test_raw_file_is_untouched_text():
    raw = load_raw()
    assert all(pd.api.types.is_string_dtype(raw[c]) for c in raw.columns)
    assert (raw["TotalCharges"].str.strip() == "").sum() == 11  # blank strings, not missing values


def test_blank_total_charges_become_nan_for_new_customers_only(df):
    blank = df[df["TotalCharges"].isna()]
    assert len(blank) == 11
    assert (blank["tenure"] == 0).all() and (blank[data.TARGET] == 0).all()
    assert df["TotalCharges"].dtype == float


def test_clean_does_not_fill_or_drop_anything(df):
    assert len(df) == len(load_raw())
    assert df.drop(columns=["TotalCharges"]).notna().all().all()


def test_loaded_data_passes_validation_and_ids_are_unique(df):
    validate(df)
    assert df[data.ID_COLUMN].is_unique


def test_feature_list_matches_the_columns(df):
    assert len(data.FEATURES) == 19
    assert set(data.FEATURES) | {data.ID_COLUMN, data.TARGET} == set(df.columns)


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda d: d.drop(columns=["Contract"]), "missing columns"),
        (lambda d: d.assign(customerID=d["customerID"].where(d.index != 1, d["customerID"].iloc[0])), "unique"),
        (lambda d: d.assign(Contract=d["Contract"].where(d.index != 0, "Weekly")), "unexpected values"),
        (lambda d: d.assign(gender=d["gender"].where(d.index != 0, None)), "missing values"),
        (lambda d: d.assign(tenure=d["tenure"].where(d.index != 0, -1)), "negative"),
        (lambda d: d.assign(MonthlyCharges=d["MonthlyCharges"].where(d.index != 0, 0.0)), "positive"),
        (lambda d: d.assign(TotalCharges=d["TotalCharges"].where(d.index != 0, -5.0)), "negative"),
        (lambda d: d.assign(SeniorCitizen=d["SeniorCitizen"].where(d.index != 0, 2)), "0 or 1"),
        (lambda d: d.assign(churned=d["churned"].where(d.index != 0, 3)), "0 or 1"),
    ],
)
def test_validate_rejects_corrupted_data(df, mutate, message):
    with pytest.raises(DataError, match=message):
        validate(mutate(df.copy()))


def test_clean_rejects_unknown_churn_label():
    raw = load_raw().head(3).copy()
    raw.loc[0, "Churn"] = "Maybe"
    with pytest.raises(DataError):
        clean(raw)


def test_clean_handles_whitespace_total_charges():
    raw = load_raw().head(2).copy()
    raw.loc[0, "TotalCharges"] = "  "
    out = clean(raw)
    assert pd.isna(out.loc[0, "TotalCharges"]) and not pd.isna(out.loc[1, "TotalCharges"])

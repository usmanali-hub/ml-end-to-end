"""One preprocessing recipe for every model, fitted on training rows only.

``build_preprocessor`` returns an unfitted scikit-learn ``ColumnTransformer``. Put it first in a
``Pipeline`` and every statistic it learns (medians, means, category lists, bin edges) comes from the rows the
pipeline is fitted on, so cross-validation and the final test cannot leak information backwards.

The default recipe:

* numeric columns   median imputation (only ``TotalCharges`` has gaps) then standardisation
* categorical columns  one-hot encoding; a category never seen in training becomes all zeros instead of an error

Options exist so the choices can be *measured* (see ``compare_variants``) rather than assumed.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OneHotEncoder, StandardScaler

from .data import CATEGORIES, NUMERIC

LOG_COLUMN = "TotalCharges"  # the one strongly right-skewed money column


@dataclass(frozen=True)
class Options:
    """Switches for the preprocessing recipe. The defaults are the reference recipe."""

    scale: bool = True
    missing_indicator: bool = False  # add a 0/1 column marking which rows were imputed
    log_total_charges: bool = False  # log(1 + x) on TotalCharges before scaling
    tenure_bins: int = 0  # also one-hot encode tenure in this many quantile bins (0 = off)


DEFAULT_OPTIONS = Options()


def _numeric_pipeline(scale: bool, log: bool) -> Pipeline:
    steps: list[tuple[str, BaseEstimator]] = [("impute", SimpleImputer(strategy="median"))]
    if log:
        steps.append(("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def build_preprocessor(
    numeric: Sequence[str] = tuple(NUMERIC),
    categorical: Sequence[str] = tuple(CATEGORIES),
    options: Options = DEFAULT_OPTIONS,
) -> ColumnTransformer:
    """An unfitted transformer that turns the raw feature columns into a numeric matrix with no missing values."""
    plain = [c for c in numeric if not (options.log_total_charges and c == LOG_COLUMN)]
    logged = [c for c in numeric if options.log_total_charges and c == LOG_COLUMN]
    parts: list[tuple[str, BaseEstimator, list[str]]] = []
    if plain:
        parts.append(("num", _numeric_pipeline(options.scale, False), plain))
    if logged:
        parts.append(("log", _numeric_pipeline(options.scale, True), logged))
    if options.missing_indicator and numeric:
        # a separate branch, so the 0/1 flag is not standardised along with the numbers
        parts.append(("missing", MissingIndicator(features="missing-only"), list(numeric)))
    if options.tenure_bins and "tenure" in numeric:
        bins = KBinsDiscretizer(n_bins=options.tenure_bins, encode="onehot-dense", strategy="quantile", subsample=None)
        parts.append(("tenure_bins", bins, ["tenure"]))
    if categorical:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical)))
    return ColumnTransformer(parts)


def feature_names(prep: ColumnTransformer) -> list[str]:
    """Names of the output columns of a *fitted* preprocessor."""
    return [str(n) for n in prep.get_feature_names_out()]


def transform_frame(prep: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    """Apply a fitted preprocessor and keep the column names and row index."""
    return pd.DataFrame(np.asarray(prep.transform(df)), columns=feature_names(prep), index=df.index)


VARIANTS: dict[str, Options] = {
    "reference": DEFAULT_OPTIONS,
    "no_scaling": Options(scale=False),
    "missing_indicator": Options(missing_indicator=True),
    "log_total_charges": Options(log_total_charges=True),
    "tenure_bins_6": Options(tenure_bins=6),
}

RECOMMENDED = "log_total_charges"  # the recipe later models start from, chosen by the comparison in reports/preprocessing.md

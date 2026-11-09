"""Where the chosen models go wrong, and which inputs they rely on.

Everything here uses the training split only (the test split has not been used) and out-of-fold predictions: every
customer is scored by a model that never saw them, with the same 5 folds as the rest of the project.

Error analysis cuts the out-of-fold errors by customer segment. Each segment's *calibration gap* (observed churn rate
minus the mean predicted probability) is compared with its sampling noise. About fifty segments are examined, so a
few will look unusual by chance alone; the report says how many to expect.

Importance is measured two ways for each model, per raw input column (a one-hot group is never split up):

* *permutation*: shuffle one column in the held-out fold and see how far PR-AUC falls (the model is not refitted);
* *drop-column*: refit without the column and see how far held-out PR-AUC falls.

They disagree when columns carry overlapping information, which is the point of measuring both.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from . import boosting, models
from .baseline import SEED
from .data import CATEGORIES, NUMERIC, TARGET
from .evaluation import best_f1_cutoff, confusion, sweep
from .preprocess import build_preprocessor
from .trees import COLUMNS, TREE_OPTIONS, folds_of

REPEATS = 10
MIN_SEGMENT = 60  # smaller segments are too noisy to say anything about
Z = 2.0  # a calibration gap beyond this many standard errors is flagged

Builder = Callable[[tuple[str, ...]], Pipeline]  # columns to leave out -> unfitted pipeline


def logistic_builder(spec: models.Spec) -> Builder:
    return lambda drop: models.build(models.Spec(name=spec.name, C=spec.C, class_weight=spec.class_weight, drop=drop))


def boosted_builder(spec: boosting.BoostSpec) -> Builder:
    def build(drop: tuple[str, ...]) -> Pipeline:
        prep = build_preprocessor([c for c in NUMERIC if c not in drop], [c for c in CATEGORIES if c not in drop], TREE_OPTIONS)
        model: LGBMClassifier = boosting.booster(spec.leaves, spec.rate, spec.rounds)
        return Pipeline([("prep", prep), ("model", model)])

    return build


def _pr(pipe: Pipeline, df: pd.DataFrame) -> float:
    return float(average_precision_score(df[TARGET], np.asarray(pipe.predict_proba(df[COLUMNS]))[:, 1]))


def importance(train: pd.DataFrame, build: Builder, repeats: int = REPEATS, columns: list[str] | None = None) -> pd.DataFrame:
    """Per fold and column: PR-AUC fall from permuting the column, and from refitting without it."""
    cols = columns if columns is not None else COLUMNS
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float | int | str]] = []
    for fold, (fit_idx, val_idx) in enumerate(folds_of(train)):
        fit_df, val_df = train.iloc[fit_idx], train.iloc[val_idx]
        full = build(()).fit(fit_df[COLUMNS], fit_df[TARGET])
        base = _pr(full, val_df)
        for col in cols:
            drops = []
            for _ in range(repeats):
                shuffled = val_df.copy()
                shuffled[col] = rng.permutation(shuffled[col].to_numpy())
                drops.append(base - _pr(full, shuffled))
            refit = build((col,)).fit(fit_df[COLUMNS], fit_df[TARGET])
            rows.append({"fold": fold, "column": col, "permutation": float(np.mean(drops)), "drop_column": base - _pr(refit, val_df)})
    return pd.DataFrame(rows)


def summarize_importance(table: pd.DataFrame) -> pd.DataFrame:
    g = table.groupby("column")
    out = pd.DataFrame(
        {
            "permutation": g["permutation"].mean(),
            "permutation_sd": g["permutation"].std(ddof=1),
            "drop_column": g["drop_column"].mean(),
            "drop_column_sd": g["drop_column"].std(ddof=1),
        }
    )
    out["permutation_rank"] = out["permutation"].rank(ascending=False, method="first").astype(int)
    out["drop_rank"] = out["drop_column"].rank(ascending=False, method="first").astype(int)
    return out.sort_values("permutation", ascending=False)


def rank_agreement(a: pd.Series, b: pd.Series) -> float:
    """Spearman correlation between two importance columns over the same inputs."""
    return float(spearmanr(a.reindex(b.index), b).statistic)


def segments(train: pd.DataFrame) -> dict[str, pd.Series]:
    """Customer segmentations: each maps every customer to a label."""
    seg: dict[str, pd.Series] = {c: train[c].astype(str) for c in ("Contract", "InternetService", "PaymentMethod", "PaperlessBilling", "Partner", "Dependents", "SeniorCitizen", "TechSupport")}
    seg["tenure"] = pd.cut(train["tenure"], [-1, 12, 24, 48, 100], labels=["0-12 months", "13-24", "25-48", "49+"]).astype(str)
    seg["MonthlyCharges"] = pd.qcut(train["MonthlyCharges"], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]).astype(str)
    return seg


def segment_table(train: pd.DataFrame, s: np.ndarray, cutoff: float, min_size: int = MIN_SEGMENT) -> pd.DataFrame:
    """Per segment: size, churn rate, mean score, calibration gap and its z score, and precision and recall at ``cutoff``."""
    y = train[TARGET].to_numpy()
    rows = []
    for name, labels in segments(train).items():
        lab = labels.to_numpy()
        for value in sorted(set(lab)):
            m = lab == value
            if m.sum() < min_size:
                continue
            k = confusion(y[m], s[m], cutoff)
            rate, mean = float(y[m].mean()), float(s[m].mean())
            se = float(np.sqrt(max(rate * (1 - rate), 1e-9) / m.sum()))
            rows.append(
                {
                    "segment": name, "value": value, "n": int(m.sum()), "churn_rate": rate, "mean_score": mean, "gap": rate - mean,
                    "z": (rate - mean) / se, "precision": k.precision, "recall": k.recall, "errors": k.fp + k.fn, "error_rate": (k.fp + k.fn) / k.n,
                }
            )
    return pd.DataFrame(rows)


def expected_flags(n_segments: int, z: float = Z) -> float:
    """How many segments a perfectly calibrated model would flag at ``z`` by chance (normal approximation)."""
    from scipy.stats import norm

    return float(n_segments * 2 * norm.sf(z))


def error_profile(train: pd.DataFrame, s: np.ndarray, cutoff: float) -> pd.DataFrame:
    """Churners the model caught against those it missed, and loyal customers it flagged against those it left alone."""
    y = train[TARGET].to_numpy().astype(bool)
    flag = s >= cutoff
    groups = {"caught churners": y & flag, "missed churners": y & ~flag, "wrongly flagged": ~y & flag, "correctly left alone": ~y & ~flag}
    rows = []
    for name, m in groups.items():
        d = train[m]
        rows.append(
            {
                "group": name, "n": int(m.sum()), "mean_score": float(s[m].mean()), "tenure": float(d["tenure"].mean()),
                "monthly": float(d["MonthlyCharges"].mean()), "month_to_month": float((d["Contract"] == "Month-to-month").mean()),
                "fiber": float((d["InternetService"] == "Fiber optic").mean()),
            }
        )
    return pd.DataFrame(rows).set_index("group")


def error_overlap(y: np.ndarray, a: np.ndarray, b: np.ndarray, cutoff_a: float, cutoff_b: float) -> dict[str, float]:
    """Do two models make the same mistakes? Jaccard overlap of their misses and of their false alarms, plus score correlation."""
    yb = y.astype(bool)
    fa, fb = a >= cutoff_a, b >= cutoff_b
    miss_a, miss_b = yb & ~fa, yb & ~fb
    fp_a, fp_b = ~yb & fa, ~yb & fb

    def jac(u: np.ndarray, v: np.ndarray) -> float:
        return float((u & v).sum() / max((u | v).sum(), 1))

    return {"miss_overlap": jac(miss_a, miss_b), "false_alarm_overlap": jac(fp_a, fp_b), "score_corr": float(spearmanr(a, b).statistic), "misses_a": int(miss_a.sum()), "misses_b": int(miss_b.sum())}


@dataclass
class ErrorAnalysis:
    cutoff: float
    segments: pd.DataFrame
    profile: pd.DataFrame
    importance: dict[str, pd.DataFrame]  # model -> summarize_importance
    agreement: float  # Spearman between the two models' permutation importances
    overlap: dict[str, float]
    n: int


def analyse(train: pd.DataFrame, logistic: models.Spec, boost: boosting.BoostSpec, s_logistic: np.ndarray, s_boost: np.ndarray) -> ErrorAnalysis:
    y = train[TARGET].to_numpy()
    cutoff = best_f1_cutoff(sweep(y, s_logistic))
    cutoff_b = best_f1_cutoff(sweep(y, s_boost))
    imp = {"logistic": summarize_importance(importance(train, logistic_builder(logistic))), "lightgbm": summarize_importance(importance(train, boosted_builder(boost)))}
    return ErrorAnalysis(
        cutoff, segment_table(train, s_logistic, cutoff), error_profile(train, s_logistic, cutoff), imp,
        rank_agreement(imp["logistic"]["permutation"], imp["lightgbm"]["permutation"]), error_overlap(y, s_logistic, s_boost, cutoff, cutoff_b), len(train),
    )


def _importance_rows(a: ErrorAnalysis) -> list[str]:
    lg, gb = a.importance["logistic"].to_dict("index"), a.importance["lightgbm"].to_dict("index")
    order = (a.importance["logistic"]["permutation"] + a.importance["lightgbm"]["permutation"]).sort_values(ascending=False).index
    rows = []
    for col in order:
        l, g = lg[col], gb[col]
        rows.append(
            f"| {col} | {l['permutation']:.3f} ± {l['permutation_sd']:.3f} | {l['drop_column']:.3f} ± {l['drop_column_sd']:.3f} "
            f"| {g['permutation']:.3f} ± {g['permutation_sd']:.3f} | {g['drop_column']:.3f} ± {g['drop_column_sd']:.3f} |"
        )
    return rows


def _biggest_gap(imp: pd.DataFrame, top: int = 5) -> tuple[str, float, float]:
    """Among the model's top columns by permutation importance, the one where permutation exceeds drop-column importance by the most."""
    head = imp.head(top)
    col = str((head["permutation"] - head["drop_column"]).idxmax())
    row = imp.to_dict("index")[col]
    return col, float(row["permutation"]), float(row["drop_column"])


def markdown(a: ErrorAnalysis) -> str:
    seg = a.segments
    prof = a.profile.to_dict("index")
    lg, gb = a.importance["logistic"], a.importance["lightgbm"]
    lgd, gbd = lg.to_dict("index"), gb.to_dict("index")
    missed, wrong, caught, quiet = prof["missed churners"], prof["wrongly flagged"], prof["caught churners"], prof["correctly left alone"]
    errors_total = int(missed["n"] + wrong["n"])
    contract = seg[seg["segment"] == "Contract"].set_index("value").to_dict("index")
    m2m, one, two = contract["Month-to-month"], contract["One year"], contract["Two year"]
    sitting = round(float(one["churn_rate"] * one["n"] + two["churn_rate"] * two["n"]))
    flagged = seg[seg["z"].abs() > Z]
    expected = expected_flags(len(seg))
    lc, lp, ld = _biggest_gap(lg)
    gc, gp, gd = _biggest_gap(gb)
    noise_cols = [c for c in ("gender", "Partner", "DeviceProtection") if c in lg.index]
    lines = [
        "# Error analysis and feature importance",
        "",
        (
            f"Out-of-fold predictions on the training split ({a.n:,} customers); the test split has not been used. Errors are for `logistic_l2_chosen` at the cutoff "
            f"{a.cutoff:.2f} that maximises F1 on those same out-of-fold scores (chosen by looking at the training errors, so it is a description, not a held-out estimate)."
        ),
        "",
        "## 1. Who the model gets wrong",
        "",
        "| group | customers | mean score | mean tenure (months) | mean monthly charge | month-to-month | fibre optic |",
        "|---|---|---|---|---|---|---|",
        *[
            f"| {name} | {int(r['n']):,} | {r['mean_score']:.3f} | {r['tenure']:.1f} | {r['monthly']:.1f} | {r['month_to_month']:.1%} | {r['fiber']:.1%} |"
            for name, r in a.profile.iterrows()
        ],
        "",
        (
            f"{int(missed['n']):,} churners are missed and {int(wrong['n']):,} loyal customers are flagged. "
            f"The customers the model wrongly flags look like the churners it catches (tenure {wrong['tenure']:.0f} against {caught['tenure']:.0f} months, "
            f"{wrong['month_to_month']:.0%} against {caught['month_to_month']:.0%} on month-to-month contracts). "
            f"The churners it misses look like loyal customers (tenure {missed['tenure']:.0f} months, {missed['month_to_month']:.0%} month-to-month, "
            f"against {quiet['month_to_month']:.0%} for the customers it correctly leaves alone). "
            "In both directions the inputs describe an ordinary customer. Boosting and a forest, tuned earlier, did no better than this model, which is consistent with the table not containing what would separate them; it does not prove that."
        ),
        "",
        (
            f"Month-to-month customers account for {int(m2m['errors']):,} of the {errors_total:,} errors ({m2m['errors'] / errors_total:.0%}) and are {m2m['n'] / a.n:.0%} of customers. "
            f"Recall is {one['recall']:.2f} on one-year and {two['recall']:.2f} on two-year contracts, because almost no one on them is flagged; "
            f"{sitting:,} churners sit there."
        ),
        "",
        "### By segment",
        "",
        "`gap` is observed churn rate minus mean predicted probability; `z` divides it by its sampling standard error.",
        "",
        "| segment | value | customers | churn rate | mean score | gap | z | precision | recall | error rate |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *[
            f"| {r['segment']} | {r['value']} | {int(r['n']):,} | {r['churn_rate']:.3f} | {r['mean_score']:.3f} | {r['gap']:+.3f} | {r['z']:+.1f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['error_rate']:.3f} |"
            for _, r in seg.iterrows()
        ],
        "",
        (
            f"Segments with a calibration gap beyond {Z:.0f} standard errors: {len(flagged)} of {len(seg)}"
            + (f" ({', '.join(f'{r.segment} = {r.value}, z {r.z:+.1f}' for r in flagged.itertuples())})" if len(flagged) else "")
            + f"; with {len(seg)} segments examined and segments overlapping, about {expected:.1f} would be expected by chance even for a perfectly calibrated model. "
            "Treat a flag as a question to ask, not a finding."
        ),
        "",
        "## 2. Do the two models fail on the same customers?",
        "",
        (
            f"`lightgbm_chosen` (its own best-F1 cutoff) misses {a.overlap['misses_b']:.0f} churners against {a.overlap['misses_a']:.0f} for the logistic model; "
            f"the sets of missed churners overlap by {a.overlap['miss_overlap']:.0%} (Jaccard) and the sets of false alarms by {a.overlap['false_alarm_overlap']:.0%}. "
            f"Their scores have rank correlation {a.overlap['score_corr']:.3f}. Two models that make nearly the same mistakes will not fix each other, so averaging them is unlikely to add much."
        ),
        "",
        "## 3. Which inputs the models rely on",
        "",
        (
            f"Per raw input column (one-hot groups kept together), mean ± standard deviation over the 5 folds of the fall in held-out PR-AUC. *Permutation* shuffles the column in the held-out fold "
            f"({REPEATS} shuffles) without refitting; *drop-column* refits without it. Columns are ordered by the two models' permutation importance combined."
        ),
        "",
        "| column | logistic: permutation | logistic: drop-column | LightGBM: permutation | LightGBM: drop-column |",
        "|---|---|---|---|---|",
        *_importance_rows(a),
        "",
        (
            f"The two measures disagree sharply where inputs overlap. For the logistic model, permuting `{lc}` costs {lp:.3f} PR-AUC but refitting without it costs {ld:.3f}: "
            "as fitted, the model leans hard on it, but most of that reliance disappears when the model is refitted, because other columns can stand in for it. "
            f"For LightGBM, `{gc}` is {gp:.3f} by permutation and {gd:.3f} by drop-column. "
            "`TotalCharges` is roughly tenure times monthly charge, so the three columns carry overlapping information, and permuting one of them also creates customer profiles that cannot occur "
            "(a long tenure with a tiny total charge) which the model never saw. Permutation importance therefore overstates reliance on overlapping columns; drop-column importance understates it, because a substitute exists."
        ),
        "",
        (
            f"Rank agreement between the models' permutation importances is Spearman {a.agreement:.2f}. "
            f"`Contract` ranks {lgd['Contract']['permutation_rank']} for the logistic model and {gbd['Contract']['permutation_rank']} for LightGBM by permutation. "
            f"For both models the least important columns include {', '.join(f'`{c}`' for c in noise_cols)} "
            f"(logistic permutation importance no larger than {math.ceil(max(abs(lgd[c]['permutation']) for c in noise_cols) * 1000) / 1000:.3f} in absolute value)."
        ),
        "",
        "## What this does not show",
        "",
        "- Importance is reliance of a fitted model on a column, not a cause of churn. Customers choose their contract; offering a two-year contract to someone does not turn them into a two-year customer.",
        "- Differences smaller than the fold standard deviation (several columns in the middle of the table) are ties; the ranks of those columns should not be read.",
        "- Segment errors depend on the cutoff, which was chosen on the same scores. At a different cutoff the segments with the most errors would change.",
        "- Everything is cross-validation on the training split of one tutorial dataset. The segment findings are descriptive of these 5,634 customers.",
        "",
    ]
    return "\n".join(lines).replace("-0.000", "0.000")

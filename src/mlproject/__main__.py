"""Command line: ``python -m mlproject {check,card,baseline}`` (run with PYTHONPATH=src or after installing)."""

import argparse
import json
from pathlib import Path

import pandas as pd

from . import baseline, card, preprocess
from .data import CATEGORIES, NUMERIC, ROOT, load
from .metrics import PRIMARY
from .split import SEED, split

CARD_PATH = ROOT / "docs" / "data_card.md"
REPORT_MD = ROOT / "reports" / "baseline.md"
REPORT_JSON = ROOT / "reports" / "baseline.json"
TENURE_MD = ROOT / "reports" / "tenure_dependence.md"
PREPROCESS_MD = ROOT / "reports" / "preprocessing.md"


def baseline_markdown(summary: pd.DataFrame, n_train: int, prevalence: float) -> str:
    cols = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy", "recall", "precision", "f1"]
    lines = [
        "# Baseline results",
        "",
        f"5-fold stratified cross-validation on the training split ({n_train:,} rows, seed {SEED}), mean ± standard deviation over folds.",
        f"The churn rate is {prevalence:.1%}, so a model with no ranking ability scores a PR-AUC near {prevalence:.3f} and a ROC-AUC near 0.5.",
        "The test split has not been used.",
        "",
        "| baseline | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for name, row in summary.iterrows():
        cells = [f"{row[c + '_mean']:.3f} ± {row[c + '_std']:.3f}" for c in cols]
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    best = str(summary[f"{PRIMARY}_mean"].idxmax())
    best_value = float(summary[f"{PRIMARY}_mean"].max())
    lines += [
        "",
        f"**Primary metric: PR-AUC.** Best baseline: `{best}` at {best_value:.3f}. A model is only worth keeping if it clearly beats that, by more than the fold-to-fold spread above.",
        "",
    ]
    return "\n".join(lines)


def tenure_markdown(summary: pd.DataFrame, n_train: int) -> str:
    lines = [
        "# How much does the model lean on tenure?",
        "",
        f"The data card warns that `tenure` partly encodes the outcome (a churned customer's tenure is how long they stayed). This measures it: the same logistic regression, 5-fold cross-validation on the training split ({n_train:,} rows), with the time-related columns removed. The test split has not been used.",
        "",
        "| inputs | PR-AUC | ROC-AUC | recall | precision |",
        "|---|---|---|---|---|",
    ]
    for name, row in summary.iterrows():
        cells = [f"{row[c + '_mean']:.3f} ± {row[c + '_std']:.3f}" for c in ["pr_auc", "roc_auc", "recall", "precision"]]
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    means = summary["pr_auc_mean"].to_dict()
    full = float(means["all_features"])
    cut = float(means["without_tenure_and_total_charges"])
    lines += [
        "",
        (
            f"Removing `tenure` and `TotalCharges` changes PR-AUC by {cut - full:+.3f} ({(cut - full) / full:+.1%}). "
            f"The model keeps {cut / full:.1%} of its PR-AUC without them and still scores well above the best rule baseline, "
            "so it does not depend on the time columns alone. "
            "This only bounds the effect: `MonthlyCharges`, `Contract` and the add-on columns may also reflect how long a customer has stayed."
        ),
        "",
    ]
    return "\n".join(lines)


def preprocessing_markdown(train: pd.DataFrame, summary: pd.DataFrame) -> str:
    prep = preprocess.build_preprocessor().fit(train[[*NUMERIC, *CATEGORIES]])
    width = len(preprocess.feature_names(prep))
    lines = [
        "# Preprocessing",
        "",
        (
            "One recipe turns the 19 raw columns into a numeric matrix, and it is the first step of every model pipeline, so "
            "each statistic it learns comes from the rows that model is fitted on. "
            "The test split has not been used."
        ),
        "",
        "- **Numeric columns** (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`): median imputation, then standardisation. Only `TotalCharges` has gaps (8 blank values in the training split, all brand-new customers).",
        "- **Categorical columns** (15): one-hot encoding. A category that never appeared in training becomes all zeros rather than an error.",
        f"- **Output:** {width} columns, no missing values.",
        "",
        f"## Does the recipe matter? Five variants, same logistic regression, 5-fold CV on {len(train):,} training rows",
        "",
        (
            "`Δ vs reference` is the PR-AUC difference from the reference recipe on the *same* folds (mean ± standard deviation over folds), "
            "which is far less noisy than comparing two averages."
        ),
        "",
        "| recipe | PR-AUC | Δ vs reference | folds better | solver converged |",
        "|---|---|---|---|---|",
    ]
    for name, row in summary.iterrows():
        delta = "-" if name == "reference" else f"{row['delta_mean']:+.4f} ± {row['delta_std']:.4f}"
        better = "-" if name == "reference" else f"{int(row['folds_better'])} of 5"
        lines.append(f"| `{name}` | {row['pr_auc_mean']:.4f} ± {row['pr_auc_std']:.4f} | {delta} | {better} | {'yes' if row['converged'] else 'no'} |")
    rec_delta = float(summary["delta_mean"].to_dict()[preprocess.RECOMMENDED])
    rec_better = int(summary["folds_better"].to_dict()[preprocess.RECOMMENDED])
    ref_score = float(summary["pr_auc_mean"].to_dict()["reference"])
    lines += [
        "",
        "What the comparison shows:",
        "",
        f"- **Scaling does not change the score here but does change whether the solver finishes.** Without it the PR-AUC is the same to three decimals, yet logistic regression hits its {baseline.MAX_ITER:,}-iteration limit without converging. Scale-sensitive models (KNN, SVM, neural networks) are a different story.",
        "- **A missing-value indicator adds nothing**: only 8 training rows were imputed.",
        f"- **Log-transforming `TotalCharges` helps a little, consistently**: {rec_delta:+.4f} PR-AUC, better in {rec_better} of 5 folds. It is a small gain, about {rec_delta / ref_score:.1%} of the reference score.",
        "- **Binning `tenure` is inconclusive**: a small average gain with a spread as large as the gain, so it is not adopted.",
        "",
        f"Later models start from `{preprocess.RECOMMENDED}`. The baseline table keeps the plain reference recipe so the bar it sets does not move.",
        "",
    ]
    return "\n".join(lines)


def build_outputs() -> tuple[str, str, str, str, str]:
    df = load()
    train, _ = split(df)
    summary = baseline.summarize(baseline.cross_validate(train))
    md = baseline_markdown(summary, len(train), float(train["churned"].mean()))
    as_json = json.dumps({k: {m: round(float(v), 4) for m, v in row.items()} for k, row in summary.iterrows()}, indent=2)
    tenure_md = tenure_markdown(baseline.tenure_ablation(train), len(train))
    preprocessing_md = preprocessing_markdown(train, baseline.compare_variants(train))
    return card.render(df), md, as_json + "\n", tenure_md, preprocessing_md


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlproject")
    parser.add_argument("command", choices=["check", "card", "baseline", "all"])
    args = parser.parse_args(argv)
    if args.command == "check":
        df = load()
        print(f"data is valid: {len(df):,} rows, churn rate {df['churned'].mean():.1%}")
        return 0
    card_md, report_md, report_json, tenure_md, preprocessing_md = build_outputs()
    if args.command in ("card", "all"):
        CARD_PATH.write_text(card_md, encoding="utf-8", newline="\n")
        print(f"wrote {CARD_PATH.relative_to(ROOT)}")
    if args.command in ("baseline", "all"):
        Path(REPORT_MD).write_text(report_md, encoding="utf-8", newline="\n")
        Path(REPORT_JSON).write_text(report_json, encoding="utf-8", newline="\n")
        Path(TENURE_MD).write_text(tenure_md, encoding="utf-8", newline="\n")
        Path(PREPROCESS_MD).write_text(preprocessing_md, encoding="utf-8", newline="\n")
        print(f"wrote {REPORT_MD.relative_to(ROOT)}, {REPORT_JSON.relative_to(ROOT)}, {TENURE_MD.relative_to(ROOT)} and {PREPROCESS_MD.relative_to(ROOT)}")
        print(report_md)
        print(tenure_md)
        print(preprocessing_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line: ``python -m mlproject {check,card,baseline}`` (run with PYTHONPATH=src or after installing)."""

import argparse
import json
from pathlib import Path

import pandas as pd

from . import baseline, card
from .data import ROOT, load
from .metrics import PRIMARY
from .split import SEED, split

CARD_PATH = ROOT / "docs" / "data_card.md"
REPORT_MD = ROOT / "reports" / "baseline.md"
REPORT_JSON = ROOT / "reports" / "baseline.json"
TENURE_MD = ROOT / "reports" / "tenure_dependence.md"


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


def build_outputs() -> tuple[str, str, str, str]:
    df = load()
    train, _ = split(df)
    summary = baseline.summarize(baseline.cross_validate(train))
    md = baseline_markdown(summary, len(train), float(train["churned"].mean()))
    as_json = json.dumps({k: {m: round(float(v), 4) for m, v in row.items()} for k, row in summary.iterrows()}, indent=2)
    tenure_md = tenure_markdown(baseline.tenure_ablation(train), len(train))
    return card.render(df), md, as_json + "\n", tenure_md


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlproject")
    parser.add_argument("command", choices=["check", "card", "baseline", "all"])
    args = parser.parse_args(argv)
    if args.command == "check":
        df = load()
        print(f"data is valid: {len(df):,} rows, churn rate {df['churned'].mean():.1%}")
        return 0
    card_md, report_md, report_json, tenure_md = build_outputs()
    if args.command in ("card", "all"):
        CARD_PATH.write_text(card_md, encoding="utf-8", newline="\n")
        print(f"wrote {CARD_PATH.relative_to(ROOT)}")
    if args.command in ("baseline", "all"):
        Path(REPORT_MD).write_text(report_md, encoding="utf-8", newline="\n")
        Path(REPORT_JSON).write_text(report_json, encoding="utf-8", newline="\n")
        Path(TENURE_MD).write_text(tenure_md, encoding="utf-8", newline="\n")
        print(f"wrote {REPORT_MD.relative_to(ROOT)}, {REPORT_JSON.relative_to(ROOT)} and {TENURE_MD.relative_to(ROOT)}")
        print(report_md)
        print(tenure_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line: ``python -m mlproject {check,card,baseline}`` (run with PYTHONPATH=src or after installing)."""

import argparse
import json
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import (
    baseline,
    boosting,
    calibration,
    card,
    diskcache,
    errors,
    evaluation,
    final,
    models,
    neural,
    preprocess,
    registry,
    search,
    tracking,
    trees,
)
from .data import CATEGORIES, NUMERIC, ROOT, TARGET, load
from .metrics import PRIMARY
from .split import SEED, split
from .tracking import data_fingerprint, git_commit

CARD_PATH = ROOT / "docs" / "data_card.md"
REPORT_MD = ROOT / "reports" / "baseline.md"
REPORT_JSON = ROOT / "reports" / "baseline.json"
TENURE_MD = ROOT / "reports" / "tenure_dependence.md"
PREPROCESS_MD = ROOT / "reports" / "preprocessing.md"
LINEAR_MD = ROOT / "reports" / "linear_models.md"
TREES_MD = ROOT / "reports" / "trees_and_forests.md"
CONFUSION_MD = ROOT / "reports" / "confusion_matrix.md"
BOOSTING_MD = ROOT / "reports" / "gradient_boosting.md"
FINAL_MD = ROOT / "reports" / "final_test.md"
SEARCH_MD = ROOT / "reports" / "validation_and_search.md"
CALIBRATION_MD = ROOT / "reports" / "calibration_and_ranking.md"
NEURAL_MD = ROOT / "reports" / "neural_network.md"
ERRORS_MD = ROOT / "reports" / "error_analysis.md"
CALIBRATION_PNG = ROOT / "reports" / "figures" / "calibration.png"


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
        tied = abs(float(row["delta_mean"])) < 0.001  # below this, fold counts are solver noise and differ between library versions
        # a solver that did not converge has a path-dependent fold spread, so show it to 2 decimals only
        digits = 3 if row["converged"] else 2
        delta = "-" if name == "reference" else f"{row['delta_mean']:+.3f} ± {row['delta_std']:.{digits}f}"
        better = "-" if name == "reference" else ("tied" if tied else f"{int(row['folds_better'])} of 5")
        lines.append(
            f"| `{name}` | {row['pr_auc_mean']:.3f} ± {row['pr_auc_std']:.{digits}f} | {delta} | {better} | {'yes' if row['converged'] else 'no'} |"
        )
    rec_delta = float(summary["delta_mean"].to_dict()[preprocess.RECOMMENDED])
    rec_better = int(summary["folds_better"].to_dict()[preprocess.RECOMMENDED])
    ref_score = float(summary["pr_auc_mean"].to_dict()["reference"])
    lines += [
        "",
        "What the comparison shows:",
        "",
        f"- **Scaling does not change the score here but does change whether the solver finishes.** Without it the PR-AUC is the same to within 0.001, yet logistic regression hits its {baseline.MAX_ITER:,}-iteration limit without converging. Scale-sensitive models (KNN, SVM, neural networks) are a different story.",
        "- **A missing-value indicator adds nothing**: only 8 training rows were imputed.",
        f"- **Log-transforming `TotalCharges` helps a little, consistently**: {rec_delta:+.3f} PR-AUC, better in {rec_better} of 5 folds. It is a small gain, about {rec_delta / ref_score:.1%} of the reference score.",
        "- **Binning `tenure` is inconclusive**: a small average gain with a spread as large as the gain, so it is not adopted.",
        "",
        f"Later models start from `{preprocess.RECOMMENDED}`. The baseline table keeps the plain reference recipe so the bar it sets does not move.",
        "",
    ]
    return "\n".join(lines)


def _span(low: int, high: int) -> str:
    return str(low) if low == high else f"{low} to {high}"


def linear_markdown(train: pd.DataFrame, a: models.Analysis) -> str:
    fam = a.families
    chosen_row = a.sweep.loc[a.sweep["C"] == a.chosen_l1.C].iloc[0]
    nz = _span(int(chosen_row["l1_nonzero_min"]), int(chosen_row["l1_nonzero_max"]))
    lines = [
        "# Linear and logistic regression",
        "",
        (
            f"Every model starts from the `{preprocess.RECOMMENDED}` preprocessing recipe, so scaling, imputation and encoding are "
            f"fitted on training rows only. 5-fold stratified cross-validation on the training split ({len(train):,} rows), the same folds as the baselines; "
            "all choices below are made on those folds. The test split has not been used."
        ),
        "",
        "## 1. How strong should the penalty be?",
        "",
        "Smaller `C` means a stronger penalty. L2 shrinks every coefficient; L1 pushes some to exactly zero. PR-AUC is the mean over folds.",
        "",
        "| C | L2 PR-AUC | L1 PR-AUC | L1 non-zero coefficients (of 45) |",
        "|---|---|---|---|",
    ]
    for _, row in a.sweep.iterrows():
        lines.append(f"| {row['C']:g} | {row['l2_pr_auc']:.3f} | {row['l1_pr_auc']:.3f} | {_span(int(row['l1_nonzero_min']), int(row['l1_nonzero_max']))} |")
    lines += [
        "",
        (
            "- **Only a strong penalty hurts.** From C = 0.03 upward the PR-AUC is flat (0.670 to 0.674), so the choice is not delicate. "
            "At C = 0.001 L1 removes every coefficient and falls to the churn rate."
        ),
        (
            f"- **L1 matches L2 with half the inputs.** At C = {a.chosen_l1.C:g} it keeps {nz} of 45 coefficients and loses nothing measurable. "
            "The exact count varies with the solver's visiting order because the one-hot columns are exactly redundant, so L1 can give a duplicate column's weight to any of its copies. "
            f"The *model* does not change: across orderings no customer's predicted probability moves by more than {a.l1_prediction_spread:.1e}."
        ),
        (
            f"- **Chosen: L2 with C = {a.chosen.C:g}**, the strongest penalty within {models.TOLERANCE} PR-AUC of the best setting. "
            "Settings that tie are not worth telling apart, and the more heavily penalised one is the safer pick. "
            "The tolerance is a judgement call, so here is what other tolerances would pick:"
        ),
        "",
        "| tolerance | C picked | PR-AUC at that C |",
        "|---|---|---|",
        *[f"| {row['tolerance']:g} | {row['C']:g} | {row['pr_auc']:.3f} |" for _, row in a.tolerances.iterrows()],
        "",
        (
            f"Every tolerance lands on a PR-AUC within {a.tolerances['pr_auc'].max() - a.tolerances['pr_auc'].min():.3f} of the others, so the rule decides which of several "
            "equivalent models to report, not whether the model is good."
        ),
        "",
        "## 2. Five ways to model the same data (differences are from `logistic_l2_chosen` on the same folds; `tied` means within 0.002)",
        "",
        "| model | PR-AUC | Δ vs chosen | folds better | ROC-AUC | recall | precision | Brier |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, row in fam.iterrows():
        ref = name == "logistic_l2_chosen"
        tied = abs(float(row["delta_mean"])) < 0.002
        delta = "-" if ref else f"{row['delta_mean']:+.3f} ± {row['delta_std']:.3f}"
        better = "-" if ref else ("tied" if tied else f"{int(row['folds_better'])} of 5")
        lines.append(
            f"| `{name}` | {row['pr_auc_mean']:.3f} ± {row['pr_auc_std']:.3f} | {delta} | {better} | {row['roc_auc_mean']:.3f} | "
            f"{row['recall_mean']:.3f} | {row['precision_mean']:.3f} | {row['brier_mean']:.3f} |"
        )
    lines += [
        "",
        "Brier score is the mean squared error of the predicted probabilities (lower is better; a model that always says the churn rate scores 0.195).",
        "",
        (
            f"- **A linear regression on the 0/1 outcome ranks customers almost as well as logistic regression** (about {abs(float(fam['delta_mean'].to_dict()['linear_probability'])):.3f} PR-AUC behind), "
            f"but it is not a probability: it puts {a.linear_out_of_range:.1%} of customers below 0 or above 1."
        ),
        "- **Reweighting the classes does not improve the ranking.** PR-AUC is unchanged. It trades precision for recall by moving the 0.5 cutoff, and it makes the probabilities worse (higher Brier). Move the decision threshold instead of reweighting, when the goal is a different recall.",
        "",
        "## 3. What the chosen model leans on",
        "",
        "Largest coefficients as odds ratios. Numeric inputs are standardised, so their ratio is per one standard deviation; a one-hot input's ratio is for having that value versus not.",
        "",
        "| input | coefficient | odds ratio |",
        "|---|---|---|",
    ]
    for _, row in a.odds.iterrows():
        lines.append(f"| `{row['input']}` | {row['coefficient']:+.2f} | {row['odds_ratio']:.2f} |")
    lines += [
        "",
        (
            "Contract type, internet service and payment method point the way the exploration found. `TotalCharges` has a *negative* coefficient even though customers who stay longer have paid more: "
            f"it is almost exactly `tenure` times `MonthlyCharges` (correlation {a.charges_overlap:.3f}), so the three share credit and a single sign is not interpretable. "
            f"Removing `TotalCharges` lowers PR-AUC from {a.families['pr_auc_mean'].to_dict()['logistic_l2_chosen']:.3f} to {a.without_total_charges['pr_auc']:.3f}, so it does carry some information, and the `tenure` coefficient moves from {a.without_total_charges['tenure_with']:+.2f} to {a.without_total_charges['tenure_without']:+.2f} "
            f"and `MonthlyCharges` from {a.without_total_charges['monthly_with']:+.2f} to {a.without_total_charges['monthly_without']:+.2f}, which shows how much of each coefficient was an artefact of the overlap. "
            "Read these as a description of this fitted model, not as the effect of changing a customer's contract or bill."
        ),
        "",
    ]
    return "\n".join(lines)


@diskcache.disk_cached("variants")
def cached_variants() -> pd.DataFrame:
    """The preprocessing comparison, computed once per process (it is used by the report and by tests)."""
    return baseline.compare_variants(split(load())[0])


@diskcache.disk_cached("linear")
def cached_analysis() -> models.Analysis:
    """The linear-model analysis, computed once per process."""
    return models.analyse(split(load())[0])


def trees_markdown(train: pd.DataFrame, a: trees.TreeAnalysis, rule_pr_auc: float, linear: models.Analysis) -> str:
    fam = a.families
    sweep = a.sweep.set_index("depth")
    best_depth = str(a.sweep["depth"].iloc[int(a.sweep["cv_pr_auc"].to_numpy().argmax())])
    unlimited = sweep.loc["none"]
    noise = a.importance.loc["random_noise"].to_dict()
    real = a.importance.drop(index="random_noise")
    outranked = int((real["impurity"] < noise["impurity"]).sum())
    delta = fam["delta_mean"].to_dict()
    forest_gain = float(fam["pr_auc_mean"].to_dict()["random_forest_chosen"] - fam["pr_auc_mean"].to_dict()["decision_tree_chosen"])
    spec = a.chosen_forest
    lines = [
        "# Decision trees and random forests",
        "",
        (
            "Trees are invariant to scaling, so the inputs are only imputed and one-hot encoded (fitted on training rows only). "
            f"5-fold stratified cross-validation on the training split ({len(train):,} rows), the same folds as the baselines and the linear models. "
            "The test split has not been used."
        ),
        "",
        "## 1. A single tree: depth is the whole story",
        "",
        "`fit` is the PR-AUC on the rows the tree was grown from; `validation` is on held-out rows. The gap between them is overfitting.",
        "",
        "| max depth | leaves | fit PR-AUC | validation PR-AUC |",
        "|---|---|---|---|",
        *[
            f"| {row['depth']} | {int(row['leaves'])} | {row['fit_pr_auc']:.3f} | {row['cv_pr_auc']:.3f} ± {row['cv_std']:.3f} |"
            for _, row in a.sweep.iterrows()
        ],
        "",
        f"- Validation PR-AUC peaks at depth {best_depth} and then falls. Chosen: **depth {a.chosen_depth}**, the shallowest tree within {trees.TOLERANCE} of the best.",
        (
            f"- A tree allowed to grow without limit fits the training rows perfectly ({unlimited['fit_pr_auc']:.3f}) and scores {unlimited['cv_pr_auc']:.3f} on held-out rows, "
            f"**worse than the one-line month-to-month contract rule ({rule_pr_auc:.3f})**. A model that memorises is not better than a rule."
        ),
        "",
        "## 2. A forest: tuning the trees",
        "",
        f"{trees.GRID_TREES} trees per cell. `leaf` is the minimum rows in a leaf; `max_features` is the share of inputs each split may look at.",
        "",
        "| min leaf | max_features | fit PR-AUC | validation PR-AUC |",
        "|---|---|---|---|",
        *[f"| {int(r['min_samples_leaf'])} | {r['max_features']} | {r['fit_pr_auc']:.3f} | {r['cv_pr_auc']:.3f} |" for _, r in a.grid.iterrows()],
        "",
        f"- Leaf size matters most: with leaf = {int(a.grid['min_samples_leaf'].min())} the forest overfits a little; from leaf = 20 upward it is flat. How many inputs a split may see barely matters.",
        (
            f"- Chosen: **leaf = {spec.min_samples_leaf}, max_features = {spec.max_features}**: within {trees.TOLERANCE} of the best cell, preferring the most regularised setting."
        ),
        "",
        "More trees reduce variance with diminishing returns:",
        "",
        "| trees | validation PR-AUC |",
        "|---|---|",
        *[f"| {int(r['trees'])} | {r['cv_pr_auc']:.3f} |" for _, r in a.curve.iterrows()],
        "",
        "Beyond about 50 trees the score is flat, so 400 is more than needed and costs only time.",
        "",
        "## 3. Against the logistic model (differences are from `logistic_l2_chosen` on the same folds; `tied` means within 0.002)",
        "",
        "| model | fit PR-AUC | validation PR-AUC | Δ vs logistic | folds better | recall | precision | Brier |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, row in fam.iterrows():
        ref = name == "logistic_l2_chosen"
        tied = abs(float(row["delta_mean"])) < 0.002
        d = "-" if ref else f"{row['delta_mean']:+.3f} ± {row['delta_std']:.3f}"
        better = "-" if ref else ("tied" if tied else f"{int(row['folds_better'])} of 5")
        lines.append(
            f"| `{name}` | {row['fit_pr_auc_mean']:.3f} | {row['pr_auc_mean']:.3f} ± {row['pr_auc_std']:.3f} | {d} | {better} | {row['recall_mean']:.3f} | {row['precision_mean']:.3f} | {row['brier_mean']:.3f} |"
        )
    lines += [
        "",
        f"- **Averaging many trees repairs most of a single tree's weakness**: the tuned forest is {forest_gain:+.3f} PR-AUC above the tuned tree. Tuning matters too: the default forest (leaf = 1) overfits to a fit PR-AUC of {fam['fit_pr_auc_mean'].to_dict()['random_forest_default']:.3f} and scores {fam['pr_auc_mean'].to_dict()['random_forest_default']:.3f}.",
        (
            f"- **The forest does not beat logistic regression here** ({delta['random_forest_chosen']:+.3f} ± {fam['delta_std'].to_dict()['random_forest_chosen']:.3f}, better in "
            f"{int(fam['folds_better'].to_dict()['random_forest_chosen'])} of 5 folds). The churn signal is mostly additive (contract, tenure, internet service), which is what a linear model captures; "
            "the extra flexibility of trees has little left to find in 5,634 rows. A more complex model is not automatically a better one."
        ),
        "",
        "## 4. What the forest leans on, and why two importance measures disagree",
        "",
        "A pure-noise column (`random_noise`, drawn at random and unrelated to churn) was added as a yardstick. Impurity importance is computed while the trees grow; permutation importance is the validation PR-AUC lost when a column is shuffled. Fitted on 80% of the training split, evaluated on the other 20%.",
        "",
        "| column | permutation drop (± sd over 10 shuffles) | permutation rank | impurity importance | impurity rank |",
        "|---|---|---|---|---|",
        *[
            f"| `{col}` | {r['permutation_drop']:+.4f} ± {r['permutation_std']:.4f} | {int(r['permutation_rank'])} | {r['impurity']:.4f} | {int(r['impurity_rank'])} |"
            for col, r in a.importance.head(10).iterrows()
        ],
        f"| `random_noise` | {noise['permutation_drop']:+.4f} ± {noise['permutation_std']:.4f} | {int(noise['permutation_rank'])} | {noise['impurity']:.4f} | {int(noise['impurity_rank'])} |",
        "",
        (
            f"- **Impurity importance rewards noise.** The meaningless column gets impurity rank {int(noise['impurity_rank'])} of {len(a.importance)} and outranks {outranked} real columns, "
            f"because a continuous column offers many places to split and so many chances to fit chance patterns. Its permutation importance is {noise['permutation_drop']:+.4f}, about zero, as it should be."
        ),
        "- **The top of the ranking agrees with the exploration and the logistic model**: contract, tenure and internet service lead. Below the top few the permutation drops are smaller than their spread, so the order there should not be trusted.",
        "- `tenure`, `TotalCharges` and `MonthlyCharges` overlap (the linear-model report shows `TotalCharges` is almost `tenure` times `MonthlyCharges`), so shuffling one leaves the others to compensate and each looks less important than the group is.",
        "",
    ]
    del linear
    return "\n".join(lines)


def confusion_markdown(train: pd.DataFrame, oof: np.ndarray) -> str:
    y = train[TARGET].to_numpy()
    n, churned = len(y), int(y.sum())
    k = evaluation.confusion(y, oof, 0.5)
    table = evaluation.sweep(y, oof)
    lines = [
        "# Confusion matrices on the churn data",
        "",
        (
            f"Every training customer ({n:,}, of whom {churned:,} churned, {churned / n:.1%}) is scored by the chosen logistic model fitted on the *other* four folds, "
            "so no customer is scored by a model that saw them. The test split has not been used. A confusion matrix needs a cutoff: "
            "flag a customer when the predicted churn probability is at or above it."
        ),
        "",
        "## 1. Reading one matrix (cutoff 0.5)",
        "",
        "| | flagged as churn | not flagged |",
        "|---|---|---|",
        f"| **actually churned** ({churned:,}) | {k.tp:,} true positives | {k.fn:,} false negatives (missed) |",
        f"| **actually stayed** ({n - churned:,}) | {k.fp:,} false positives (wasted offers) | {k.tn:,} true negatives |",
        "",
        "| measure | arithmetic | value | in customers |",
        "|---|---|---|---|",
        f"| accuracy | ({k.tp:,} + {k.tn:,}) / {n:,} | {k.accuracy:.3f} | {k.tp + k.tn:,} of {n:,} customers classed correctly |",
        f"| precision | {k.tp:,} / ({k.tp:,} + {k.fp:,}) | {k.precision:.3f} | of {k.flagged:,} flagged customers, {k.tp:,} really would leave |",
        f"| recall | {k.tp:,} / ({k.tp:,} + {k.fn:,}) | {k.recall:.3f} | of {churned:,} who left, {k.tp:,} were caught |",
        f"| specificity | {k.tn:,} / ({k.tn:,} + {k.fp:,}) | {k.specificity:.3f} | of {n - churned:,} who stayed, {k.tn:,} were left alone |",
        f"| F1 | 2 x precision x recall / (precision + recall) | {k.f1:.3f} | one number balancing precision and recall |",
        "",
        "## 2. Why accuracy alone misleads",
        "",
        "| strategy | flagged | true positives | accuracy | precision | recall |",
        "|---|---|---|---|---|---|",
    ]
    for name, cut in (("flag nobody", 1.01), ("flag everybody", 0.0), ("the model at 0.5", 0.5)):
        c = evaluation.confusion(y, oof, cut)
        lines.append(f"| {name} | {c.flagged:,} | {c.tp:,} | {c.accuracy:.3f} | {c.precision:.3f} | {c.recall:.3f} |")
    lines += [
        "",
        "Flagging nobody is more than 73% accurate and finds no churner at all. With one customer in four leaving, accuracy rewards doing nothing; precision and recall say what the model actually does.",
        "",
        "## 3. The cutoff is a dial",
        "",
        "| cutoff | flagged | true positives | false positives | false negatives | precision | recall |",
        "|---|---|---|---|---|---|---|",
    ]
    for cut in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        r = table.loc[np.isclose(table["cutoff"], cut)].iloc[0]
        flagged = int(r["tp"] + r["fp"])
        lines.append(f"| {cut:g} | {flagged:,} | {int(r['tp']):,} | {int(r['fp']):,} | {int(r['fn']):,} | {r['precision']:.3f} | {r['recall']:.3f} |")
    best = evaluation.best_f1_cutoff(table)
    bf = evaluation.confusion(y, oof, best)
    lines += [
        "",
        "Lowering the cutoff catches more churners and wastes more offers; raising it does the opposite. Nothing about the model changes, only which mistake it prefers.",
        f"The cutoff with the best F1 is {best:g} (F1 {bf.f1:.3f}, recall {bf.recall:.3f}, precision {bf.precision:.3f}).",
        "",
        "## 4. Choosing a cutoff from what mistakes cost",
        "",
        (
            "The right cutoff depends on two business numbers that are not in this data: what an offer to a loyal customer costs, and what a lost customer is worth. "
            "The table is illustrative: it assumes a wasted offer costs 1 unit and shows what happens as a missed churner costs 1, 2, 5 or 10 units."
        ),
        "",
        "| a miss costs | best cutoff | flagged | false positives | false negatives | recall | precision | total cost | cost at cutoff 0.5 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for ratio in evaluation.COST_RATIOS:
        cut = evaluation.cost_optimal_cutoff(table, ratio)
        c = evaluation.confusion(y, oof, cut)
        base = evaluation.confusion(y, oof, 0.5)
        lines.append(
            f"| {ratio}x | {cut:g} | {c.flagged:,} | {c.fp:,} | {c.fn:,} | {c.recall:.3f} | {c.precision:.3f} | {c.fp + ratio * c.fn:,} | {base.fp + ratio * base.fn:,} |"
        )
    lines += [
        "",
        "The more a miss costs, the lower the cutoff should be, and the larger the saving over the default 0.5. If a miss costs about the same as a wasted offer, 0.5 is already close to right.",
        "",
        "## 5. With a limited retention budget",
        "",
        f"Contact only the highest-scoring customers. The base rate is {churned / n:.1%}, so a random list of the same size would be right about that often.",
        "",
        "| contact the top | customers | caught churners | hit rate (precision) | share of all churners caught | lift over random |",
        "|---|---|---|---|---|---|",
    ]
    for share in (0.05, 0.10, 0.20, 0.30):
        c = evaluation.top_share(y, oof, share)
        lines.append(f"| {share:.0%} | {c.flagged:,} | {c.tp:,} | {c.precision:.3f} | {c.recall:.3f} | {c.precision / (churned / n):.1f}x |")
    lines += [
        "",
        "Scores are out-of-fold on the training split. A final cutoff would be set here and then applied once to the untouched test split.",
        "",
    ]
    return "\n".join(lines)


def boosting_markdown(train: pd.DataFrame, a: boosting.BoostAnalysis, linear_pr: float) -> str:
    fam = a.families
    pr = fam["pr_auc_mean"].to_dict()
    delta = fam["delta_mean"].to_dict()
    spec = a.chosen
    curve = a.curve
    peak_rounds = int(curve["rounds"].iloc[int(curve["cv_pr_auc"].to_numpy().argmax())])
    last = curve.iloc[-1]
    nest = a.nested
    configs = nest.groupby(["leaves", "rate", "rounds"]).size()
    optimism = pr["lightgbm_chosen"] - pr["lightgbm_nested_tuning"]
    lines = [
        "# Gradient boosting",
        "",
        (
            "LightGBM grows shallow trees one after another, each fitted to the errors of the ensemble so far. "
            f"Fixed settings: at least {boosting.MIN_CHILD} rows per leaf, {boosting.ROW_SAMPLE:.0%} of rows and {boosting.COLUMN_SAMPLE:.0%} of columns per tree, "
            "one thread (so results reproduce). Inputs are imputed and one-hot encoded, fitted on training rows only. "
            f"5-fold stratified cross-validation on the training split ({len(train):,} rows), the same folds as every earlier model. The test split has not been used."
        ),
        "",
        "## 1. Rounds: boosting keeps overfitting if you let it",
        "",
        f"Tree size and learning rate as chosen below ({spec.leaves} leaves, rate {spec.rate}). `fit` is PR-AUC on the rows the booster was fitted to; `validation` is on held-out rows.",
        "",
        "| rounds | fit PR-AUC | validation PR-AUC |",
        "|---|---|---|",
        *[f"| {int(r['rounds'])} | {r['fit_pr_auc']:.3f} | {r['cv_pr_auc']:.3f} ± {r['cv_std']:.3f} |" for _, r in curve.iterrows()],
        "",
        (
            f"- Validation PR-AUC peaks around {peak_rounds} rounds and then declines slowly, while the fit score keeps rising "
            f"({last['fit_pr_auc']:.3f} at {int(last['rounds'])} rounds against {last['cv_pr_auc']:.3f} on held-out rows). "
            "The number of rounds is a regularisation setting, not something to maximise."
        ),
        "",
        "## 2. Tree size and learning rate",
        "",
        "For each setting, the round count with the best validation score:",
        "",
        "| leaves | learning rate | best rounds | fit PR-AUC | validation PR-AUC |",
        "|---|---|---|---|---|",
        *[
            f"| {int(r['leaves'])} | {r['rate']} | {int(r['rounds'])} | {r['fit_pr_auc']:.3f} | {r['cv_pr_auc']:.3f} ± {r['cv_std']:.3f} |"
            for _, r in a.best.iterrows()
        ],
        "",
        (
            f"- Smaller learning rates need more rounds for the same score, and bigger trees peak earlier and lower: the best setting is within {a.best['cv_pr_auc'].max() - a.best['cv_pr_auc'].min():.3f} "
            "of the worst, so none of these knobs is decisive."
        ),
        f"- Chosen: **{spec.leaves} leaves, rate {spec.rate}, {spec.rounds} rounds**: within {boosting.TOLERANCE} of the best cell, preferring the simplest (fewest leaves, then fewest rounds).",
        "",
        "## 3. Against the earlier models (differences from `logistic_l2_chosen` on the same folds; `tied` means within 0.002)",
        "",
        "| model | fit PR-AUC | validation PR-AUC | Δ vs logistic | folds better | recall | precision | Brier |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, row in fam.iterrows():
        ref = name == "logistic_l2_chosen"
        tied = abs(float(row["delta_mean"])) < 0.002
        d = "-" if ref else f"{row['delta_mean']:+.3f} ± {row['delta_std']:.3f}"
        better = "-" if ref else ("tied" if tied else f"{int(row['folds_better'])} of 5")
        extra = ["-"] * 4 if pd.isna(row["fit_pr_auc_mean"]) else [f"{row[c]:.3f}" for c in ["fit_pr_auc_mean", "recall_mean", "precision_mean", "brier_mean"]]
        lines.append(f"| `{name}` | {extra[0]} | {row['pr_auc_mean']:.3f} ± {row['pr_auc_std']:.3f} | {d} | {better} | {extra[1]} | {extra[2]} | {extra[3]} |")
    lines += [
        "",
        (
            f"- **Tuned boosting ties logistic regression** ({delta['lightgbm_chosen']:+.3f} ± {fam['delta_std'].to_dict()['lightgbm_chosen']:.3f}, better in "
            f"{int(fam['folds_better'].to_dict()['lightgbm_chosen'])} of 5 folds) and is {pr['lightgbm_chosen'] - pr['random_forest_chosen']:+.3f} above the tuned forest. "
            "Boosting is often the strongest learner on tabular data, but here it finds little the linear model missed: the signal is mostly additive and there are only 5,634 rows."
        ),
        (
            f"- **Defaults hurt**: LightGBM's own defaults (31 leaves, no sampling) score {pr['lightgbm_default']:.3f}, {pr['lightgbm_chosen'] - pr['lightgbm_default']:.3f} below the tuned version, "
            f"and {linear_pr - pr['lightgbm_default']:.3f} below logistic regression."
        ),
        (
            f"- Two other implementations with the same tree size, rate and rounds land in the same place: scikit-learn's boosting scores {pr['sklearn_hist_same_settings']:.3f} "
            f"({pr['sklearn_hist_same_settings'] - pr['lightgbm_chosen']:+.3f} from LightGBM) and XGBoost {pr['xgboost_same_settings']:.3f} "
            f"({pr['xgboost_same_settings'] - pr['lightgbm_chosen']:+.3f}). XGBoost's leaf limit works differently from LightGBM's, so its settings are the closest equivalent, not identical."
        ),
        "",
        "## 4. How optimistic is the tuned score?",
        "",
        "The tuned row above picked its settings on the same folds it is scored on. `lightgbm_nested_tuning` repeats the whole search (same grid, same rule) inside each outer fold, using an inner 3-fold split of that fold's fitting rows only, and scores the outer fold once.",
        "",
        "| outer fold | chosen leaves | rate | rounds | PR-AUC on the outer fold |",
        "|---|---|---|---|---|",
        *[f"| {int(r['fold'])} | {int(r['leaves'])} | {r['rate']} | {int(r['rounds'])} | {r['pr_auc']:.3f} |" for _, r in nest.iterrows()],
        "",
        (
            f"- The nested estimate is {pr['lightgbm_nested_tuning']:.3f}, {optimism:.3f} below the tuned row ({pr['lightgbm_chosen']:.3f}): that gap is the optimism from tuning on the reported folds."
        ),
        f"- The procedure chose {len(configs)} different settings across 5 folds, and the best and worst cells of the grid differ by only {a.best['cv_pr_auc'].max() - a.best['cv_pr_auc'].min():.3f}, so no single setting is clearly best.",
        "",
        "## What this does not show",
        "",
        "- All scores are cross-validation estimates on the training split; the test split is still untouched and will be used once, for the final model.",
        "- One dataset of 7,043 customers. On much larger or more interaction-heavy data boosting often pulls ahead of a linear model.",
        "- Grid is small and the other regularisation settings are fixed, so a different search could move the tuned score by a few thousandths.",
    ]
    at_edge = a.best[a.best["rounds"] == max(boosting.ROUNDS)]
    if len(at_edge):
        lines.append(
            f"- {len(at_edge)} of {len(a.best)} settings peak at the largest round count tried ({max(boosting.ROUNDS)}), so a longer run might score a little higher."
        )
    lines.append("")
    return "\n".join(lines)


def search_markdown(train: pd.DataFrame, a: search.SearchAnalysis) -> str:
    r = a.repeats
    o = a.optimism.set_index("size")
    last, first = o.loc[max(search.SIZES)], o.loc[min(search.SIZES)]
    wide = a.wide_optimism.set_index("size").loc[max(search.SIZES)]
    vs = r["vs_ref_mean"].to_dict()
    se = float(r["fold_se"].max())
    noise = float(r["repeat_sd"].max())
    lines = [
        "# Cross-validation noise and hyperparameter search",
        "",
        (
            f"Both analyses use only the training split ({len(train):,} rows); the test split has not been used. "
            "Question 1: how much of a gap between two models is just which rows landed in which fold? "
            "Question 2: does picking the best of many tried settings inflate the score we report for it?"
        ),
        "",
        f"## 1. Fold-assignment noise: the same comparison under {search.REPEATS} different 5-fold splits",
        "",
        "`repeat sd` is the spread of the mean score across the repeats (different fold assignments, same data). `fold SE` is the standard error of a single 5-fold mean, from the fold-to-fold spread. `vs logistic` is the model minus `logistic_l2_chosen` within each repeat.",
        "",
        "| model | mean PR-AUC | repeat sd | fold SE | vs logistic (mean) | vs logistic (range over repeats) | repeats ahead of logistic |",
        "|---|---|---|---|---|---|---|",
        *[
            f"| `{name}` | {row['mean']:.3f} | {row['repeat_sd']:.3f} | {row['fold_se']:.3f} | {'-' if name == 'logistic_l2_chosen' else format(row['vs_ref_mean'], '+.3f')} | "
            f"{'-' if name == 'logistic_l2_chosen' else format(row['vs_ref_range'], '.3f')} | {'-' if name == 'logistic_l2_chosen' else str(int(row['repeats_ahead'])) + ' of ' + str(search.REPEATS)} |"
            for name, row in r.iterrows()
        ],
        "",
        (
            f"- Reshuffling the folds moves a model's mean by about {noise:.3f} (the largest sd across models), but a single 5-fold mean has a standard error of about {se:.3f}. "
            "Reshuffling does not remove that: every repeat sees the same 5,634 customers, so it measures split noise, not sampling noise."
        ),
        (
            f"- Tuned boosting is {vs['lightgbm_chosen']:+.3f} from logistic regression, inside one standard error, so their order is not established by this data. "
            f"The tuned forest is {vs['random_forest_chosen']:+.3f} behind. LightGBM's untuned defaults are {vs['lightgbm_default']:+.3f}, well outside the noise measured here and behind logistic regression in every repeat."
        ),
        (
            "- Earlier selections used a tolerance of 0.001, tighter than the noise: where a simpler model was chosen, it was chosen as a tie-break on simplicity, not on evidence of being better."
        ),
        "",
        "## 2. Search optimism: random search of increasing size, nested inside each outer fold",
        "",
        (
            f"{search.N_CONFIGS} LightGBM settings were drawn at random (leaves, learning rate, rounds, minimum leaf rows, row and column sampling, L2 penalty). "
            f"For each outer fold, every setting was scored by an inner {search.INNER_FOLDS}-fold split of that fold's fitting rows; a search of size n tries the first n settings and takes the best. "
            "The winner is then scored once on the outer fold, which played no part in choosing it."
        ),
        "",
        "| settings tried | best inner score | winner on outer fold | inner minus outer (± sd over folds) | average setting on outer fold | best setting in hindsight | distinct winners |",
        "|---|---|---|---|---|---|---|",
        *[
            f"| {int(row['size'])} | {row['inner_best']:.3f} | {row['outer']:.3f} | {row['gap']:+.3f} ± {row['gap_std']:.3f} | {row['average_config']:.3f} | {row['hindsight_best']:.3f} | {int(row['distinct_winners'])} |"
            for _, row in o.reset_index().iterrows()
        ],
        "",
        (
            f"- **No optimism large enough to measure here.** Inner minus outer is {last['gap']:+.3f} for the full search, but its fold-to-fold sd is {last['gap_std']:.3f}, "
            f"so five outer folds cannot resolve anything smaller than roughly {last['gap_std'] / (5 ** 0.5):.3f}. The earlier nested check on the boosting grid found a similar small gap."
        ),
        (
            f"- **Searching helped a little.** The winner of {max(search.SIZES)} settings scores {last['outer']:.3f} on outer folds, "
            f"{last['outer'] - last['average_config']:+.3f} above an average setting ({last['average_config']:.3f}), "
            f"and {last['hindsight_best'] - last['outer']:.3f} below the best setting chosen with knowledge of the outer fold ({last['hindsight_best']:.3f}). "
            f"The very first setting drawn scores {first['outer']:.3f} on its own, so the extra tries added only {last['outer'] - first['outer']:+.3f} over that draw."
        ),
        (
            f"- The space is narrow (4 to 24 leaves, rates 0.01 to 0.2, at least 10 rows per leaf, an L2 penalty from 0.1 to 10): an average setting and the best one in hindsight differ by only {last['hindsight_best'] - last['average_config']:.3f}. "
            "Optimism is a bigger risk when settings differ a lot, so the same search was repeated in a much wider space below."
        ),
        "",
        "### The same search in a wide space",
        "",
        (
            "Leaves 8 to 64, learning rate 0.02 to 0.3, as few as 1 row per leaf, 30% to 100% of rows and columns per tree, and an L2 penalty down to 0.001: "
            "unregularised settings are allowed, so many are poor and the settings differ a lot."
        ),
        "",
        "| settings tried | best inner score | winner on outer fold | inner minus outer (± sd over folds) | average setting on outer fold | best setting in hindsight | distinct winners |",
        "|---|---|---|---|---|---|---|",
        *[
            f"| {int(row['size'])} | {row['inner_best']:.3f} | {row['outer']:.3f} | {row['gap']:+.3f} ± {row['gap_std']:.3f} | {row['average_config']:.3f} | {row['hindsight_best']:.3f} | {int(row['distinct_winners'])} |"
            for _, row in a.wide_optimism.reset_index().iterrows()
        ],
        "",
        (
            f"- **Selecting mattered much more here**: the winner of {max(search.SIZES)} settings scores {wide['outer']:.3f}, {wide['outer'] - wide['average_config']:+.3f} above an average setting ({wide['average_config']:.3f}), "
            f"and within {wide['hindsight_best'] - wide['outer']:.3f} of the best setting in hindsight. In the narrow space an average setting was only {last['hindsight_best'] - last['average_config']:.3f} from the best."
        ),
        (
            f"- **Inner minus outer is still about zero** ({wide['gap']:+.3f} ± {wide['gap_std']:.3f}), so no optimism shows up. This is not proof it is absent: inner scores come from models fitted on about 53% of the training rows "
            "(a third of the outer fitting rows held out) while the outer score comes from models fitted on 80%, so inner scores run low, and that bias works against the optimism being measured. "
            "A cleaner test would need more rows or repeated outer folds."
        ),
        "",
        "## What this does not show",
        "",
        "- One dataset of 7,043 customers; the standard errors above would shrink with more rows.",
        "- Repeated CV re-splits the same data, so it cannot estimate how a model would do on a different sample of customers.",
        "- Only boosting was searched, in two spaces, with one random draw of settings in each.",
        "- The search measure compares inner scores (models fitted on about 53% of the training rows) with outer scores (80%), so it cannot cleanly separate optimism from the effect of training-set size.",
        "- The test split is still untouched.",
        "",
    ]
    return "\n".join(lines)


def calibration_markdown(a: calibration.CalibrationAnalysis, y: np.ndarray) -> str:
    tab = a.table
    models_ = list(a.scores)
    raw = pd.DataFrame(tab.xs("raw", level="variant")).reindex(models_)
    chosen, balanced, default = "logistic_l2_chosen", "logistic_balanced", "lightgbm_default"
    roc_spread = float(raw["roc_auc"].max() - raw["roc_auc"].min())
    pr_spread = float(raw["pr_auc"].max() - raw["pr_auc"].min())
    rel_chosen = calibration.reliability(y, a.scores[chosen])
    rel_bal = calibration.reliability(y, a.scores[balanced])
    platt = pd.DataFrame(tab.xs("platt", level="variant")).reindex(models_)
    iso = pd.DataFrame(tab.xs("isotonic", level="variant")).reindex(models_)
    best_iso_gain = float((platt["ece"] - iso["ece"]).max())
    R, P, I = raw.to_dict("index"), platt.to_dict("index"), iso.to_dict("index")
    lines = [
        "# Ranking and calibration",
        "",
        (
            f"All scores are out-of-fold on the training split ({a.n:,} customers, churn rate {a.prevalence:.1%}): every customer is scored by a model that never saw them, on the same 5 folds as the other reports. "
            "Recalibration is also fitted out-of-fold, so recalibrated numbers are not graded on rows the calibrator saw. The test split has not been used."
        ),
        "",
        "![ROC, precision-recall and reliability curves](figures/calibration.png)",
        "",
        "## 1. Ranking: ROC against precision-recall",
        "",
        "`precision at 50% recall` is the best precision among cutoffs that still catch half the churners; `recall at 5% FPR` is the share of churners caught while wrongly flagging at most 5% of loyal customers.",
        "",
        "| model | ROC-AUC | PR-AUC | precision at 50% recall | precision at 80% recall | recall at 5% FPR | recall at 10% FPR |",
        "|---|---|---|---|---|---|---|",
        *[
            f"| `{n}` | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} | {r['precision_at_recall_50']:.3f} | {r['precision_at_recall_80']:.3f} | {r['recall_at_fpr_05']:.3f} | {r['recall_at_fpr_10']:.3f} |"
            for n, r in raw.iterrows()
        ],
        "",
        (
            f"- ROC-AUC separates these models over a range of {roc_spread:.3f}; PR-AUC over {pr_spread:.3f}, about {pr_spread / roc_spread:.1f} times as wide. "
            f"With only {a.prevalence:.1%} churners, a large number of loyal customers dilutes precision, which ROC (built from rates within each class) does not show. "
            f"A no-skill model scores 0.500 on ROC-AUC but only {a.prevalence:.3f} on PR-AUC, so PR-AUC is the more informative scale here."
        ),
        (
            f"- Read operationally, `{chosen}` catches {R[chosen]['recall_at_fpr_10']:.0%} of churners at a 10% false-alarm rate, and to catch 80% of churners about "
            f"{1 - R[chosen]['precision_at_recall_80']:.0%} of the flagged customers would be loyal ones."
        ),
        (
            f"- `{balanced}` has the same ROC-AUC and PR-AUC as `{chosen}` ({R[balanced]['roc_auc']:.3f} and {R[balanced]['pr_auc']:.3f}): class weights change the scores, not the order of customers. The next section shows what they do cost."
        ),
        "",
        "## 2. Calibration: do the probabilities mean what they say?",
        "",
        "`slope` and `intercept` come from regressing the outcome on the log-odds of the score: perfect is slope 1, intercept 0; slope below 1 means overconfident scores, a negative intercept means scores that are too high. `mean score` should equal the churn rate.",
        "",
        "| model | Brier | log loss | ECE | slope | intercept | mean score |",
        "|---|---|---|---|---|---|---|",
        *[f"| `{n}` | {r['brier']:.3f} | {r['log_loss']:.3f} | {r['ece']:.3f} | {r['slope']:.2f} | {r['intercept']:+.2f} | {r['mean_score']:.3f} |" for n, r in raw.iterrows()],
        "",
        f"Reliability of `{chosen}` and of `{balanced}`, in ten equal-count groups from lowest to highest predicted risk:",
        "",
        "| group | customers | `" + chosen + "` predicted | observed | `" + balanced + "` predicted | observed |",
        "|---|---|---|---|---|---|",
        *[
            f"| {int(a_['bin'])} | {int(a_['n'])} | {a_['predicted']:.3f} | {a_['observed']:.3f} | {b_['predicted']:.3f} | {b_['observed']:.3f} |"
            for (_, a_), (_, b_) in zip(rel_chosen.iterrows(), rel_bal.iterrows(), strict=True)
        ],
        "",
        (
            f"- **The chosen models are already well calibrated.** `{chosen}` has ECE {R[chosen]['ece']:.3f}, slope {R[chosen]['slope']:.2f}, and its mean score {R[chosen]['mean_score']:.3f} matches the churn rate {a.prevalence:.3f}. "
            "Regularised logistic regression is built to produce probabilities, and tuning for PR-AUC did not damage that."
        ),
        (
            f"- **Class weights wreck calibration while leaving the ranking alone.** `{balanced}` scores the same on ROC-AUC and PR-AUC but reports a mean churn probability of {R[balanced]['mean_score']:.3f} "
            f"for a population that churns at {a.prevalence:.3f} (ECE {R[balanced]['ece']:.3f}, Brier {R[balanced]['brier']:.3f} against {R[chosen]['brier']:.3f}). "
            "A score of 0.5 from that model does not mean a coin flip."
        ),
        (
            f"- **Untuned boosting is overconfident**: `{default}` has slope {R[default]['slope']:.2f} (scores more extreme than the outcomes justify), the lowest slope in the table, and a log loss of {R[default]['log_loss']:.3f} against {R[chosen]['log_loss']:.3f} for `{chosen}`."
        ),
        "",
        "## 3. Recalibration, fitted out-of-fold",
        "",
        "Platt scaling fits a logistic curve to the score's log-odds; isotonic regression fits a monotone step function. Each row was recalibrated by a calibrator fitted on the other four folds.",
        "",
        "| model | Brier raw | Platt | isotonic | ECE raw | Platt | isotonic | PR-AUC raw | Platt | isotonic |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *[
            f"| `{n}` | {R[n]['brier']:.3f} | {P[n]['brier']:.3f} | {I[n]['brier']:.3f} | {R[n]['ece']:.3f} | {P[n]['ece']:.3f} | {I[n]['ece']:.3f} | "
            f"{R[n]['pr_auc']:.3f} | {P[n]['pr_auc']:.3f} | {I[n]['pr_auc']:.3f} |"
            for n in models_
        ],
        "",
        (
            f"- **Platt scaling repairs the class-weighted model completely** (Brier {R[balanced]['brier']:.3f} to {P[balanced]['brier']:.3f}, ECE {R[balanced]['ece']:.3f} to {P[balanced]['ece']:.3f}) and leaves its ranking unchanged. "
            "So the weights cost nothing that a two-parameter correction cannot undo, provided someone applies it."
        ),
        (
            f"- For models that were already calibrated Platt scaling changes little (`{chosen}` ECE {R[chosen]['ece']:.3f} to {P[chosen]['ece']:.3f}); for `{default}` it helps modestly "
            f"(log loss {R[default]['log_loss']:.3f} to {P[default]['log_loss']:.3f})."
        ),
        (
            f"- **Isotonic regression lowers PR-AUC** (`{chosen}` {R[chosen]['pr_auc']:.3f} to {I[chosen]['pr_auc']:.3f}): a step function gives many customers identical scores, so their order is lost, and with {a.n:,} rows its ECE is at most {best_iso_gain:.3f} better than Platt's on any model. "
            "It needs more data than this to pay off."
        ),
        "",
        "## What this does not show",
        "",
        "- All numbers are cross-validation estimates on the training split; the test split is untouched. Calibration measured on 5,634 rows has noise: ECE values of 0.01 to 0.02 are within what ten bins of about 560 customers can produce by chance.",
        "- One dataset, with a roughly additive signal that suits logistic regression; a model class that is poorly calibrated here may be fine elsewhere.",
        "- Recalibrated scores come from a calibrator fitted on out-of-fold scores; in production the calibrator would be fitted on held-out data from the final model.",
        "",
    ]
    return "\n".join(lines)


@diskcache.disk_cached("oof")
def cached_oof() -> np.ndarray:
    """Out-of-fold churn scores from the chosen logistic model, computed once per process."""
    train = split(load())[0]
    spec = cached_analysis().chosen
    return evaluation.out_of_fold_scores(train, lambda: models.build(spec))


@diskcache.disk_cached("trees")
def cached_trees() -> trees.TreeAnalysis:
    """The tree and forest analysis, computed once per process."""
    train = split(load())[0]
    spec = cached_analysis().chosen
    return trees.analyse(train, lambda: models.build(spec))


@diskcache.disk_cached("boosting")
def cached_boosting() -> boosting.BoostAnalysis:
    """The gradient boosting analysis, computed once per process."""
    train = split(load())[0]
    spec = cached_analysis().chosen
    return boosting.analyse(train, lambda: models.build(spec), cached_trees().chosen_forest)


@diskcache.disk_cached("calibration")
def cached_calibration() -> calibration.CalibrationAnalysis:
    """Calibration and ranking analysis, computed once per process."""
    train = split(load())[0]
    spec = cached_analysis().chosen
    forest = cached_trees().chosen_forest
    boost = cached_boosting().chosen
    builders = {
        "logistic_l2_chosen": lambda: models.build(spec),
        "logistic_balanced": lambda: models.build(models.Spec(name="balanced", C=spec.C, class_weight="balanced")),
        "random_forest_chosen": lambda: trees.make_forest(trees.ForestSpec(forest.min_samples_leaf, forest.max_features, 400)),
        "lightgbm_chosen": lambda: boosting.make_boosted(boost),
        "lightgbm_default": boosting.make_default,
    }
    return calibration.analyse(train, builders)


@diskcache.disk_cached("errors")
def cached_errors() -> errors.ErrorAnalysis:
    """Error analysis and out-of-fold feature importance for the chosen logistic and boosting models."""
    train = split(load())[0]
    scores = cached_calibration().scores
    return errors.analyse(train, cached_analysis().chosen, cached_boosting().chosen, scores["logistic_l2_chosen"], scores["lightgbm_chosen"])


@diskcache.disk_cached("neural")
def cached_neural() -> neural.NeuralAnalysis:
    """The PyTorch MLP grid, comparison, seed spread and training curve."""
    train = split(load())[0]
    spec = cached_analysis().chosen
    return neural.analyse(train, lambda: models.build(spec), cached_boosting().chosen)


@diskcache.disk_cached("search")
def cached_search() -> search.SearchAnalysis:
    """The repeated-CV and search-optimism analysis, computed once per process."""
    train = split(load())[0]
    spec = cached_analysis().chosen
    forest = cached_trees().chosen_forest
    boost = cached_boosting().chosen
    builders = {
        "logistic_l2_chosen": lambda: models.build(spec),
        "random_forest_chosen": lambda: trees.make_forest(trees.ForestSpec(forest.min_samples_leaf, forest.max_features, search.FOREST_TREES)),
        "lightgbm_chosen": lambda: boosting.make_boosted(boost),
        "lightgbm_default": boosting.make_default,
    }
    return search.analyse(train, builders, "logistic_l2_chosen")


@cache
def build_outputs() -> tuple[str, str, str, str, str, str, str, str, str, str, str, str, str]:
    df = load()
    train, _ = split(df)
    summary = baseline.summarize(baseline.cross_validate(train))
    md = baseline_markdown(summary, len(train), float(train["churned"].mean()))
    as_json = json.dumps({k: {m: round(float(v), 4) for m, v in row.items()} for k, row in summary.iterrows()}, indent=2)
    tenure_md = tenure_markdown(baseline.tenure_ablation(train), len(train))
    preprocessing_md = preprocessing_markdown(train, cached_variants())
    linear_md = linear_markdown(train, cached_analysis())
    rule = float(summary["pr_auc_mean"].to_dict()["month_to_month"])
    trees_md = trees_markdown(train, cached_trees(), rule, cached_analysis())
    confusion_md = confusion_markdown(train, cached_oof())
    linear_pr = float(cached_trees().families["pr_auc_mean"].to_dict()["logistic_l2_chosen"])
    boosting_md = boosting_markdown(train, cached_boosting(), linear_pr)
    search_md = search_markdown(train, cached_search())
    calibration_md = calibration_markdown(cached_calibration(), train[TARGET].to_numpy())
    errors_md = errors.markdown(cached_errors())
    neural_md = neural.markdown(cached_neural())
    return card.render(df), md, as_json + "\n", tenure_md, preprocessing_md, linear_md, trees_md, confusion_md, boosting_md, search_md, calibration_md, errors_md, neural_md


def run_final(force: bool = False) -> int:
    """Score the untouched test split once, with models chosen by cross-validation. Not part of ``all``."""
    if FINAL_MD.exists() and not force:
        print(f"{FINAL_MD.name} already exists: the test split has been used once. Pass --force to overwrite.")
        return 1
    train, test = split(load())
    linear = cached_analysis()
    forest = cached_trees().chosen_forest
    boost = cached_boosting().chosen
    builders = {
        "logistic_l2_chosen": lambda: models.build(linear.chosen),
        "random_forest_chosen": lambda: trees.make_forest(trees.ForestSpec(forest.min_samples_leaf, forest.max_features, 400)),
        "lightgbm_chosen": lambda: boosting.make_boosted(boost),
    }
    fam = cached_boosting().families["pr_auc_mean"].to_dict()
    cv = {name: float(fam[name]) for name in builders}
    result = final.evaluate_final(train, test, builders, "logistic_l2_chosen", cv)
    text = final.markdown(result, len(train))
    final.write_once(FINAL_MD, text, force=force)
    print(text)
    return 0


def tracked_candidates() -> list[tracking.Candidate]:
    """The models chosen in the earlier reports, with the settings that define them, ready to be logged."""
    lin, forest, boost, mlp = cached_analysis().chosen, cached_trees().chosen_forest, cached_boosting().chosen, cached_neural().chosen
    big_forest = trees.ForestSpec(forest.min_samples_leaf, forest.max_features, 400)
    return [
        tracking.Candidate("logistic_l2_chosen", {"family": "logistic regression", "penalty": lin.penalty, "C": lin.C}, lambda: models.build(lin)),
        tracking.Candidate(
            "random_forest_chosen",
            {"family": "random forest", "min_samples_leaf": big_forest.min_samples_leaf, "max_features": str(big_forest.max_features), "n_estimators": big_forest.n_estimators},
            lambda: trees.make_forest(big_forest),
        ),
        tracking.Candidate(
            "lightgbm_chosen",
            {"family": "gradient boosting", "num_leaves": boost.leaves, "learning_rate": boost.rate, "n_estimators": boost.rounds},
            lambda: boosting.make_boosted(boost),
        ),
        tracking.Candidate(
            "mlp_chosen",
            {"family": "neural network", "hidden": "x".join(map(str, mlp.hidden)) or "none", "weight_decay": mlp.weight_decay, "dropout": neural.DROPOUT},
            lambda: neural.make_mlp(mlp),
        ),
    ]


def run_tracking(command: str, store: str | None) -> int:
    path = Path(store) if store else tracking.DEFAULT_STORE
    uri = tracking.store_uri(path)
    if command == "track":
        train = split(load())[0]
        ids = tracking.log_runs(train, tracked_candidates(), uri)
        print(f"logged {len(ids)} runs to {path}")
    table = tracking.comparison(uri)
    print(tracking.markdown_table(table) if not table.empty else "no runs logged yet: run `python -m mlproject track`")
    return 0


MODEL_NAME = "churn-logistic"


def run_train(threshold: float) -> int:
    """Fit the chosen logistic model on the training split only and save it with its metadata."""
    train, _ = split(load())
    pipe = models.build(models.Spec(name="logistic", C=0.03)).fit(train[models.COLUMNS], train[TARGET])
    path = registry.save(
        pipe, MODEL_NAME, train_rows=len(train), data_fingerprint=data_fingerprint(), commit=git_commit(), threshold=threshold
    )
    print(f"saved {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    return 0


def run_predict(model_name: str, version: str | None, source: str | None, output: str | None) -> int:
    if source is None:
        print("predict needs --input <customers.csv>")
        return 2
    try:
        model = registry.load_model(model_name, version)
        scored = registry.score_frame(model, pd.read_csv(source, dtype=str, keep_default_na=False))
    except (registry.ModelError, registry.InputError, OSError) as err:
        print(f"error: {err}")
        return 1
    text = scored.to_csv(index=False, lineterminator="\n")
    if output:
        Path(output).write_text(text, encoding="utf-8", newline="\n")
        print(f"scored {len(scored)} customers with model {model.version}; {int(scored['flagged'].sum())} flagged -> {output}")
    else:
        print(text, end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlproject")
    parser.add_argument("command", choices=["check", "card", "baseline", "all", "final", "track", "runs", "train", "predict"])
    parser.add_argument("--input", help="with `predict`: CSV of customers with the 19 feature columns")
    parser.add_argument("--output", help="with `predict`: write the scores here instead of printing them")
    parser.add_argument("--model", default=MODEL_NAME, help="with `predict`: saved model name")
    parser.add_argument("--version", help="with `predict`: saved model version (default: newest)")
    parser.add_argument("--threshold", type=float, default=0.5, help="with `train`: probability at or above which a customer is flagged")
    parser.add_argument("--store", help="with `track` or `runs`: path of the MLflow SQLite database (default .mlflow/mlflow.db)")
    parser.add_argument("--force", action="store_true", help="with `final`: overwrite an existing final test report")
    args = parser.parse_args(argv)
    if args.command == "train":
        return run_train(args.threshold)
    if args.command == "predict":
        return run_predict(args.model, args.version, args.input, args.output)
    if args.command == "final":
        return run_final(force=args.force)
    if args.command in ("track", "runs"):
        return run_tracking(args.command, args.store)
    if args.command == "check":
        df = load()
        print(f"data is valid: {len(df):,} rows, churn rate {df['churned'].mean():.1%}")
        return 0
    card_md, report_md, report_json, tenure_md, preprocessing_md, linear_md, trees_md, confusion_md, boosting_md, search_md, calibration_md, errors_md, neural_md = build_outputs()
    if args.command in ("card", "all"):
        CARD_PATH.write_text(card_md, encoding="utf-8", newline="\n")
        print(f"wrote {CARD_PATH.relative_to(ROOT)}")
    if args.command in ("baseline", "all"):
        Path(REPORT_MD).write_text(report_md, encoding="utf-8", newline="\n")
        Path(REPORT_JSON).write_text(report_json, encoding="utf-8", newline="\n")
        Path(TENURE_MD).write_text(tenure_md, encoding="utf-8", newline="\n")
        Path(PREPROCESS_MD).write_text(preprocessing_md, encoding="utf-8", newline="\n")
        Path(LINEAR_MD).write_text(linear_md, encoding="utf-8", newline="\n")
        Path(TREES_MD).write_text(trees_md, encoding="utf-8", newline="\n")
        Path(CONFUSION_MD).write_text(confusion_md, encoding="utf-8", newline="\n")
        Path(BOOSTING_MD).write_text(boosting_md, encoding="utf-8", newline="\n")
        Path(SEARCH_MD).write_text(search_md, encoding="utf-8", newline="\n")
        Path(CALIBRATION_MD).write_text(calibration_md, encoding="utf-8", newline="\n")
        Path(ERRORS_MD).write_text(errors_md, encoding="utf-8", newline="\n")
        Path(NEURAL_MD).write_text(neural_md, encoding="utf-8", newline="\n")
        calibration.figure(cached_calibration(), split(load())[0][TARGET].to_numpy(), CALIBRATION_PNG)
        print(f"wrote {REPORT_MD.relative_to(ROOT)}, {REPORT_JSON.relative_to(ROOT)}, {TENURE_MD.relative_to(ROOT)}, {PREPROCESS_MD.relative_to(ROOT)}, {LINEAR_MD.relative_to(ROOT)}, {TREES_MD.relative_to(ROOT)}, {CONFUSION_MD.relative_to(ROOT)}, {BOOSTING_MD.relative_to(ROOT)}, {SEARCH_MD.relative_to(ROOT)}, {CALIBRATION_MD.relative_to(ROOT)}, {ERRORS_MD.relative_to(ROOT)} and {NEURAL_MD.relative_to(ROOT)}")
        print(report_md)
        print(tenure_md)
        print(preprocessing_md)
        print(linear_md)
        print(trees_md)
        print(confusion_md)
        print(boosting_md)
        print(search_md)
        print(calibration_md)
        print(errors_md)
        print(neural_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line: ``python -m mlproject {check,card,baseline}`` (run with PYTHONPATH=src or after installing)."""

import argparse
import json
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import baseline, card, evaluation, models, preprocess, trees
from .data import CATEGORIES, NUMERIC, ROOT, TARGET, load
from .metrics import PRIMARY
from .split import SEED, split

CARD_PATH = ROOT / "docs" / "data_card.md"
REPORT_MD = ROOT / "reports" / "baseline.md"
REPORT_JSON = ROOT / "reports" / "baseline.json"
TENURE_MD = ROOT / "reports" / "tenure_dependence.md"
PREPROCESS_MD = ROOT / "reports" / "preprocessing.md"
LINEAR_MD = ROOT / "reports" / "linear_models.md"
TREES_MD = ROOT / "reports" / "trees_and_forests.md"
CONFUSION_MD = ROOT / "reports" / "confusion_matrix.md"


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


@cache
def cached_variants() -> pd.DataFrame:
    """The preprocessing comparison, computed once per process (it is used by the report and by tests)."""
    return baseline.compare_variants(split(load())[0])


@cache
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


@cache
def cached_oof() -> np.ndarray:
    """Out-of-fold churn scores from the chosen logistic model, computed once per process."""
    train = split(load())[0]
    spec = cached_analysis().chosen
    return evaluation.out_of_fold_scores(train, lambda: models.build(spec))


@cache
def cached_trees() -> trees.TreeAnalysis:
    """The tree and forest analysis, computed once per process."""
    train = split(load())[0]
    spec = cached_analysis().chosen
    return trees.analyse(train, lambda: models.build(spec))


@cache
def build_outputs() -> tuple[str, str, str, str, str, str, str, str]:
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
    return card.render(df), md, as_json + "\n", tenure_md, preprocessing_md, linear_md, trees_md, confusion_md


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlproject")
    parser.add_argument("command", choices=["check", "card", "baseline", "all"])
    args = parser.parse_args(argv)
    if args.command == "check":
        df = load()
        print(f"data is valid: {len(df):,} rows, churn rate {df['churned'].mean():.1%}")
        return 0
    card_md, report_md, report_json, tenure_md, preprocessing_md, linear_md, trees_md, confusion_md = build_outputs()
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
        print(f"wrote {REPORT_MD.relative_to(ROOT)}, {REPORT_JSON.relative_to(ROOT)}, {TENURE_MD.relative_to(ROOT)}, {PREPROCESS_MD.relative_to(ROOT)}, {LINEAR_MD.relative_to(ROOT)}, {TREES_MD.relative_to(ROOT)} and {CONFUSION_MD.relative_to(ROOT)}")
        print(report_md)
        print(tenure_md)
        print(preprocessing_md)
        print(linear_md)
        print(trees_md)
        print(confusion_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

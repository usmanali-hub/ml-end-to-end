"""A small neural network (PyTorch multilayer perceptron) for churn, compared honestly with the earlier models.

On a table of 19 columns and a few thousand rows, neural networks are not expected to beat a regularised logistic
regression or gradient boosting, and this module is built to find out rather than assume. The network is wrapped as a
scikit-learn estimator so it uses the same preprocessing, the same 5 stratified folds and the same metrics as every
other model. The test split has not been used.

Training is made reproducible on CPU: a fixed seed for initialisation, dropout and batch order, one thread, and early
stopping on a held-out slice of the *fitting rows* (never the evaluation fold). Values are reported to two decimals
because floating-point summation order differs between processors and a trained network amplifies that.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from . import boosting, calibration
from .baseline import SEED
from .data import TARGET
from .evaluation import out_of_fold_scores
from .preprocess import RECOMMENDED, VARIANTS, build_preprocessor
from .trees import COLUMNS, cross_validate, folds_of

HIDDEN = ((), (16,), (64,), (64, 32))
WEIGHT_DECAYS = (0.0, 1e-3, 1e-2)
DROPOUT = 0.2
TOLERANCE = 0.001
SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class MLPSpec:
    hidden: tuple[int, ...]
    weight_decay: float

    @property
    def label(self) -> str:
        layers = "none (logistic regression)" if not self.hidden else " x ".join(str(h) for h in self.hidden)
        return f"hidden {layers}, weight decay {self.weight_decay:g}"


def _dense(x: Any) -> np.ndarray:
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float32)


def _network(n_in: int, hidden: tuple[int, ...], dropout: float) -> torch.nn.Sequential:
    layers: list[torch.nn.Module] = []
    width = n_in
    for h in hidden:
        layers += [torch.nn.Linear(width, h), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
        width = h
    layers.append(torch.nn.Linear(width, 1))
    return torch.nn.Sequential(*layers)


class TorchMLP(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Binary classifier: ReLU layers, dropout, Adam with L2 weight decay, early stopping on a slice of the fitting rows."""

    def __init__(
        self,
        hidden: tuple[int, ...] = (32,),
        dropout: float = DROPOUT,
        weight_decay: float = 1e-3,
        lr: float = 1e-3,
        batch_size: int = 128,
        max_epochs: int = 150,
        patience: int = 12,
        val_fraction: float = 0.15,
        seed: int = SEED,
    ) -> None:
        self.hidden = hidden
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.seed = seed

    def fit(self, X: Any, y: Any) -> "TorchMLP":
        x_all, y_all = _dense(X), np.asarray(y, dtype=np.float32)
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        tr, va = train_test_split(np.arange(len(x_all)), test_size=self.val_fraction, stratify=y_all, random_state=self.seed)
        xt, yt = torch.from_numpy(x_all[tr]), torch.from_numpy(y_all[tr])
        xv, yv = torch.from_numpy(x_all[va]), torch.from_numpy(y_all[va])
        net = _network(x_all.shape[1], tuple(self.hidden), self.dropout)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        gen = torch.Generator().manual_seed(self.seed)
        best, best_state, wait = float("inf"), {k: v.clone() for k, v in net.state_dict().items()}, 0
        self.history_: list[tuple[float, float]] = []
        self.best_epoch_ = 0
        for epoch in range(1, self.max_epochs + 1):
            net.train()
            order = torch.randperm(len(xt), generator=gen)
            for i in range(0, len(xt), self.batch_size):
                batch = order[i : i + self.batch_size]
                opt.zero_grad()
                loss_fn(net(xt[batch]).squeeze(1), yt[batch]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                train_loss, val_loss = float(loss_fn(net(xt).squeeze(1), yt)), float(loss_fn(net(xv).squeeze(1), yv))
            self.history_.append((train_loss, val_loss))
            if val_loss < best - 1e-4:
                best, best_state, wait, self.best_epoch_ = val_loss, {k: v.clone() for k, v in net.state_dict().items()}, 0, epoch
            else:
                wait += 1
                if wait >= self.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        self.net_ = net
        self.classes_ = np.array([0, 1])
        self.n_parameters_ = int(sum(p.numel() for p in net.parameters()))
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        with torch.no_grad():
            p = torch.sigmoid(self.net_(torch.from_numpy(_dense(X))).squeeze(1)).numpy().astype(np.float64)
        return np.column_stack([1 - p, p])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_mlp(spec: MLPSpec, seed: int = SEED) -> Pipeline:
    return Pipeline(
        [("prep", build_preprocessor(options=VARIANTS[RECOMMENDED])), ("model", TorchMLP(spec.hidden, DROPOUT, spec.weight_decay, seed=seed))]
    )


def _pr(pipe: Pipeline, df: pd.DataFrame) -> float:
    return float(average_precision_score(df[TARGET], np.asarray(pipe.predict_proba(df[COLUMNS]))[:, 1]))


def grid(train: pd.DataFrame) -> pd.DataFrame:
    """Held-out PR-AUC, epochs used and parameter count for every architecture and weight decay."""
    rows = []
    for hidden in HIDDEN:
        for wd in WEIGHT_DECAYS:
            spec = MLPSpec(hidden, wd)
            scores, epochs, params = [], [], 0
            for fit_idx, val_idx in folds_of(train):
                pipe = make_mlp(spec).fit(train.iloc[fit_idx][COLUMNS], train.iloc[fit_idx][TARGET])
                scores.append(_pr(pipe, train.iloc[val_idx]))
                model: TorchMLP = pipe.named_steps["model"]
                epochs.append(model.best_epoch_)
                params = model.n_parameters_
            rows.append(
                {"hidden": hidden, "weight_decay": wd, "parameters": params, "cv_pr_auc": float(np.mean(scores)), "cv_std": float(np.std(scores, ddof=1)), "epochs": float(np.mean(epochs))}
            )
    return pd.DataFrame(rows)


def choose(table: pd.DataFrame, tolerance: float = TOLERANCE) -> MLPSpec:
    """Within ``tolerance`` of the best score, the network with the fewest parameters, then the stronger weight decay."""
    limit = float(table["cv_pr_auc"].max()) - tolerance
    ok = table[table["cv_pr_auc"] >= limit].sort_values(["parameters", "weight_decay"], ascending=[True, False])
    best = ok.iloc[0]
    return MLPSpec(tuple(best["hidden"]), float(best["weight_decay"]))


def seed_spread(train: pd.DataFrame, spec: MLPSpec) -> pd.DataFrame:
    """Mean cross-validated PR-AUC for each training seed (same folds), to separate seed noise from fold noise."""
    rows = []
    for seed in SEEDS:
        scores = []
        for fit_idx, val_idx in folds_of(train):
            pipe = make_mlp(spec, seed).fit(train.iloc[fit_idx][COLUMNS], train.iloc[fit_idx][TARGET])
            scores.append(_pr(pipe, train.iloc[val_idx]))
        rows.append({"seed": seed, "pr_auc": float(np.mean(scores))})
    return pd.DataFrame(rows)


def history(train: pd.DataFrame, spec: MLPSpec) -> pd.DataFrame:
    """Training and validation loss by epoch for the first fold (validation = the slice held out of the fitting rows)."""
    fit_idx, _ = folds_of(train)[0]
    pipe = make_mlp(spec).fit(train.iloc[fit_idx][COLUMNS], train.iloc[fit_idx][TARGET])
    model: TorchMLP = pipe.named_steps["model"]
    h = pd.DataFrame(model.history_, columns=["train_loss", "validation_loss"])
    h.insert(0, "epoch", np.arange(1, len(h) + 1))
    h.attrs["best_epoch"] = model.best_epoch_
    return h


@dataclass
class NeuralAnalysis:
    grid: pd.DataFrame
    chosen: MLPSpec
    comparison: pd.DataFrame  # per model: pr_auc mean/sd, roc_auc, brier, delta vs logistic, folds better
    seeds: pd.DataFrame
    history: pd.DataFrame
    best_epoch: int
    calibration: dict[str, float]
    n: int


def analyse(train: pd.DataFrame, logistic: Callable[[], Pipeline], boost: boosting.BoostSpec) -> NeuralAnalysis:
    table = grid(train)
    spec = choose(table)
    named: dict[str, Callable[[], Pipeline]] = {
        "logistic_l2_chosen": logistic,
        "mlp_chosen": partial(make_mlp, spec),
        "lightgbm_chosen": partial(boosting.make_boosted, boost),
    }
    folds = cross_validate(train, named)
    pr = folds.pivot(index="fold", columns="model", values="pr_auc")
    paired = pr.sub(pr["logistic_l2_chosen"], axis=0)
    g = folds.groupby("model")
    comparison = pd.DataFrame(
        {
            "pr_auc": g["pr_auc"].mean(),
            "pr_auc_sd": g["pr_auc"].std(ddof=1),
            "roc_auc": g["roc_auc"].mean(),
            "brier": g["brier"].mean(),
            "delta": paired.mean(),
            "delta_sd": paired.std(ddof=1),
            "folds_better": (paired > 0).sum(),
        }
    ).loc[list(named)]
    h = history(train, spec)
    oof = out_of_fold_scores(train, partial(make_mlp, spec))
    cal = calibration.metrics(train[TARGET].to_numpy(), oof)
    return NeuralAnalysis(table, spec, comparison, seed_spread(train, spec), h, int(h.attrs["best_epoch"]), cal, len(train))




def markdown(a: NeuralAnalysis) -> str:
    g, comp = a.grid, a.comparison.to_dict("index")
    mlp, logi, gbm = comp["mlp_chosen"], comp["logistic_l2_chosen"], comp["lightgbm_chosen"]
    none = g[g["parameters"] == g["parameters"].min()]
    best_none = float(none["cv_pr_auc"].max())
    spread = float(g["cv_pr_auc"].max() - g["cv_pr_auc"].min())
    seeds = a.seeds["pr_auc"]
    chosen = a.chosen
    h = a.history
    shown = sorted({*range(1, len(h) + 1, 5), a.best_epoch, len(h)})
    cal = a.calibration
    lines = [
        "# Neural network (PyTorch MLP)",
        "",
        (
            f"A multilayer perceptron on the same preprocessing as the logistic model, trained on out-of-fold splits of the {a.n:,}-row training split (the test split has not been used). "
            f"ReLU layers with dropout {DROPOUT}, Adam (learning rate 1e-3, batch 128) with L2 weight decay, and early stopping on 15% of the *fitting* rows (patience 12, at most 150 epochs), "
            "so the evaluation fold never influences training. CPU, float32, one thread, fixed seeds. Floating-point summation order differs between machines and a trained network amplifies that, "
            "so MLP figures are rounded to three decimals at most and the freshness test compares them with a tolerance (0.01), not exactly."
        ),
        "",
        "## 1. Architecture and weight decay",
        "",
        "PR-AUC on held-out folds (mean and standard deviation over 5 folds, training seed 42). `none` is no hidden layer: logistic regression trained by Adam, a check that the hidden layers earn their keep.",
        "",
        "| hidden layers | weight decay | parameters | PR-AUC | fold sd | epochs used |",
        "|---|---|---|---|---|---|",
        *[
            f"| {'none' if not r['hidden'] else ' x '.join(str(x) for x in r['hidden'])} | {r['weight_decay']:g} | {int(r['parameters']):,} | {r['cv_pr_auc']:.3f} | {r['cv_std']:.3f} | {r['epochs']:.0f} |"
            for _, r in g.iterrows()
        ],
        "",
        (
            f"All {len(g)} configurations lie within {spread:.3f} PR-AUC of each other, against a fold standard deviation of about {float(g['cv_std'].median()):.3f}. "
            f"The best network with no hidden layer scores {best_none:.3f} and the best of the rest {float(g['cv_pr_auc'].max()):.3f}. "
            f"Rule used (as for the other models): within {TOLERANCE} of the best score, the fewest parameters, then the stronger weight decay. Chosen: {chosen.label}."
        ),
        "",
        "## 2. Against the earlier models",
        "",
        "Same 5 folds. `vs logistic` is the paired per-fold difference in PR-AUC.",
        "",
        "| model | PR-AUC | fold sd | ROC-AUC | Brier | vs logistic (mean ± sd) | folds better |",
        "|---|---|---|---|---|---|---|",
        *[
            f"| `{n}` | {r['pr_auc']:.3f} | {r['pr_auc_sd']:.3f} | {r['roc_auc']:.3f} | {r['brier']:.3f} | {r['delta']:+.3f} ± {r['delta_sd']:.3f} | {int(r['folds_better'])} of 5 |"
            for n, r in comp.items()
        ],
        "",
        (
            f"The network differs from regularised logistic regression by {mlp['delta']:+.3f} PR-AUC, with a standard deviation of {mlp['delta_sd']:.3f} across folds; LightGBM differs by {gbm['delta']:+.3f} ± {gbm['delta_sd']:.3f}. "
            f"The logistic model's own score is {logi['pr_auc']:.3f}."
        ),
        "",
        "## 3. How much of that is the random seed",
        "",
        (
            f"The architecture was picked, and the comparison above run, with seed 42. Retraining the chosen network with five other seeds on the same folds gives PR-AUC "
            f"{seeds.min():.3f} to {seeds.max():.3f} (mean {seeds.mean():.3f}); seed 42 gave {mlp['pr_auc']:.3f}. "
            + (
                "Seed 42 is above the five-seed average, so part of the network's headline figure is the luck of one seed plus picking the best-looking configuration on it; "
                if float(mlp["pr_auc"]) > float(seeds.mean()) + 0.001
                else "Seed 42 is not above the five-seed average here; "
            )
            + f"the seed average is the fairer number to compare with the logistic model. For scale, the seeds span {seeds.max() - seeds.min():.3f} and the architectures in section 1 span {spread:.3f}."
        ),
        "",
        "## 4. Training curve (first fold)",
        "",
        "Binary cross-entropy on the fitting rows used for gradient steps and on the 15% slice held out of them.",
        "",
        "| epoch | training loss | validation loss |",
        "|---|---|---|",
        *[f"| {int(h.loc[e - 1, 'epoch'])} | {h.loc[e - 1, 'train_loss']:.3f} | {h.loc[e - 1, 'validation_loss']:.3f} |" for e in shown],
        "",
        f"Early stopping kept the weights from epoch {a.best_epoch}, the epoch with the lowest validation loss, and stopped training after epoch {len(h)}.",
        "",
        "## 5. Calibration",
        "",
        f"Out-of-fold scores from the chosen network: ECE {cal['ece']:.2f}, calibration slope {cal['slope']:.2f}, mean predicted churn {cal['mean_score']:.2f} against the true rate of the training split. Log loss {cal['log_loss']:.2f}, Brier {cal['brier']:.2f}.",
        "",
        "## What this does not show",
        "",
        "- One architecture family on one small tabular dataset; larger networks, embeddings for the categories or longer schedules were not tried and may behave differently.",
        "- The grid was searched with one seed, so the chosen configuration is slightly optimistic; section 3 shows how much that can matter.",
        "- All figures are cross-validation on the training split, with early stopping using 15% of each fitting set, so the network sees 85% of the rows that the other models see.",
        f"- The largest paired difference between models above is {max(abs(mlp['delta']), abs(gbm['delta'])):.3f}, against fold standard deviations of {mlp['delta_sd']:.3f} and {gbm['delta_sd']:.3f}.",
        "",
    ]
    return "\n".join(lines).replace("-0.000", "0.000")

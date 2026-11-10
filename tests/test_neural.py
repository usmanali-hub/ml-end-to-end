import numpy as np
import pandas as pd

from mlproject import neural
from mlproject.data import TARGET, load
from mlproject.split import split
from mlproject.trees import COLUMNS


def separable(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
    return x, y


def test_network_learns_a_separable_problem_and_returns_valid_probabilities():
    x, y = separable()
    m = neural.TorchMLP(hidden=(8,), max_epochs=60).fit(x, y)
    p = m.predict_proba(x)
    assert p.shape == (600, 2) and np.allclose(p.sum(axis=1), 1.0) and ((p >= 0) & (p <= 1)).all()
    assert (m.predict(x) == y).mean() > 0.9


def test_same_seed_gives_identical_predictions_and_different_seeds_do_not():
    x, y = separable()
    a = neural.TorchMLP(hidden=(8,), max_epochs=15, seed=1).fit(x, y).predict_proba(x)
    b = neural.TorchMLP(hidden=(8,), max_epochs=15, seed=1).fit(x, y).predict_proba(x)
    c = neural.TorchMLP(hidden=(8,), max_epochs=15, seed=2).fit(x, y).predict_proba(x)
    assert np.array_equal(a, b) and not np.allclose(a, c)


def test_early_stopping_keeps_the_best_epoch_and_stops_before_the_limit_on_noise():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(500, 6)).astype(np.float32)
    y = rng.integers(0, 2, 500)  # pure noise: validation loss cannot keep improving
    m = neural.TorchMLP(hidden=(32,), max_epochs=150, patience=5).fit(x, y)
    assert len(m.history_) < 150 and 1 <= m.best_epoch_ <= len(m.history_)
    assert m.history_[m.best_epoch_ - 1][1] <= min(v for _, v in m.history_) + 1e-4  # an epoch must beat the best by 1e-4 to count


def test_no_hidden_layer_is_a_linear_model_with_one_weight_per_input_plus_bias():
    x, y = separable()
    m = neural.TorchMLP(hidden=(), max_epochs=5).fit(x, y)
    assert m.n_parameters_ == 5 + 1


def test_parameter_counts_follow_the_architecture():
    x, y = separable()
    m = neural.TorchMLP(hidden=(4, 3), max_epochs=2).fit(x, y)
    assert m.n_parameters_ == (5 * 4 + 4) + (4 * 3 + 3) + (3 + 1)


def test_choose_takes_the_smallest_network_within_the_tolerance_of_the_best():
    table = pd.DataFrame(
        {
            "hidden": [(), (16,), (64,)],
            "weight_decay": [0.001, 0.001, 0.001],
            "parameters": [46, 753, 3009],
            "cv_pr_auc": [0.6715, 0.6710, 0.6720],
        }
    )
    assert neural.choose(table).hidden == ()  # 0.6715 is within 0.001 of 0.6720
    table.loc[0, "cv_pr_auc"] = 0.660
    assert neural.choose(table).hidden == (16,)


def test_spec_label_names_the_no_hidden_layer_case():
    assert "logistic" in neural.MLPSpec((), 0.01).label and "64 x 32" in neural.MLPSpec((64, 32), 0.0).label


def test_pipeline_fits_on_real_churn_rows_and_beats_the_base_rate():
    train = split(load())[0].iloc[:1500]
    pipe = neural.make_mlp(neural.MLPSpec((8,), 0.001)).fit(train[COLUMNS], train[TARGET])
    p = pipe.predict_proba(train[COLUMNS])[:, 1]
    from sklearn.metrics import average_precision_score

    assert average_precision_score(train[TARGET], p) > train[TARGET].mean() + 0.2


def test_report_states_its_caveats():
    from mlproject import __main__ as cli

    text = cli.NEURAL_MD.read_text(encoding="utf-8")
    assert "## What this does not show" in text and "## 3. How much of that is the random seed" in text

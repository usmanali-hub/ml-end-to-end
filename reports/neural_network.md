# Neural network (PyTorch MLP)

A multilayer perceptron on the same preprocessing as the logistic model, trained on out-of-fold splits of the 5,634-row training split (the test split has not been used). ReLU layers with dropout 0.2, Adam (learning rate 1e-3, batch 128) with L2 weight decay, and early stopping on 15% of the *fitting* rows (patience 12, at most 150 epochs), so the evaluation fold never influences training. CPU, float32, one thread, fixed seeds. Floating-point summation order differs between machines and a trained network amplifies that, so MLP figures are rounded to three decimals at most and the freshness test compares them with a tolerance (0.01), not exactly.

## 1. Architecture and weight decay

PR-AUC on held-out folds (mean and standard deviation over 5 folds, training seed 42). `none` is no hidden layer: logistic regression trained by Adam, a check that the hidden layers earn their keep.

| hidden layers | weight decay | parameters | PR-AUC | fold sd | epochs used |
|---|---|---|---|---|---|
| none | 0 | 46 | 0.671 | 0.020 | 66 |
| none | 0.001 | 46 | 0.670 | 0.020 | 68 |
| none | 0.01 | 46 | 0.671 | 0.018 | 68 |
| 16 | 0 | 753 | 0.672 | 0.022 | 23 |
| 16 | 0.001 | 753 | 0.673 | 0.023 | 27 |
| 16 | 0.01 | 753 | 0.672 | 0.019 | 41 |
| 64 | 0 | 3,009 | 0.675 | 0.020 | 16 |
| 64 | 0.001 | 3,009 | 0.674 | 0.019 | 14 |
| 64 | 0.01 | 3,009 | 0.673 | 0.018 | 30 |
| 64 x 32 | 0 | 5,057 | 0.671 | 0.020 | 10 |
| 64 x 32 | 0.001 | 5,057 | 0.671 | 0.019 | 10 |
| 64 x 32 | 0.01 | 5,057 | 0.670 | 0.018 | 16 |

All 12 configurations lie within 0.004 PR-AUC of each other, against a fold standard deviation of about 0.019. The best network with no hidden layer scores 0.671 and the best of the rest 0.675. Rule used (as for the other models): within 0.001 of the best score, the fewest parameters, then the stronger weight decay. Chosen: hidden 64, weight decay 0.001.

## 2. Against the earlier models

Same 5 folds. `vs logistic` is the paired per-fold difference in PR-AUC.

| model | PR-AUC | fold sd | ROC-AUC | Brier | vs logistic (mean ± sd) | folds better |
|---|---|---|---|---|---|---|
| `logistic_l2_chosen` | 0.674 | 0.018 | 0.850 | 0.133 | +0.000 ± 0.000 | 0 of 5 |
| `mlp_chosen` | 0.674 | 0.019 | 0.850 | 0.133 | +0.000 ± 0.003 | 2 of 5 |
| `lightgbm_chosen` | 0.672 | 0.022 | 0.850 | 0.133 | -0.002 ± 0.007 | 3 of 5 |

The network differs from regularised logistic regression by +0.000 PR-AUC, with a standard deviation of 0.003 across folds; LightGBM differs by -0.002 ± 0.007. The logistic model's own score is 0.674.

## 3. How much of that is the random seed

The architecture was picked, and the comparison above run, with seed 42. Retraining the chosen network with five other seeds on the same folds gives PR-AUC 0.667 to 0.672 (mean 0.669); seed 42 gave 0.674. Seed 42 is above the five-seed average, so part of the network's headline figure is the luck of one seed plus picking the best-looking configuration on it; the seed average is the fairer number to compare with the logistic model. For scale, the seeds span 0.004 and the architectures in section 1 span 0.004.

## 4. Training curve (first fold)

Binary cross-entropy on the fitting rows used for gradient steps and on the 15% slice held out of them.

| epoch | training loss | validation loss |
|---|---|---|
| 1 | 0.515 | 0.518 |
| 6 | 0.407 | 0.413 |
| 11 | 0.403 | 0.413 |
| 15 | 0.401 | 0.411 |
| 16 | 0.400 | 0.412 |
| 21 | 0.399 | 0.411 |
| 26 | 0.399 | 0.414 |
| 27 | 0.396 | 0.412 |

Early stopping kept the weights from epoch 15, the epoch with the lowest validation loss, and stopped training after epoch 27.

## 5. Calibration

Out-of-fold scores from the chosen network: ECE 0.01, calibration slope 1.01, mean predicted churn 0.26 against the true rate of the training split. Log loss 0.41, Brier 0.13.

## What this does not show

- One architecture family on one small tabular dataset; larger networks, embeddings for the categories or longer schedules were not tried and may behave differently.
- The grid was searched with one seed, so the chosen configuration is slightly optimistic; section 3 shows how much that can matter.
- All figures are cross-validation on the training split, with early stopping using 15% of each fitting set, so the network sees 85% of the rows that the other models see.
- The largest paired difference between models above is 0.002, against fold standard deviations of 0.003 and 0.007.

# Confusion matrices on the churn data

Every training customer (5,634, of whom 1,495 churned, 26.5%) is scored by the chosen logistic model fitted on the *other* four folds, so no customer is scored by a model that saw them. The test split has not been used. A confusion matrix needs a cutoff: flag a customer when the predicted churn probability is at or above it.

## 1. Reading one matrix (cutoff 0.5)

| | flagged as churn | not flagged |
|---|---|---|
| **actually churned** (1,495) | 794 true positives | 701 false negatives (missed) |
| **actually stayed** (4,139) | 374 false positives (wasted offers) | 3,765 true negatives |

| measure | arithmetic | value | in customers |
|---|---|---|---|
| accuracy | (794 + 3,765) / 5,634 | 0.809 | 4,559 of 5,634 customers classed correctly |
| precision | 794 / (794 + 374) | 0.680 | of 1,168 flagged customers, 794 really would leave |
| recall | 794 / (794 + 701) | 0.531 | of 1,495 who left, 794 were caught |
| specificity | 3,765 / (3,765 + 374) | 0.910 | of 4,139 who stayed, 3,765 were left alone |
| F1 | 2 x precision x recall / (precision + recall) | 0.596 | one number balancing precision and recall |

## 2. Why accuracy alone misleads

| strategy | flagged | true positives | accuracy | precision | recall |
|---|---|---|---|---|---|
| flag nobody | 0 | 0 | 0.735 | 0.000 | 0.000 |
| flag everybody | 5,634 | 1,495 | 0.265 | 0.265 | 1.000 |
| the model at 0.5 | 1,168 | 794 | 0.809 | 0.680 | 0.531 |

Flagging nobody is more than 73% accurate and finds no churner at all. With one customer in four leaving, accuracy rewards doing nothing; precision and recall say what the model actually does.

## 3. The cutoff is a dial

| cutoff | flagged | true positives | false positives | false negatives | precision | recall |
|---|---|---|---|---|---|---|
| 0.1 | 3,582 | 1,422 | 2,160 | 73 | 0.397 | 0.951 |
| 0.2 | 2,709 | 1,300 | 1,409 | 195 | 0.480 | 0.870 |
| 0.3 | 2,066 | 1,138 | 928 | 357 | 0.551 | 0.761 |
| 0.4 | 1,581 | 973 | 608 | 522 | 0.615 | 0.651 |
| 0.5 | 1,168 | 794 | 374 | 701 | 0.680 | 0.531 |
| 0.6 | 749 | 546 | 203 | 949 | 0.729 | 0.365 |
| 0.7 | 410 | 334 | 76 | 1,161 | 0.815 | 0.223 |

Lowering the cutoff catches more churners and wastes more offers; raising it does the opposite. Nothing about the model changes, only which mistake it prefers.
The cutoff with the best F1 is 0.33 (F1 0.640, recall 0.726, precision 0.572).

## 4. Choosing a cutoff from what mistakes cost

The right cutoff depends on two business numbers that are not in this data: what an offer to a loyal customer costs, and what a lost customer is worth. The table is illustrative: it assumes a wasted offer costs 1 unit and shows what happens as a missed churner costs 1, 2, 5 or 10 units.

| a miss costs | best cutoff | flagged | false positives | false negatives | recall | precision | total cost | cost at cutoff 0.5 |
|---|---|---|---|---|---|---|---|---|
| 1x | 0.49 | 1,212 | 394 | 677 | 0.547 | 0.675 | 1,071 | 1,075 |
| 2x | 0.35 | 1,788 | 738 | 445 | 0.702 | 0.587 | 1,628 | 1,776 |
| 5x | 0.19 | 2,787 | 1,467 | 175 | 0.883 | 0.474 | 2,342 | 3,879 |
| 10x | 0.1 | 3,582 | 2,160 | 73 | 0.951 | 0.397 | 2,890 | 7,384 |

The more a miss costs, the lower the cutoff should be, and the larger the saving over the default 0.5. If a miss costs about the same as a wasted offer, 0.5 is already close to right.

## 5. With a limited retention budget

Contact only the highest-scoring customers. The base rate is 26.5%, so a random list of the same size would be right about that often.

| contact the top | customers | caught churners | hit rate (precision) | share of all churners caught | lift over random |
|---|---|---|---|---|---|
| 5% | 282 | 244 | 0.865 | 0.163 | 3.3x |
| 10% | 563 | 430 | 0.764 | 0.288 | 2.9x |
| 20% | 1,127 | 772 | 0.685 | 0.516 | 2.6x |
| 30% | 1,690 | 1,012 | 0.599 | 0.677 | 2.3x |

Scores are out-of-fold on the training split. A final cutoff would be set here and then applied once to the untouched test split.

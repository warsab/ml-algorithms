<div align="center">

# 🧠 ml-algorithms

### *Twenty-four machine learning algorithms, explained and ready to run.*

**A reference collection of scikit-learn model templates — each one paired with a
written explanation of how the method works and when to reach for it.**

<img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
<img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge" />

[![templates](https://github.com/warsab/ml-algorithms/actions/workflows/templates.yml/badge.svg)](https://github.com/warsab/ml-algorithms/actions/workflows/templates.yml)

</div>

---

## What this is

Every file follows the same two-part shape:

1. **Model definition** — a plain-English breakdown of the algorithm: the mechanics,
   its assumptions, and the situations it suits.
2. **Template** — a runnable scikit-learn implementation with example data, ready to
   swap your own dataset into.
3. **Evaluation** — every template scores itself. Classifiers print accuracy, a
   per-class precision/recall/F1 report and a confusion matrix; regressors print
   MSE, RMSE, MAE and R²; clustering prints silhouette scores. All of them also run
   5-fold cross-validation, because a single train/test split can flatter a model.

The point is to have the explanation and the working code in the same place, so
picking an algorithm and starting with it are the same step.

---

---

## 🧭 Which algorithm should I use?

Start with the question you're actually asking, not the algorithm you've heard of.

### Do you have labelled examples?

| Situation | Go to |
|---|---|
| **Yes**, and the answer is a **number** (price, temperature, score) | [Regression](#-regression) |
| **Yes**, and the answer is a **category** (spam/not, churn/stay) | [Supervised](#-supervised-learning) or [Ensemble](#-ensemble-methods) |
| **No** — you want to discover structure | [Unsupervised](#-unsupervised-learning) |
| Your data is **text** | [NLP](#-nlp) |

### Predicting a number

| If you need | Use | Why |
|---|---|---|
| A simple, interpretable baseline | [`linear_regression.py`](regression/linear_regression.py) | Always start here — everything else must beat it |
| Several predictors | [`multiple_linear_regression.py`](regression/multiple_linear_regression.py) | Same model, more features |
| To capture curvature | [`polynomial_regression.py`](regression/polynomial_regression.py) | Fits bends a straight line can't |
| Many correlated features | [`ridge_regression.py`](regression/ridge_regression.py) | L2 shrinks coefficients, handles collinearity |
| To drop irrelevant features automatically | [`lasso_regression.py`](regression/lasso_regression.py) | L1 forces weak coefficients to exactly zero |
| Both of the above | [`elasticnet_regression.py`](regression/elasticnet_regression.py) | Blends L1 and L2 |

### Predicting a category

| If you need | Use | Why |
|---|---|---|
| An interpretable baseline | [`logistic_regression.py`](supervised/logistic_regression.py) | Fast, and gives calibrated probabilities |
| Rules a human can read | [`decision_trees.py`](supervised/decision_trees.py) | The tree *is* the explanation |
| Strong accuracy with little tuning | [`random_forest.py`](ensemble/random_forest.py) | The reliable default for tabular data |
| Maximum accuracy on tabular data | [`gradient_boosting.py`](ensemble/gradient_boosting.py) | Usually the top performer, needs more tuning |
| A text classifier, or a fast baseline | [`naive_bayes.py`](supervised/naive_bayes.py) | Cheap, and surprisingly strong on text |
| To work in high dimensions | [`support_vector_machine.py`](supervised/support_vector_machine.py) | Effective when features outnumber samples |
| Something simple with no training step | [`k_nearest_neighbors.py`](supervised/k_nearest_neighbors.py) | Just stores the data; slow at predict time |
| To model complex non-linear patterns | [`neural_networks.py`](supervised/neural_networks.py) | Flexible, but needs volume and tuning |

### Finding structure without labels

| If you need | Use | Why |
|---|---|---|
| Groups, and you know how many | [`kmeans_clustering.py`](unsupervised/kmeans_clustering.py) | Fast, but you must pick k up front |
| Groups, and you *don't* know how many | [`dbscan.py`](unsupervised/dbscan.py) | Finds k itself, and labels outliers as noise |
| To see how groups nest | [`hierarchical_clustering.py`](unsupervised/hierarchical_clustering.py) | Dendrogram lets you cut at any level |
| Fewer features / a plot of high-dim data | [`pca.py`](unsupervised/pca.py) | Compresses to the directions carrying most variance |
| To flag unusual records | [`anomaly_detection.py`](unsupervised/anomaly_detection.py) | Isolates rare points — fraud, faults, outliers |

### Rules of thumb

- **Start simple.** A linear or logistic baseline tells you whether the problem is
  even learnable, and gives every later model something to beat.
- **Tabular data? Reach for trees.** Random forest and gradient boosting beat neural
  networks on most spreadsheet-shaped problems.
- **Interpretability is a requirement, not a nicety.** If you must explain a decision,
  a decision tree or linear model is worth the accuracy you give up.
- **Don't trust one train/test split.** Every template here also reports 5-fold
  cross-validation for exactly this reason.
- **Accuracy lies on imbalanced data.** A model calling everything "not fraud" scores
  99% on a 1%-fraud dataset. Read the precision, recall and confusion matrix.

## 📈 Regression

| File | Algorithm |
|---|---|
| [`linear_regression.py`](regression/linear_regression.py) | Ordinary least squares |
| [`multiple_linear_regression.py`](regression/multiple_linear_regression.py) | Multiple predictors |
| [`polynomial_regression.py`](regression/polynomial_regression.py) | Non-linear via polynomial features |
| [`ridge_regression.py`](regression/ridge_regression.py) | L2 regularisation |
| [`lasso_regression.py`](regression/lasso_regression.py) | L1 regularisation, feature selection |
| [`elasticnet_regression.py`](regression/elasticnet_regression.py) | Combined L1 + L2 |

## 🎯 Supervised learning

| File | Algorithm |
|---|---|
| [`logistic_regression.py`](supervised/logistic_regression.py) | Binary and multiclass classification |
| [`decision_trees.py`](supervised/decision_trees.py) | Recursive splitting |
| [`k_nearest_neighbors.py`](supervised/k_nearest_neighbors.py) | Instance-based learning |
| [`support_vector_machine.py`](supervised/support_vector_machine.py) | Maximum-margin classification |
| [`neural_networks.py`](supervised/neural_networks.py) | Multi-layer perceptron |
| [`naive_bayes.py`](supervised/naive_bayes.py) | Probabilistic classification |

## 🔍 Unsupervised learning

| File | Algorithm |
|---|---|
| [`kmeans_clustering.py`](unsupervised/kmeans_clustering.py) | Centroid-based clustering |
| [`hierarchical_clustering.py`](unsupervised/hierarchical_clustering.py) | Agglomerative clustering |
| [`dbscan.py`](unsupervised/dbscan.py) | Density-based clustering |
| [`pca.py`](unsupervised/pca.py) | Dimensionality reduction |
| [`anomaly_detection.py`](unsupervised/anomaly_detection.py) | Outlier identification |

## 🌲 Ensemble methods

| File | Algorithm |
|---|---|
| [`random_forest.py`](ensemble/random_forest.py) | Bagged decision trees |
| [`bagging.py`](ensemble/bagging.py) | Bootstrap aggregating |
| [`boosting.py`](ensemble/boosting.py) | Sequential weak learners |
| [`adaboost.py`](ensemble/adaboost.py) | Adaptive boosting |
| [`gradient_boosting.py`](ensemble/gradient_boosting.py) | Gradient-based boosting |
| [`stacking.py`](ensemble/stacking.py) | Meta-learner over base models |

## 💬 NLP

| File | Topic |
|---|---|
| [`nlp_pipeline.py`](nlp/nlp_pipeline.py) | Text preprocessing and modelling |

---

## Usage

Each file runs standalone against generated example data:

```bash
pip install -r requirements.txt
python regression/linear_regression.py
```

Every template generates its own sample data, so each file executes as-is — no
dataset needed to try one out. Replace that block with your own data to put the
model to work.

**Every template is executed on each push** by the [templates workflow](.github/workflows/templates.yml),
against Python 3.9 and 3.12, so nothing here silently rots.

Templates that plot call `plt.show()`. On a machine with no display, set
`MPLBACKEND=Agg` to stop that blocking:

```bash
MPLBACKEND=Agg python unsupervised/dbscan.py
```

---

## 📝 License

Released under the MIT License — see [LICENSE](LICENSE).

MIT © 2025 Warrick Sabatta

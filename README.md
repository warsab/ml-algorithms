<div align="center">

# 🧠 ml-algorithms

### *Twenty-four machine learning algorithms, explained and ready to run.*

**A reference collection of scikit-learn model templates — each one paired with a
written explanation of how the method works and when to reach for it.**

<img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
<img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge" />

</div>

---

## What this is

Every file follows the same two-part shape:

1. **Model definition** — a plain-English breakdown of the algorithm: the mechanics,
   its assumptions, and the situations it suits.
2. **Template** — a runnable scikit-learn implementation with example data, ready to
   swap your own dataset into.

The point is to have the explanation and the working code in the same place, so
picking an algorithm and starting with it are the same step.

---

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
pip install scikit-learn numpy pandas
python regression/linear_regression.py
```

Every template generates its own sample data via scikit-learn's `make_*` helpers, so
each file executes as-is. Replace that block with your own dataset to put the model
to work.

---

## 📝 License

MIT © 2025 Warrick Sabatta

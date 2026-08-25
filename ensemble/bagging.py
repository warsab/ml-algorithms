#================================================================ Model Def:
'''
Bagging, short for Bootstrap Aggregating, is a machine learning ensemble method used for improving the stability and accuracy of machine learning algorithms. Here's a breakdown of how Bagging works and when to use it:

    >   Bootstrap Sampling: Bagging generates multiple bootstrap samples from the training dataset by randomly selecting samples with replacement. Each bootstrap sample has the same size as the original dataset but may contain duplicate instances.

    >   Base Learners: Bagging trains a base learner (base estimator) on each bootstrap sample. Typically, the base learner is a high-variance model, such as decision trees, that can benefit from averaging multiple predictions.

    >   Aggregation: Bagging aggregates the predictions from all base learners to make the final prediction. For regression tasks, predictions are usually averaged, while for classification tasks, predictions are often combined through majority voting.

    >   Parallel Training: Bagging allows base learners to be trained in parallel, making it computationally efficient and suitable for large datasets.

When to use Bagging:

* Variance Reduction: Bagging reduces variance by averaging predictions from multiple base learners, making it effective for reducing overfitting and improving generalization performance.
* Stability Improvement: Bagging improves the stability of models by reducing the sensitivity to variations in the training data, making it suitable for noisy datasets or when the training data is limited.
* Parallelizability: Bagging allows base learners to be trained independently in parallel, making it suitable for distributed computing environments and speeding up training on large datasets.
'''

#================================================================ Template:
#====== Importing needed libraries:
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#====== Generate some example data (replace this with your actual data):
X, y = make_classification(n_samples = 1000, n_features = 20, n_classes = 2, random_state = 42)

#====== Or you can use the below to generate with your data:
# X = IDM_df.drop(columns = ['target_column'])  # Replace 'target_column' with the name of your target column
# y = IDM_df['target_column']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the Bagging classifier
bagging_classifier = BaggingClassifier(n_estimators=50, random_state=42)  # You can adjust the number of estimators

# Train the classifier
bagging_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = bagging_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#================================================================ Notes on Model construction:
'''
Make_classification model:
This is a function provided by scikit-learn that generates a random n-class classification problem.

Parameters:
>   n_samples: It specifies the number of samples to generate. Each sample is a data point.
>   n_features: It specifies the number of features (or dimensions) of each sample. Each feature is a characteristic
    or attribute of the data point.
>   n_classes: It specifies the number of classes (or categories) of the target variable.
    In a binary classification problem, n_classes is typically set to 2.
>   n_clusters_per_class: It specifies the number of clusters per class. 
    This parameter controls the separation of the classes. Higher values lead to more separated clusters.
>   random_state: It controls the random seed for reproducibility. When you set random_state to a specific value,
    the generated data will be the same each time you run the code, which is useful for reproducibility.

BaggingClassifier parameters:

>   base_estimator: The base estimator to fit on random subsets of the dataset. If None, then the base estimator is a decision tree.

>   n_estimators: The number of base estimators in the ensemble.

>   max_samples: The number of samples to draw from X to train each base estimator.

>   max_features: The number of features to draw from X to train each base estimator.

>   bootstrap: Whether samples are drawn with replacement. If False, sampling without replacement is performed.

>   bootstrap_features: Whether features are drawn with replacement.

>   oob_score: Whether to use out-of-bag samples to estimate the generalization accuracy.

>   n_jobs: The number of jobs to run in parallel for both fit and predict. -1 means using all processors.

>   random_state: Controls the random seed for reproducibility. When you set random_state to a specific value,
    the results will be the same each time you run the code, which is useful for reproducibility.
'''

#================================================================ Model Def:
'''
Random Forest is an ensemble learning method that constructs a multitude of decision trees during training and outputs the mode of the classes (classification) or the mean prediction (regression) of the individual trees. Here's a breakdown of how Random Forest works and when to use it:

    >   Ensemble of Decision Trees: Random Forest builds multiple decision trees during training by randomly selecting subsets of features and data points. Each tree is trained independently on a bootstrapped sample of the training data.

    >   Bagging: Random Forest employs a technique called bagging (bootstrap aggregating) to improve the stability and accuracy of the individual trees. Bagging reduces variance and overfitting by averaging predictions from multiple trees.

    >   Feature Randomness: Random Forest introduces additional randomness by considering only a random subset of features at each split in the decision tree. This further decorrelates the trees and reduces the chance of overfitting.

    >   Voting: For classification tasks, Random Forest combines the predictions of all individual trees through a voting mechanism and outputs the class with the most votes. For regression tasks, it outputs the mean prediction of all trees.

When to use Random Forest:

* Classification and Regression: Random Forest can be used for both classification and regression tasks.
* High Accuracy: Random Forest often produces highly accurate models, making it suitable for tasks where accuracy is crucial.
* Handling Large Datasets: Random Forest can handle large datasets with high-dimensional feature spaces efficiently due to its parallelizable nature.
* Feature Importance: Random Forest provides a measure of feature importance, allowing you to identify the most relevant features for prediction.
* Outlier Detection: Random Forest can effectively detect outliers by identifying instances that are consistently misclassified by the individual trees.
'''

#================================================================ Template:
#====== Importing needed libraries:
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

#====== Generate some example data (replace this with your actual data):
X, y = make_classification(n_samples = 1000, n_features = 20, n_classes = 2, random_state = 42)

#====== Or you can use the below to generate with your data:
# X = IDM_df.drop(columns = ['target_column'])  # Replace 'target_column' with the name of your target column
# y = IDM_df['target_column']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the Random Forest classifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust the number of estimators

# Train the classifier
random_forest_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = random_forest_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

#====== Accuracy alone hides per-class performance and is misleading when the
#       classes are imbalanced. Precision, recall and F1 per class tell you more:
print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, y_pred))

#====== A single train/test split can be lucky. Cross-validation refits the model
#       across k folds and reports the spread, which is a far more honest estimate:
cv_scores = cross_val_score(random_forest_classifier, X, y, cv=5)
print(f"\n5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

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

RandomForestClassifier parameters:

>   n_estimators: The number of decision trees in the forest.
>   criterion: The function to measure the quality of a split. Supported criteria are 'gini' for the Gini impurity and 'entropy' for the information gain.
>   max_depth: The maximum depth of the decision trees. If None, the trees are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
>   min_samples_split: The minimum number of samples required to split an internal node.
>   min_samples_leaf: The minimum number of samples required to be at a leaf node.
>   min_weight_fraction_leaf: The minimum weighted fraction of the sum total of weights (of all input samples) required to be at a leaf node.
>   max_features: The number of features to consider when looking for the best split. If 'auto', it considers the square root of the total number of features.
>   max_leaf_nodes: The maximum number of leaf nodes in each decision tree.
>   min_impurity_decrease: A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
>   bootstrap: Whether samples are drawn with replacement. If False, sampling without replacement is performed.
>   oob_score: Whether to use out-of-bag samples to estimate the generalization accuracy.
>   n_jobs: The number of jobs to run in parallel for both fit and predict. -1 means using all processors.
>   random_state: Controls the random seed for reproducibility. When you set random_state to a specific value,
    the results will be the
    
'''
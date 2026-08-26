#================================================================ Model Def:
'''
k-Nearest Neighbors (k-NN) is a non-parametric supervised learning algorithm used for classification and regression tasks. It makes predictions based on the similarity of input data points to their nearest neighbors in the feature space. Here's a breakdown of how k-NN works and when to use it:

    >   Distance Metric: k-NN calculates the distance between data points in the feature space using a chosen distance metric, such as Euclidean distance or Manhattan distance. The choice of distance metric affects the model's performance and should be selected based on the problem domain.

    >   K Neighbors: k-NN predicts the class or value of a data point by considering the labels or values of its k nearest neighbors. The value of k is a hyperparameter that determines the number of neighbors to consider. Larger values of k result in smoother decision boundaries but may lead to increased bias.

    >   Decision Rule: For classification tasks, k-NN typically uses a majority voting rule to assign the class label of a data point based on the labels of its nearest neighbors. For regression tasks, k-NN calculates the mean or median value of the target variable among its neighbors.

    >   Tuning K: The choice of the optimal value of k is crucial in k-NN. A smaller value of k can capture local patterns in the data but may lead to increased variance and sensitivity to noise. On the other hand, a larger value of k may result in smoother decision boundaries but may overlook local patterns.

When to use k-NN:

* Small to Medium-sized Datasets: k-NN is suitable for datasets with a small to medium number of samples and features, as it requires computing distances between all data points.
* Nonlinear Relationships: k-NN can capture nonlinear relationships between features and the target variable, making it suitable for complex datasets.
* Localized Patterns: k-NN is effective when the decision boundary is irregular and localized, as it can adapt to the local structure of the data.
* Simple Baseline Model: k-NN serves as a simple and intuitive baseline model for classification and regression tasks, providing a reference for more complex algorithms.
'''

#================================================================ Template:
#====== Importing needed libraries:
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

#====== Generate some example data (replace this with your actual data):
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

#====== Or you can use the below to generate with your data:
# X = IDM_df.drop(columns=['target_column'])  # Replace 'target_column' with the name of your target column
# y = IDM_df['target_column']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the k-NN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5, weights='uniform')  # You can adjust n_neighbors and weights

# Train the classifier
knn_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = knn_classifier.predict(X_test)

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
cv_scores = cross_val_score(knn_classifier, X, y, cv=5)
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

KNeighborsClassifier parameters:

>   n_neighbors: The number of neighbors to consider when making predictions.
>   weights: The weight function used in prediction. It can be 'uniform' (all neighbors have equal weight) or 'distance' (weights are inversely proportional to the distance).
>   algorithm: The algorithm used to compute the nearest neighbors. It can be 'auto', 'ball_tree', 'kd_tree', or 'brute'.
>   leaf_size: The size of the leaf nodes for the tree-based algorithms.
>   p: The power parameter for the Minkowski distance metric. It is only used when the 'minkowski' distance metric is selected.
'''

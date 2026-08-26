#================================================================ Model Def:
'''
Decision Trees are a non-parametric supervised learning method used for classification and regression tasks. They create a model that predicts the value of a target variable based on several input features. Here's a breakdown of how Decision Trees work and when to use them:

    >   Tree Structure: Decision Trees recursively split the data into subsets based on the feature that provides the best split. Each split maximizes the information gain or reduces impurity, such as Gini impurity or entropy, in the resulting subsets.

    >   Nodes and Leaves: The internal nodes of a Decision Tree represent features and the branches represent the decision rules. The leaf nodes represent the final predictions or outcomes. Decision Trees are built by selecting the best feature to split on at each node, resulting in a hierarchical tree structure.

    >   Interpretability: Decision Trees are highly interpretable, as the decision rules can be easily visualized. They mimic human decision-making processes, making them easy to understand and interpret.

    >   Overfitting: Decision Trees are prone to overfitting, especially with deep trees that capture noise in the data. Pruning techniques, limiting the tree depth, or using ensemble methods like Random Forest can help mitigate overfitting.

When to use Decision Trees:

* Interpretability: Decision Trees are suitable when interpretability is important, as they provide transparent decision rules that can be easily understood by non-experts.
* Nonlinear Relationships: Decision Trees can capture nonlinear relationships between features and the target variable, making them suitable for complex datasets.
* Feature Importance: Decision Trees naturally rank features based on their importance in the tree, making them useful for feature selection and understanding the data.
* Discrete and Continuous Data: Decision Trees can handle both discrete and continuous input features, making them versatile for a wide range of data types.
'''

#================================================================ Template:
#====== Importing needed libraries:
from sklearn.tree import DecisionTreeClassifier
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

# Instantiate the Decision Tree classifier
decision_tree_classifier = DecisionTreeClassifier(max_depth=None, random_state=42)  # You can adjust max_depth

# Train the classifier
decision_tree_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = decision_tree_classifier.predict(X_test)

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
cv_scores = cross_val_score(decision_tree_classifier, X, y, cv=5)
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

DecisionTreeClassifier parameters:

>   criterion: The function to measure the quality of a split. It can be 'gini' for Gini impurity or 'entropy' for information gain.
>   max_depth: The maximum depth of the tree. If None, nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
>   min_samples_split: The minimum number of samples required to split an internal node.
>   min_samples_leaf: The minimum number of samples required to be at a leaf node.
>   max_features: The number of features to consider when looking for the best split.
>   random_state: Controls the random seed for reproducibility. When you set random_state to a specific value,
    the results will be the same each time you run the code, which is useful for reproducibility.
'''

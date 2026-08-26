#================================================================ Model Def:
'''
Logistic Regression is a linear classification algorithm used to model the probability of a binary outcome based on one or more input features. Despite its name, Logistic Regression is used for classification rather than regression tasks. It estimates the probability that a given input belongs to a particular class using a logistic (or sigmoid) function. Here's a breakdown of how Logistic Regression works and when to use it:

    >   Probability Estimation: Logistic Regression models the probability of the positive class (usually labeled as 1) using the logistic function, which maps the input features to probabilities between 0 and 1. The logistic function produces an S-shaped curve, where values closer to 1 indicate a higher probability of belonging to the positive class.

    >   Decision Boundary: Logistic Regression predicts the class label by applying a decision threshold to the estimated probabilities. By default, the threshold is set to 0.5, meaning that instances with probabilities greater than or equal to 0.5 are predicted as the positive class, and those with probabilities below 0.5 are predicted as the negative class.

    >   Training: Logistic Regression is typically trained using maximum likelihood estimation or gradient descent to optimize the parameters (coefficients) of the linear model. The objective is to maximize the likelihood of the observed class labels given the input features.

    >   Regularization: Logistic Regression can be regularized to prevent overfitting by adding a penalty term to the loss function. L1 (Lasso) and L2 (Ridge) regularization techniques are commonly used to penalize large coefficients.

When to use Logistic Regression:

* Binary Classification: Logistic Regression is well-suited for binary classification problems where the target variable has two classes.
* Probability Estimation: Logistic Regression provides probabilistic outputs, making it suitable for tasks where understanding the confidence of predictions is important.
* Linear Decision Boundary: Logistic Regression assumes a linear relationship between the input features and the log-odds of the target variable, resulting in a linear decision boundary between classes.
* Interpretable Coefficients: Logistic Regression coefficients represent the impact of each input feature on the probability of the positive class, allowing for easy interpretation of feature importance.
'''

#================================================================ Template:
#====== Importing needed libraries:
from sklearn.linear_model import LogisticRegression
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

# Instantiate the Logistic Regression model
logistic_classifier = LogisticRegression()

# Train the model
logistic_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = logistic_classifier.predict(X_test)

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
cv_scores = cross_val_score(logistic_classifier, X, y, cv=5)
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

LogisticRegression parameters:

>   penalty: The type of regularization to apply. It can be 'l1' (Lasso) or 'l2' (Ridge). Default is 'l2'.
>   C: Inverse of regularization strength. Smaller values specify stronger regularization.
>   solver: The optimization algorithm to use. For small datasets, 'liblinear' is a good choice. For larger datasets, 'lbfgs' or 'sag' may be more efficient.
>   max_iter: The maximum number of iterations for optimization algorithms to converge.
>   random_state: Controls the random seed for reproducibility. When you set random_state to a specific value,
    the results will be the same each time you run the code, which is useful for reproducibility.
'''

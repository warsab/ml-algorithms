#================================================================ Model Def:
'''
Gradient Boosting is a machine learning ensemble method used for regression and classification tasks. It works by sequentially training weak models (typically decision trees) and adjusting their predictions based on the errors made by previous models. Here's a breakdown of how Gradient Boosting works and when to use it:

    >   Weak Learners: Gradient Boosting starts by training a base learner, often a decision tree with limited depth, on the training data. This base learner is called a weak learner because it performs only slightly better than random guessing.

    >   Gradient Descent: Gradient Boosting minimizes a loss function by iteratively fitting new models to the negative gradient of the loss function. Each new model is trained to correct the errors made by the previous models.

    >   Shrinkage: Gradient Boosting introduces a shrinkage parameter (learning rate) to control the contribution of each weak learner. A lower learning rate requires more weak learners to fit the data but improves robustness against overfitting.

    >   Regularization: Gradient Boosting supports regularization techniques such as max_depth, min_samples_split, and min_samples_leaf to prevent overfitting. These parameters control the complexity of the individual weak learners.

    >   Gradient Boosting builds a strong model by combining the predictions of multiple weak learners through a weighted sum. The final model is a weighted sum of the weak learners, where the weights are determined by the learning rate and the performance of each weak learner on the training data.

When to use Gradient Boosting:

* Regression and Classification: Gradient Boosting can be used for both regression and classification tasks.
* High Accuracy: Gradient Boosting often produces highly accurate models, making it suitable for tasks where accuracy is crucial.
* Handling Complex Relationships: Gradient Boosting can capture complex relationships between features and the target variable, making it suitable for datasets with nonlinear relationships.
* Handling Missing Data: Gradient Boosting can handle missing data effectively by using surrogate splits in decision trees.
* Handling Imbalanced Data: Gradient Boosting can handle imbalanced datasets by adjusting class weights or using class-specific metrics during training.
'''

#================================================================ Template:
#====== Importing needed libraries:
from sklearn.ensemble import GradientBoostingClassifier
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

# Instantiate the Gradient Boosting classifier
gradient_boosting_classifier = GradientBoostingClassifier(n_estimators=50, random_state=42)  # You can adjust the number of estimators

# Train the classifier
gradient_boosting_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = gradient_boosting_classifier.predict(X_test)

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

GradientBoostingClassifier parameters:

>   loss: The loss function to be optimized. It can be 'deviance' for logistic regression or 'exponential' for AdaBoost.
>   learning_rate: The learning rate shrinks the contribution of each weak learner. Lower values require more weak learners to fit the training data.
>   n_estimators: The number of boosting stages (weak learners) to be trained.
>   subsample: The fraction of samples used for fitting the individual base learners. Values less than 1.0 introduce stochasticity into the training process.
>   criterion: The function to measure the quality of a split. Supported criteria are 'friedman_mse', 'mse', and 'mae'.
>   max_depth: The maximum depth of the individual regression estimators.
>   min_samples_split: The minimum number of samples required to split an internal node.
>   min_samples_leaf: The minimum number of samples required to be at a leaf node.
>   min_weight_fraction_leaf: The minimum weighted fraction of the sum total of weights (of all input samples) required to be at a leaf node.
>   max_features: The number of features to consider when looking for the best split.
>   random_state: Controls the random seed for reproducibility. When you set random_state to a specific value,
    the results will be the same each time you run the code, which is useful for reproducibility.
>   verbose: Controls the verbosity of the output during fitting.
>   validation_fraction: The proportion of training data to set aside as validation set for early stopping.
>   n_iter_no_change: The number of consecutive iterations without improvement to wait before stopping fitting.
>   tol: The stopping criterion. If the absolute improvement in the loss function is less than tol, the optimization process stops.
>   ccp_alpha: Complexity parameter used for Minimal Cost-Complexity Pruning.
'''

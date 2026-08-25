#================================================================ Model Def:
'''
Boosting is a machine learning ensemble method that combines multiple weak learners to create a strong learner. It works by sequentially training weak models and adjusting their weights based on the performance of previous models. Here's a breakdown of how Boosting works and when to use it:

    >   Weak Learners: Boosting starts by training a base classifier, often a decision tree with limited depth, on the training data. This classifier is called a weak learner because it performs only slightly better than random guessing.

    >   Sequential Training: Boosting trains a series of weak learners sequentially, with each subsequent learner focusing more on the instances that were misclassified by the previous ones. Each new weak learner is trained on a modified version of the training data, where the weights of the misclassified instances are adjusted.

    >   Weighted Voting: After all the weak learners are trained, Boosting combines their predictions through a weighted sum or voting scheme. The predictions of the weak learners are combined using a weighted majority vote, where the weights are determined by the accuracy of each weak learner on the training data.

    >   Final Model: The final model produced by Boosting is a weighted sum of the weak learners, where the weights are determined by their accuracy. This final model is often more accurate than any individual weak learner, hence the term "boosting."

When to use Boosting:

* Classification and Regression: Boosting can be used for both classification and regression tasks.
* When High Accuracy is Required: Boosting often produces highly accurate models, making it suitable for tasks where accuracy is crucial.
* Handling Imbalanced Data: Boosting can effectively handle imbalanced datasets by focusing more on the minority class during training.
* Avoiding Overfitting: Boosting tends to generalize well and is less prone to overfitting, especially when using weak learners with low complexity.
* Interpretable Models: Boosting typically produces ensemble models that are easy to interpret, especially when using decision trees as weak learners.
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
boosting_classifier = GradientBoostingClassifier(n_estimators=50, random_state=42)  # You can adjust the number of estimators

# Train the classifier
boosting_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = boosting_classifier.predict(X_test)

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

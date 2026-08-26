#================================================================ Model Def:
'''
AdaBoost, short for Adaptive Boosting, is a machine learning algorithm primarily used for classification tasks. It's an ensemble learning
method, meaning it combines multiple weak learners to create a strong learner. Here's a breakdown of how AdaBoost works and when 
to use it:

    >   Weak Learners: AdaBoost starts by training a base classifier, often a decision tree with very limited depth (a stump), on the 
        training data. This classifier is called a weak learner because it performs only slightly better than random guessing.

    >   Weighted Training Data: After the first weak learner is trained, AdaBoost assigns higher weights to the incorrectly classified
        instances in the training data and lower weights to the correctly classified ones. This makes the misclassified instances more
        influential in subsequent rounds of training.

    >   Sequential Training: AdaBoost trains a series of weak learners sequentially, with each subsequent learner focusing more on the
        instances that were misclassified by the previous ones. Each new weak learner is trained on a modified version of the training data,
        where the weights of the misclassified instances are adjusted.

    >   Voting: After all the weak learners are trained, AdaBoost combines their predictions through a weighted sum or voting scheme.
        Typically, the predictions of the weak learners are combined using a weighted majority vote, where the weights are determined
        by the accuracy of each weak learner on the training data.

    >   Final Model: The final model produced by AdaBoost is a weighted sum of the weak learners, where the weights are determined by
        their accuracy. This final model is often more accurate than any individual weak learner, hence the term "boosting."

When to use AdaBoost:

* Binary Classification: AdaBoost is well-suited for binary classification problems.
* When Data is Imbalanced: It works well even if the dataset is imbalanced, meaning there are significantly more instances of one class than the other.
* When You Have Weak Learners: AdaBoost can boost the performance of weak classifiers, often resulting in a strong classifier.
* When You Want to Avoid Overfitting: AdaBoost tends to generalize well and is less prone to overfitting, especially when using weak learners with low complexity.
* When Interpretable Models are Needed: Since AdaBoost typically uses decision trees as weak learners, the resulting ensemble model can be easily interpreted.

'''
#================================================================ Template:
#====== Importing needed libraries:
from sklearn.ensemble import AdaBoostClassifier
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

# Instantiate the AdaBoost classifier
ada_classifier = AdaBoostClassifier(n_estimators=50, random_state=42)  # You can adjust the number of estimators

# Train the classifier
ada_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = ada_classifier.predict(X_test)

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
cv_scores = cross_val_score(ada_classifier, X, y, cv=5)
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

 AdaBoostClassifier parameters:

>   base_estimator: The base estimator from which the boosted ensemble is built. By default, it uses DecisionTreeClassifier(max_depth=1).

>   n_estimators: The maximum number of weak learners (base estimators) to train. Increasing the number of estimators
    typically improves the performance of the model, but it also increases computation time and risk of overfitting.

>   learning_rate: The contribution of each weak learner to the final prediction. A lower learning rate makes
    the model more robust by reducing the contribution of each weak learner, but it requires a higher number of estimators.

>   algorithm: The algorithm to use for the boosting process. It can be either 'SAMME' (default) or 'SAMME.R'.
    'SAMME' uses the original AdaBoost algorithm, while 'SAMME.R' is a variant that can provide better performance if
    the base estimators can predict class probabilities.

>   random_state: Controls the random seed for reproducibility. When you set random_state to a specific value,
    the results will be the same each time you run the code, which is useful for reproducibility.

'''


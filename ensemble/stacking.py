#================================================================ Model Def:
'''
Stacking, short for Stacked Generalization, is an ensemble learning technique that combines multiple base models using a meta-learner. It works by training a meta-learner to make predictions based on the outputs of the base models. Here's a breakdown of how Stacking works and when to use it:

    >   Base Models: Stacking involves training multiple base models on the training data. These base models can be of different types or trained using different algorithms.

    >   Meta-Learner: Stacking trains a meta-learner (also known as a blender or aggregator) on the outputs of the base models. The meta-learner takes the predictions of the base models as input features and learns to make final predictions.

    >   Cross-Validation: Stacking typically uses k-fold cross-validation to train the base models and generate predictions for the meta-learner. This ensures that the meta-learner is trained on diverse sets of data to generalize well.

    >   Level-1 and Level-2 Models: In Stacking, the base models are referred to as level-1 models, and the meta-learner is referred to as the level-2 model. The level-1 models make predictions on the training data, which are then used as features for training the level-2 model.

When to use Stacking:

* Improved Performance: Stacking can often outperform individual base models by leveraging the strengths of multiple algorithms.
* Model Diversity: Stacking combines predictions from diverse base models, which can lead to better generalization and robustness.
* Handling Complex Relationships: Stacking can capture complex relationships between features and the target variable by combining predictions from multiple models.
* Flexibility: Stacking allows you to use different types of base models and meta-learners, providing flexibility in model selection and tuning.
'''

#================================================================ Template:
#====== Importing needed libraries:
from sklearn.ensemble import StackingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#====== Generate some example data (replace this with your actual data):
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

#====== Or you can use the below to generate with your data:
# X = IDM_df.drop(columns=['target_column'])  # Replace 'target_column' with the name of your target column
# y = IDM_df['target_column']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models
base_models = [
    ('decision_tree', DecisionTreeClassifier(random_state=42)),
    ('svm', SVC(random_state=42)),
    ('random_forest', RandomForestClassifier(random_state=42))
]

# Instantiate the Stacking classifier with a meta-learner
stacking_classifier = StackingClassifier(estimators=base_models, final_estimator=RandomForestClassifier(random_state=42))

# Train the classifier
stacking_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = stacking_classifier.predict(X_test)

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
cv_scores = cross_val_score(stacking_classifier, X, y, cv=5)
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

StackingClassifier parameters:

>   estimators: A list of base estimators to be stacked together. Each element is a tuple (name, estimator), where
    'name' is a string identifier for the base estimator and 'estimator' is the base estimator itself.
>   final_estimator: The meta-learner (or final estimator) used to aggregate the predictions of the base models.
    It can be any classifier/regressor that implements the 'fit' and 'predict' methods.
>   cv: The number of folds used for cross-validation when training the base models.
    If an integer is specified, StratifiedKFold is used by default.
>   stack_method: The method used to transform the predictions of the base models for the meta-learner.
    It can be 'auto', 'predict_proba', 'decision_function', or 'predict'. By default, it uses 'auto', which
    selects 'predict_proba' for classifiers and 'predict' for regressors.
>   n_jobs: The number of jobs to run in parallel for both fit and predict. -1 means using all processors.
>   verbose: Controls the verbosity of the output during fitting.
>   passthrough: Whether to use the raw features along with the predictions of the base models for training the meta-learner.
    If False, only the predictions of the base models are used.
>   stratified: Whether to use stratified sampling when splitting the data for cross-validation. Defaults to True.
'''

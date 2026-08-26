#================================================================ Model Def:
'''
Naive Bayes is a family of probabilistic classifiers based on Bayes' theorem with the assumption of independence between features. Despite its simplicity, Naive Bayes is known to perform well in many real-world applications. Here's a breakdown of how Naive Bayes works and when to use it:

    >   Bayes' Theorem: Naive Bayes classifiers apply Bayes' theorem to calculate the probability of a class given a set of features. It assumes that the presence of a particular feature in a class is independent of the presence of other features.

    >   Independence Assumption: The "naive" assumption in Naive Bayes is that all features are conditionally independent given the class. Although this assumption rarely holds true in real-world data, Naive Bayes can still perform well, especially on text classification tasks.

    >   Maximum A Posteriori (MAP) Estimation: Naive Bayes calculates the probability of each class given the input features and selects the class with the highest probability as the predicted class. This is known as Maximum A Posteriori (MAP) estimation.

    >   Laplace Smoothing: To handle unseen features in the test data, Naive Bayes applies Laplace smoothing by adding a small smoothing parameter to the count of each feature.

When to use Naive Bayes:

* Text Classification: Naive Bayes is particularly effective for text classification tasks, such as spam detection and sentiment analysis.
* Simple and Fast: Naive Bayes is computationally efficient and can handle large datasets with high-dimensional feature spaces.
* Low Resource Requirements: Naive Bayes requires minimal training data and parameter tuning, making it suitable for tasks with limited resources.
* Handling Categorical Data: Naive Bayes can handle both numerical and categorical features, making it suitable for a wide range of data types.
* Baseline Model: Naive Bayes serves as a good baseline model for comparison with more complex classifiers, especially when the independence assumption holds reasonably well.
'''

#================================================================ Template:
#====== Importing needed libraries:
from sklearn.naive_bayes import GaussianNB
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

# Instantiate the Naive Bayes classifier
naive_bayes_classifier = GaussianNB()

# Train the classifier
naive_bayes_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = naive_bayes_classifier.predict(X_test)

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
cv_scores = cross_val_score(naive_bayes_classifier, X, y, cv=5)
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

GaussianNB parameters:

Naive Bayes classifiers in scikit-learn implement Gaussian Naive Bayes by default, which assumes that the features follow a Gaussian distribution.

No specific hyperparameters need to be set for GaussianNB.
'''

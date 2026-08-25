#================================================================ Model Def:
'''
Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. It works by finding the optimal hyperplane that separates the data into different classes with the maximum margin. Here's a breakdown of how SVM works and when to use it:

    >   Maximum Margin: SVM aims to find the hyperplane that maximizes the margin between the closest data points from different classes, known as support vectors. This margin allows SVM to have better generalization performance and be more robust to noise in the data.

    >   Kernel Trick: SVM can handle nonlinear decision boundaries by mapping the input features into a higher-dimensional space using kernel functions. Common kernel functions include linear, polynomial, radial basis function (RBF), and sigmoid kernels. The choice of kernel function depends on the problem and the data.

    >   Regularization: SVM uses a regularization parameter (C) to control the trade-off between maximizing the margin and minimizing the classification error. A smaller value of C results in a larger margin but may lead to more misclassifications, while a larger value of C reduces the margin to minimize misclassifications.

    >   Support Vectors: Support vectors are the data points closest to the decision boundary, which influence the position and orientation of the hyperplane. These points are crucial for defining the margin and determining the decision boundary.

When to use SVM:

* Binary and Multiclass Classification: SVM is suitable for both binary and multiclass classification tasks, where the data can be separated into distinct classes.
* Small to Medium-Sized Datasets: SVM performs well on small to medium-sized datasets with a moderate number of features, as it relies on solving a quadratic optimization problem.
* Nonlinear Decision Boundaries: SVM with kernel functions can handle nonlinear decision boundaries, making it suitable for complex classification problems.
* Text Classification: SVM is commonly used for text classification tasks, such as sentiment analysis and document classification, where the input features are often high-dimensional.
'''

#================================================================ Template:
#====== Importing needed libraries:
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#====== Generate some example data (replace this with your actual data):
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

#====== Or you can use the below to generate with your data:
# X = IDM_df.drop(columns=['target_column'])  # Replace 'target_column' with the name of your target column
# y = IDM_df['target_column']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the Support Vector Classifier
svm_classifier = SVC(kernel='rbf', C=1.0, random_state=42)

# Train the model
svm_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_classifier.predict(X_test)

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

SVC parameters:

>   kernel: Specifies the kernel type to be used in the algorithm. Common choices include 'linear', 'poly', 'rbf' (radial basis function), and 'sigmoid'. The 'rbf' kernel is commonly used for non-linear classification problems.
>   C: Regularization parameter. A smaller value of C leads to a softer margin, allowing more misclassifications, while a larger value of C results in a harder margin, potentially leading to overfitting.
>   random_state: Controls the random seed for reproducibility. When you set random_state to a specific value,
    the results will be the same each time you run the code, which is useful for reproducibility.
'''

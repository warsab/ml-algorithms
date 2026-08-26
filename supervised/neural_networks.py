#================================================================ Model Def:
'''
Neural Networks, also known as Artificial Neural Networks (ANN), are a class of machine learning models inspired by the structure and function of the human brain. They consist of interconnected nodes (neurons) organized in layers, where each neuron performs a computation and passes its output to neurons in the next layer. Here's a breakdown of how Neural Networks work and when to use them:

    >   Architecture: Neural Networks consist of an input layer, one or more hidden layers, and an output layer. Each layer contains neurons, and the connections between neurons have associated weights that are adjusted during training.

    >   Activation Function: Each neuron applies an activation function to the weighted sum of its inputs to introduce non-linearity into the model. Common activation functions include ReLU (Rectified Linear Unit), Sigmoid, and Tanh.

    >   Training: Neural Networks are trained using optimization algorithms such as gradient descent to minimize a loss function that measures the difference between predicted and actual outputs. Backpropagation is used to update the weights of the connections between neurons to minimize the loss.

    >   Deep Learning: Deep Neural Networks (DNNs) are Neural Networks with multiple hidden layers. Deep Learning has achieved remarkable success in various fields such as computer vision, natural language processing, and speech recognition.

When to use Neural Networks:

* Complex Relationships: Neural Networks are suitable for tasks where the relationship between input features and the target variable is complex and non-linear.
* Big Data: Neural Networks can handle large datasets with high dimensionality, making them suitable for big data analytics and deep learning applications.
* Feature Engineering: Neural Networks can automatically learn relevant features from raw data, reducing the need for manual feature engineering.
* Image and Speech Recognition: Convolutional Neural Networks (CNNs) are specialized types of Neural Networks commonly used for image recognition tasks, while Recurrent Neural Networks (RNNs) are used for sequential data such as speech and text.
'''

#================================================================ Template:
#====== Importing needed libraries:
from sklearn.neural_network import MLPClassifier
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

# Instantiate the Neural Network classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=42)

# Train the model
mlp_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = mlp_classifier.predict(X_test)

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
cv_scores = cross_val_score(mlp_classifier, X, y, cv=5)
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

MLPClassifier parameters:

>   hidden_layer_sizes: The number of neurons in each hidden layer. You can specify a tuple to define the architecture of the neural network.
>   activation: The activation function for the hidden layers. Common choices include 'relu' (Rectified Linear Unit), 'sigmoid', and 'tanh'.
>   solver: The optimization algorithm to use. 'adam' is a popular choice for large datasets, while 'lbfgs' may be suitable for smaller datasets.
>   alpha: L2 regularization parameter to control overfitting.
>   random_state: Controls the random seed for reproducibility. When you set random_state to a specific value,
    the results will be the same each time you run the code, which is useful for reproducibility.
'''

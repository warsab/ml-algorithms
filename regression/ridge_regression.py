#================================================================ Model Def:
'''
Ridge Regression is a linear regression model that incorporates L2 regularization to penalize large coefficients. It helps prevent overfitting by adding a penalty term to the loss function, which discourages large coefficient values. Here's a breakdown of how Ridge Regression works and when to use it:

    >   L2 Regularization: Ridge Regression adds a regularization term to the linear regression cost function, which penalizes the sum of squared coefficients (L2 norm). The regularization term is controlled by a hyperparameter, alpha, which determines the strength of the regularization.

    >   Shrinkage Effect: The regularization term in Ridge Regression shrinks the coefficients towards zero, reducing their magnitudes. This can help mitigate overfitting, especially when the number of features is large or when multicollinearity is present.

    >   Tuning Alpha: The alpha hyperparameter controls the trade-off between fitting the training data well and keeping the coefficients small. Higher values of alpha increase the regularization strength, resulting in more shrinkage of coefficients.

When to use Ridge Regression:

* Multicollinearity: Ridge Regression is effective when multicollinearity (high correlation between predictors) is present in the data, as it helps stabilize coefficient estimates.
* Regularization: Ridge Regression is suitable when regularization is desired to prevent overfitting and improve model generalization.
* Large Number of Features: Ridge Regression can handle datasets with a large number of features by shrinking less informative features' coefficients towards zero.
* Balanced Regularization: Ridge Regression provides a balanced approach to regularization, as it penalizes large coefficients without setting them exactly to zero (unlike Lasso regression).
'''

#================================================================ Template:
#====== Importing needed libraries:
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#====== Generate some example data (replace this with your actual data):
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

#====== Or you can use the below to generate with your data:
# X = IDM_df.drop(columns=['target_column'])  # Replace 'target_column' with the name of your target column
# y = IDM_df['target_column']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the Ridge Regression model
ridge_regressor = Ridge(alpha=1.0, random_state=42)  # You can adjust alpha

# Train the model
ridge_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = ridge_regressor.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

#================================================================ Notes on Model construction:
'''
Make_regression model:
This is a function provided by scikit-learn that generates a random regression problem.

Parameters:
>   n_samples: It specifies the number of samples to generate. Each sample is a data point.
>   n_features: It specifies the number of features (or dimensions) of each sample. Each feature is a characteristic
    or attribute of the data point.
>   noise: It specifies the standard deviation of the Gaussian noise added to the output.
>   random_state: It controls the random seed for reproducibility. When you set random_state to a specific value,
    the generated data will be the same each time you run the code, which is useful for reproducibility.

Ridge parameters:

>   alpha: The regularization strength. Higher values of alpha increase the penalty for large coefficients,
    which helps prevent overfitting.
>   fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations.
>   normalize: Whether to normalize the features before fitting the model. Normalization can be useful when the features have different scales.
>   solver: The solver algorithm to use. Options are 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', and 'saga'.
>   random_state: Controls the random seed for reproducibility. When you set random_state to a specific value,
    the results will be the same each time you run the code, which is useful for reproducibility.
>   max_iter: The maximum number of iterations for the optimization algorithm.
'''

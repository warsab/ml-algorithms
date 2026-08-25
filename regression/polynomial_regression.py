#================================================================ Model Def:
'''
Polynomial Regression is a form of linear regression in which the relationship between the independent variable (predictor) and the dependent variable (target) is modeled as an nth-degree polynomial. It allows for a nonlinear relationship between the variables by introducing polynomial terms. Here's a breakdown of how Polynomial Regression works and when to use it:

    >   Polynomial Features: Polynomial Regression extends linear regression by adding polynomial features to the dataset. These features are created by raising the original features to different powers, allowing the model to capture nonlinear relationships.

    >   Degree of Polynomial: The degree of the polynomial determines the complexity of the model. Higher-degree polynomials can capture more intricate relationships but are more prone to overfitting, especially with limited data.

    >   Overfitting: Polynomial Regression is susceptible to overfitting, especially with high-degree polynomials. Regularization techniques like Ridge or Lasso regression can help mitigate overfitting by penalizing large coefficients.

    >   Interpretability: While Polynomial Regression can capture complex relationships between variables, the resulting models may be less interpretable than linear models. Interpretability depends on the degree of the polynomial and the number of features.

When to use Polynomial Regression:

* Nonlinear Relationships: Polynomial Regression is useful when the relationship between the predictors and the target variable is nonlinear.
* Flexibility: Polynomial Regression can fit a wide range of functions, making it suitable for modeling diverse relationships.
* Trade-off Between Bias and Variance: By adjusting the degree of the polynomial, you can control the trade-off between bias and variance in the model.
* Interpolation: Polynomial Regression can be used for interpolation tasks, where you need to estimate values between observed data points.
'''

#================================================================ Template:
#====== Importing needed libraries:
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#====== Generate some example data (replace this with your actual data):
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

#====== Or you can use the below to generate with your data:
# X = IDM_df[['feature_column']]  # Replace 'feature_column' with the name of your feature column
# y = IDM_df['target_column']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the degree of the polynomial
degree = 3

# Instantiate PolynomialFeatures and transform the features
poly_features = PolynomialFeatures(degree=degree)
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test)

# Instantiate the Linear Regression model
poly_linear_regressor = LinearRegression()

# Train the model
poly_linear_regressor.fit(X_poly_train, y_train)

# Predict on the test set
y_pred = poly_linear_regressor.predict(X_poly_test)

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

PolynomialFeatures parameters:

>   degree: The degree of the polynomial features. It determines the maximum degree of the polynomial terms to be included.
>   interaction_only: Whether to include only interaction features (products of distinct features), excluding polynomial terms.
>   include_bias: Whether to include a bias column (intercept) in the transformed features.

LinearRegression parameters:

>   fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations.
>   normalize: Whether to normalize the features before fitting the model. Normalization can be useful when the features have different scales.
>   n_jobs: The number of jobs to run in parallel for both fit and predict. -1 means using all processors.
>   copy_X: Whether to copy X before fitting the model. If False, it may overwrite the input data.
'''

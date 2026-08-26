#================================================================ Model Def:
'''
Multiple Linear Regression is a linear regression model that predicts the target variable based on multiple predictor variables. It assumes a linear relationship between the predictors and the target variable. Here's a breakdown of how Multiple Linear Regression works and when to use it:

    >   Linear Relationship: Multiple Linear Regression models the relationship between the predictors (independent variables) and the target variable (dependent variable) using a linear equation. The model assumes that the target variable can be expressed as a weighted sum of the predictors, plus an intercept term.

    >   Ordinary Least Squares (OLS): Multiple Linear Regression estimates the coefficients (weights) of the linear equation using the Ordinary Least Squares method. OLS minimizes the sum of the squared differences between the observed and predicted values of the target variable.

    >   Interpretability: Multiple Linear Regression provides interpretable coefficients that represent the effect of each predictor variable on the target variable, holding other variables constant. This makes it easy to understand the relative importance of different features.

    >   Assumptions: Multiple Linear Regression assumes that the relationship between the predictors and the target variable is linear, and that the residuals (the differences between observed and predicted values) are normally distributed and homoscedastic (constant variance).

When to use Multiple Linear Regression:

* Relationship Exploration: Multiple Linear Regression is useful for exploring the relationship between multiple predictors and a target variable.
* Predictive Modeling: Multiple Linear Regression can be used for prediction tasks when the assumptions of linearity and constant variance hold reasonably well.
* Feature Importance: Multiple Linear Regression provides coefficients that indicate the importance of each predictor variable in predicting the target variable.
* Interpretability: Multiple Linear Regression produces a simple and interpretable model that can be easily understood by non-experts.
'''

#================================================================ Template:
#====== Importing needed libraries:
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

#====== Generate some example data (replace this with your actual data):
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

#====== Or you can use the below to generate with your data:
# X = IDM_df.drop(columns=['target_column'])  # Replace 'target_column' with the name of your target column
# y = IDM_df['target_column']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the Linear Regression model
linear_regressor = LinearRegression()

# Train the model
linear_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = linear_regressor.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#====== MSE on its own is hard to interpret, because it is in squared units and
#       has no fixed scale. RMSE is back in the units of the target, MAE is less
#       sensitive to outliers, and R2 says how much variance the model explains:
print(f"MSE  (mean squared error):   {mse:.4f}")
print(f"RMSE (root mean sq. error):  {rmse:.4f}")
print(f"MAE  (mean absolute error):  {mae:.4f}")
print(f"R2   (variance explained):   {r2:.4f}")

#====== Cross-validation across 5 folds, scored by R2:
cv_scores = cross_val_score(linear_regressor, X, y, cv=5, scoring="r2")
print(f"\n5-fold CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

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

LinearRegression parameters:

>   fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations.
>   normalize: Whether to normalize the features before fitting the model. Normalization can be useful when the features have different scales.
>   n_jobs: The number of jobs to run in parallel for both fit and predict. -1 means using all processors.
>   copy_X: Whether to copy X before fitting the model. If False, it may overwrite the input data.
'''

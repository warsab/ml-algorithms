#================================================================ Model Def:
'''
Linear Regression is a linear approach to modeling the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship between the input features and the target variable, where the target variable is modeled as a weighted sum of the input features plus an intercept term.
Here's a breakdown of how Linear Regression works and when to use it:

    >   Linearity: Linear Regression models the relationship between the input features and the target variable as a linear function. It assumes that the relationship can be represented by a straight line in a multidimensional space.

    >   Coefficients and Intercept: In Linear Regression, the coefficients (or weights) represent the slope of the line for each input feature, indicating the change in the target variable per unit change in the feature. The intercept term represents the value of the target variable when all input features are zero.

    >   Ordinary Least Squares: Linear Regression is often trained using the Ordinary Least Squares (OLS) method, which minimizes the sum of squared differences between the observed and predicted values of the target variable.

    >   Assumptions: Linear Regression assumes that the residuals (the differences between observed and predicted values) are normally distributed, and that there is a linear relationship between the input features and the target variable.

When to use Linear Regression:

* Simple Relationships: Linear Regression is suitable for modeling simple relationships between the input features and the target variable, where the relationship can be adequately captured by a straight line.
* Interpretability: Linear Regression provides interpretable coefficients, allowing for easy interpretation of the impact of each input feature on the target variable.
* Prediction: Linear Regression can be used for prediction tasks when the relationship between the input features and the target variable is approximately linear.
* Baseline Model: Linear Regression serves as a simple baseline model for regression tasks, providing a reference for more complex algorithms.
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

LinearRegression does not have any hyperparameters to tune during model instantiation. However, you can adjust various parameters such as normalization, fit_intercept, and copy_X if necessary.
'''

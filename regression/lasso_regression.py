#================================================================ Model Def:
'''
Lasso Regression, short for Least Absolute Shrinkage and Selection Operator Regression, is a linear regression model that incorporates L1 regularization to penalize large coefficients. It helps prevent overfitting by adding a penalty term to the loss function, which encourages sparsity in the coefficient values. Here's a breakdown of how Lasso Regression works and when to use it:

    >   L1 Regularization: Lasso Regression adds a regularization term to the linear regression cost function, which penalizes the sum of absolute values of coefficients (L1 norm). The regularization term is controlled by a hyperparameter, alpha, which determines the strength of the regularization.

    >   Feature Selection: Lasso Regression induces sparsity in the coefficient values by setting some coefficients exactly to zero. This results in feature selection, as features with zero coefficients are effectively ignored in the model. Lasso can be useful when you suspect that only a subset of features are relevant predictors of the target variable.

    >   Tuning Alpha: The alpha hyperparameter controls the trade-off between fitting the training data well and keeping the coefficients small. Higher values of alpha increase the regularization strength, resulting in more coefficients being set to zero.

When to use Lasso Regression:

* Feature Selection: Lasso Regression is effective when feature selection is desired, as it automatically selects a subset of relevant features by setting irrelevant coefficients to zero.
* Sparse Solutions: Lasso Regression produces sparse solutions with fewer non-zero coefficients, which can improve model interpretability and reduce overfitting.
* Regularization: Lasso Regression is suitable when regularization is desired to prevent overfitting and improve model generalization.
'''

#================================================================ Template:
#====== Importing needed libraries:
from sklearn.linear_model import Lasso
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

# Instantiate the Lasso Regression model
lasso_regressor = Lasso(alpha=1.0, random_state=42)  # You can adjust alpha

# Train the model
lasso_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = lasso_regressor.predict(X_test)

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
cv_scores = cross_val_score(lasso_regressor, X, y, cv=5, scoring="r2")
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

Lasso parameters:

>   alpha: The regularization strength. Higher values of alpha increase the penalty for large coefficients,
    which helps prevent overfitting.
>   fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations.
>   normalize: Whether to normalize the features before fitting the model. Normalization can be useful when the features have different scales.
>   precompute: Whether to use a precomputed Gram matrix to speed up calculations. 'auto', True, or False.
>   random_state: Controls the random seed for reproducibility. When you set random_state to a specific value,
    the results will be the same each time you run the code, which is useful for reproducibility.
>   max_iter: The maximum number of iterations for the optimization algorithm.
'''

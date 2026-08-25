#================================================================ Model Def:
'''
ElasticNet is a linear regression model that combines the penalties of the Lasso (L1) and Ridge (L2) regression methods. It is useful when there are multiple features that are correlated with one another. Here's a breakdown of how ElasticNet works and when to use it:

    >   Regularization: ElasticNet adds both L1 and L2 penalties to the linear regression cost function. The L1 penalty encourages sparsity by shrinking some coefficients to zero, while the L2 penalty encourages small but non-zero coefficients.

    >   Hyperparameter Tuning: ElasticNet has two hyperparameters, alpha and l1_ratio, which control the strength of the regularization penalties. Alpha determines the overall strength of the regularization, while l1_ratio determines the balance between L1 and L2 penalties.

    >   Feature Selection: ElasticNet can perform feature selection by setting some coefficients to zero, effectively ignoring those features in the final model. This can help reduce overfitting and improve model interpretability.

When to use ElasticNet:

* High-Dimensional Data: ElasticNet is suitable for datasets with a large number of features, especially when many of these features are correlated.
* Feature Selection: ElasticNet can automatically perform feature selection by shrinking irrelevant features to zero.
* Balanced Regularization: ElasticNet provides a balanced regularization approach by combining the strengths of Lasso and Ridge regression.
* Interpretability: The sparsity induced by ElasticNet can lead to more interpretable models by selecting only the most relevant features.
'''

#================================================================ Template:
#====== Importing needed libraries:
from sklearn.linear_model import ElasticNet
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

# Instantiate the ElasticNet regressor
elasticnet_regressor = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)  # You can adjust alpha and l1_ratio

# Train the regressor
elasticnet_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = elasticnet_regressor.predict(X_test)

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

ElasticNet parameters:

>   alpha: The regularization strength. Higher values of alpha increase the penalty for large coefficients, which can help prevent overfitting.
>   l1_ratio: The mixing parameter that determines the balance between L1 (Lasso) and L2 (Ridge) penalties. A value of 0 corresponds to Ridge regression, while a value of 1 corresponds to Lasso regression.
>   fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations.
>   normalize: Whether to normalize the features before fitting the model. Normalization can be useful when the features have different scales.
>   max_iter: The maximum number of iterations for the optimization algorithm.
>   random_state: Controls the random seed for reproducibility. When you set random_state to a specific value,
    the results will be the same each time you run the code, which is useful for reproducibility.
>   selection: The method used to select the features during fitting. It can be either 'cyclic' or 'random'. The default is 'cyclic'.
'''

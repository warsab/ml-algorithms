#================================================================ Model Def:
'''
Principal Component Analysis (PCA) is a dimensionality reduction technique used to transform high-dimensional data into a lower-dimensional space while preserving most of the variability in the data. It identifies the directions (principal components) in which the data varies the most and projects the data onto these components. Here's a breakdown of how PCA works and when to use it:

    >   Dimensionality Reduction: PCA reduces the dimensionality of the data by transforming it into a new coordinate system defined by the principal components. These components are orthogonal to each other and capture different amounts of variability in the data.

    >   Variance Maximization: PCA seeks to maximize the variance of the data along the principal components. The first principal component captures the most variance, followed by the second, and so on. By selecting a subset of the principal components, PCA retains most of the variability in the data while reducing its dimensionality.

    >   Interpretability: PCA provides interpretable results, as the principal components represent meaningful directions in the data space. These components can often be interpreted in terms of the original features, making it easier to understand the underlying structure of the data.

    >   Data Preprocessing: PCA is often used for data preprocessing before applying other machine learning algorithms. It can help reduce the computational complexity of models, remove redundant features, and mitigate the curse of dimensionality.

When to use PCA:

* High-Dimensional Data: PCA is suitable for datasets with a large number of features, where the dimensionality of the data needs to be reduced to improve computational efficiency or address multicollinearity issues.
* Visualization: PCA can be used to visualize high-dimensional data in a lower-dimensional space (e.g., 2D or 3D) while preserving most of the variability. This is useful for exploratory data analysis and data visualization.
* Noise Reduction: PCA can help remove noise and extract the underlying structure of the data by focusing on the principal components with the highest variance.
* Feature Engineering: PCA can be used for feature engineering by creating new composite features that capture the most important patterns in the data.
'''

#================================================================ Template:
#====== Importing needed libraries:
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

#====== Generate some example data (replace this with your actual data):
X, _ = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

#====== Or you can use the below to generate with your data:
# X = IDM_df.drop(columns=['target_column']).values  # Replace 'target_column' with the name of your target column

# Instantiate the PCA model
pca_model = PCA(n_components=2, random_state=42)

# Fit the model to the data and transform the data
X_pca = pca_model.fit_transform(X)

# Visualize the transformed data
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', alpha=0.5)
plt.title('PCA Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

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
>   random_state: It controls the random seed for reproducibility. When you set random_state to a specific value,
    the generated data will be the same each time you run the code, which is useful for reproducibility.

PCA parameters:

>   n_components: The number of principal components to retain. It determines the dimensionality of the transformed data.
>   random_state: Controls the random seed for reproducibility. When you set random_state to a specific value,
    the results will be the same each time you run the code, which is useful for reproducibility.
'''

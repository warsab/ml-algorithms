#================================================================ Model Def:
'''
KMeans Clustering is a popular clustering algorithm used for partitioning data into K distinct clusters. It aims to minimize the within-cluster variance, where each cluster is represented by its centroid (the mean of the data points in the cluster). Here's a breakdown of how KMeans Clustering works and when to use it:

    >   Centroid-Based Clustering: KMeans Clustering assigns each data point to the nearest centroid, forming K clusters. The centroids are initialized randomly, and the algorithm iteratively updates them by minimizing the sum of squared distances between each data point and its nearest centroid.

    >   Number of Clusters (K): The number of clusters (K) is a hyperparameter that needs to be specified in advance. Choosing an appropriate value for K can be challenging and often requires domain knowledge or experimentation. Techniques such as the elbow method and silhouette analysis can help determine the optimal value of K.

    >   Convergence Criteria: KMeans Clustering converges when either the centroids stop changing significantly between iterations or a maximum number of iterations is reached. The algorithm may converge to a local minimum, so it is recommended to run the algorithm multiple times with different initializations and select the solution with the lowest within-cluster variance.

    >   Scalability: KMeans Clustering is computationally efficient and scalable to large datasets, making it suitable for clustering tasks with a large number of data points and features.

When to use KMeans Clustering:

* Well-Separated Clusters: KMeans Clustering works well when the clusters are well-separated and have similar sizes and densities.
* Predefined Number of Clusters: KMeans Clustering requires the number of clusters (K) to be specified in advance, making it suitable for tasks where the desired number of clusters is known or can be estimated.
* Numeric Data: KMeans Clustering is designed for numeric data and may not perform well on categorical or mixed-type data.
* Interpretability: KMeans Clustering produces easily interpretable results, with each cluster represented by its centroid, making it suitable for exploratory data analysis and understanding the structure of the data.
'''

#================================================================ Template:
#====== Importing needed libraries:
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

#====== Generate some example data (replace this with your actual data):
X, _ = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=42)

#====== Or you can use the below to generate with your data:
# X = IDM_df[['feature1', 'feature2']].values  # Replace 'feature1' and 'feature2' with the names of your features

# Instantiate the KMeans model
kmeans_model = KMeans(n_clusters=3, random_state=42)

# Fit the model to the data
kmeans_model.fit(X)

# Visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=kmeans_model.labels_, cmap='viridis', s=50, alpha=0.7)
plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], marker='o', c='red', s=200, edgecolors='k', label='Centroids')
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

#================================================================ Notes on Model construction:
'''
Make_blobs model:
This is a function provided by scikit-learn that generates synthetic clusters of points.

Parameters:
>   n_samples: It specifies the total number of points equally divided among clusters.
>   centers: It specifies the number of centers to generate, or the fixed center locations.
>   n_features: It specifies the number of features for each sample.
>   random_state: It controls the random seed for reproducibility.

KMeans parameters:

>   n_clusters: The number of clusters (K) to form as well as the number of centroids to generate.
>   init: The method used to initialize the centroids. By default, it uses 'k-means++', which selects initial centroids in a smart way to speed up convergence.
>   max_iter: The maximum number of iterations for the algorithm to converge. If the algorithm does not converge within this number of iterations, it stops.
>   random_state: Controls the random seed for reproducibility. When you set random_state to a specific value,
    the results will be the same each time you run the code, which is useful for reproducibility.
'''

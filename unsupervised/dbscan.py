#================================================================ Model Def:
'''
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a popular clustering algorithm used for discovering clusters of varying shapes and sizes in spatial data. Unlike k-means, DBSCAN does not require the number of clusters to be specified in advance and can identify outliers as noise. Here's a breakdown of how DBSCAN works and when to use it:

    >   Density-Based Clustering: DBSCAN clusters the data based on density, where a cluster is defined as a dense region of data points separated by regions of lower density. It works well for datasets with irregularly shaped clusters and varying cluster densities.

    >   Core Points and Neighborhoods: DBSCAN identifies core points, which have a minimum number of neighboring points within a specified distance (eps). Points that are within the neighborhood of a core point but are not core points themselves are considered border points. Points that do not belong to any cluster are classified as noise points.

    >   Parameter Tuning: DBSCAN requires two parameters to be specified: eps (the maximum distance between two samples for one to be considered as in the neighborhood of the other) and min_samples (the number of samples in a neighborhood for a point to be considered as a core point). Tuning these parameters can significantly impact the clustering results.

    >   Handling Outliers: DBSCAN can automatically identify outliers as noise points that do not belong to any cluster. This makes it suitable for tasks where noise detection is important, such as anomaly detection and outlier analysis.

When to use DBSCAN:

* Arbitrary Cluster Shapes: DBSCAN is suitable for datasets with arbitrary cluster shapes and varying densities, where traditional clustering algorithms like k-means may struggle.
* Noise Handling: DBSCAN can identify and handle outliers as noise points, making it robust to noisy data and suitable for tasks where noise detection is essential.
* Large Datasets: DBSCAN is scalable to large datasets due to its efficient algorithmic design, making it suitable for clustering in high-dimensional spaces.
* No Need for Predefined Number of Clusters: DBSCAN does not require the number of clusters to be specified in advance, making it suitable for exploratory data analysis and situations where the number of clusters is unknown.
'''

#================================================================ Template:
#====== Importing needed libraries:
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

#====== Generate some example data (replace this with your actual data):
X, _ = make_classification(n_samples=1000, n_features=2, n_classes=3, n_clusters_per_class=1, random_state=42)

#====== Or you can use the below to generate with your data:
# X = IDM_df[['feature1', 'feature2']].values  # Replace 'feature1' and 'feature2' with the names of your features

# Instantiate the DBSCAN model
dbscan_model = DBSCAN(eps=0.3, min_samples=10)

# Fit the model to the data
dbscan_model.fit(X)

# Visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=dbscan_model.labels_, cmap='viridis', s=50, alpha=0.7)
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster Label')
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
>   n_clusters_per_class: It specifies the number of clusters per class. 
    This parameter controls the separation of the classes. Higher values lead to more separated clusters.
>   random_state: It controls the random seed for reproducibility. When you set random_state to a specific value,
    the generated data will be the same each time you run the code, which is useful for reproducibility.

DBSCAN parameters:

>   eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other. This parameter influences the connectivity of the clusters.
>   min_samples: The number of samples in a neighborhood for a point to be considered as a core point. Increasing min_samples can lead to more conservative clustering with fewer clusters.
'''

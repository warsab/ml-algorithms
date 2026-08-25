#================================================================ Model Def:
'''
Hierarchical Clustering is a clustering algorithm that builds a hierarchy of clusters by recursively merging or splitting data points based on their similarity or dissimilarity. It does not require the number of clusters to be specified in advance and can produce a dendrogram to visualize the hierarchical structure of the clusters. Here's a breakdown of how Hierarchical Clustering works and when to use it:

    >   Agglomerative and Divisive Clustering: Hierarchical Clustering can be performed using two main approaches: agglomerative and divisive. Agglomerative clustering starts with each data point as a separate cluster and merges similar clusters iteratively until only one cluster remains. Divisive clustering starts with all data points in a single cluster and splits the clusters recursively until each data point is in its own cluster.

    >   Distance Metrics: Hierarchical Clustering requires a distance or similarity metric to measure the distance between data points. Common distance metrics include Euclidean distance, Manhattan distance, and cosine similarity. The choice of distance metric depends on the nature of the data and the problem domain.

    >   Dendrogram Visualization: Hierarchical Clustering produces a dendrogram, which is a tree-like diagram that illustrates the arrangement of the clusters at each iteration of the algorithm. The dendrogram can be used to determine the optimal number of clusters by visually inspecting the structure and identifying significant merges or splits.

    >   Interpretability: Hierarchical Clustering provides an interpretable hierarchical structure of the data, making it suitable for exploratory data analysis and gaining insights into the relationships between data points.

When to use Hierarchical Clustering:

* Unknown Number of Clusters: Hierarchical Clustering does not require the number of clusters to be specified in advance, making it suitable for datasets where the optimal number of clusters is unknown.
* Interpretable Results: Hierarchical Clustering produces a dendrogram that visualizes the hierarchical structure of the clusters, allowing analysts to interpret the results and understand the relationships between clusters.
* Small to Medium-Sized Datasets: Hierarchical Clustering is suitable for small to medium-sized datasets with moderate to high dimensionality, as it can become computationally expensive for large datasets.
* Relationship Discovery: Hierarchical Clustering can reveal hierarchical relationships between clusters and subclusters, providing insights into the underlying structure of the data.
'''

#================================================================ Template:
#====== Importing needed libraries:
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

#====== Generate some example data (replace this with your actual data):
X, _ = make_classification(n_samples=1000, n_features=2, n_classes=3, n_clusters_per_class=1, random_state=42)

#====== Or you can use the below to generate with your data:
# X = IDM_df[['feature1', 'feature2']].values  # Replace 'feature1' and 'feature2' with the names of your features

# Compute the hierarchical clustering
Z = linkage(X, method='ward')

# Visualize the dendrogram
plt.figure(figsize=(12, 6))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data point index')
plt.ylabel('Distance')
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

Linkage method ('method' parameter):

>   method: The linkage criterion to use for merging clusters. Common methods include 'ward', 'single', 'complete', 'average', etc. 'ward' minimizes the variance of the clusters being merged and is suitable for most datasets.
'''

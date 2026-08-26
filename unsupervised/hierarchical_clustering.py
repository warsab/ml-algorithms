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
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

#====== Generate some example data (replace this with your actual data):
X, _ = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=0.60, random_state=42)

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

#====== Evaluate the clustering:
# The dendrogram shows the full merge history; cutting it at k groups gives labels
# you can actually score.
n_clusters = 3
labels = fcluster(Z, t=n_clusters, criterion="maxclust")

print(f"Clusters requested: {n_clusters}")
for cluster_id in sorted(set(labels)):
    print(f"  cluster {cluster_id}: {list(labels).count(cluster_id)} points")

# Silhouette score rates how well each point sits in its cluster, from -1 to 1.
print(f"Silhouette score: {silhouette_score(X, labels):.4f}")


#================================================================ Notes on Model construction:
'''
Make_blobs model:
This is a function provided by scikit-learn that generates isotropic Gaussian blobs - the
natural fit for demonstrating a clustering algorithm, since the data genuinely contains
groups to be found.

(make_classification is the equivalent for supervised problems. It is the wrong tool here:
it builds a labelled classification problem rather than natural clusters, and it rejects
n_features=2 outright because its default n_informative + n_redundant already sum to 4.)

Parameters:
>   n_samples: It specifies the total number of points, divided roughly equally among the clusters.
>   centers: It specifies how many cluster centres to generate, or their fixed locations.
>   n_features: It specifies the number of features (or dimensions) of each sample. Each feature is a characteristic
    or attribute of the data point.
>   cluster_std: The standard deviation of each cluster. Smaller values give tighter, better
    separated blobs; larger values make the clusters overlap and become harder to recover.
>   random_state: It controls the random seed for reproducibility. When you set random_state to a specific value,
    the generated data will be the same each time you run the code, which is useful for reproducibility.

Linkage method ('method' parameter):

>   method: The linkage criterion to use for merging clusters. Common methods include 'ward', 'single', 'complete', 'average', etc. 'ward' minimizes the variance of the clusters being merged and is suitable for most datasets.
'''

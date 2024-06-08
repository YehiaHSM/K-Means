from math import sqrt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Class representing each data item
class DataItem:
    def __init__(self, item):
        self.features = item
        self.clusterId = -1  # Initially, no cluster is assigned

# Class representing each cluster
class Cluster:
    def __init__(self, id, centroid):
        self.centroid = centroid  # Initial centroid of the cluster
        self.data = []  # Data points belonging to this cluster
        self.id = id  # Cluster ID

    def update(self, clusterData):
        # Update the cluster with new data points and recalculate the centroid
        self.data = []
        for item in clusterData:
            self.data.append(item.features)
        self.centroid = np.average(self.data, axis=0)  # New centroid is the mean of the data points

# Class for calculating similarity distances
class SimilarityDistance:
    def __init__(self):
        pass

    def euclidean_distance(self, p1, p2):
        # Calculate Euclidean distance between two points
        sum = 0
        for i in range(len(p1)):
            sum += (p1[i] - p2[i]) ** 2
        return sqrt(sum)

# Class for custom K-Means clustering
class Clustering_kmeans:
    def __init__(self, data, k, noOfIterations):
        self.data = data  # Data to be clustered
        self.k = k  # Number of clusters
        self.distance = SimilarityDistance()  # Distance metric
        self.noOfIterations = noOfIterations  # Number of iterations for the algorithm

    def initClusters(self):
        # Initialize clusters with first k data points as centroids
        self.clusters = []
        for i in range(self.k):
            self.clusters.append(Cluster(i, self.data[i*50].features))

    def getClusters(self):
        # Perform clustering
        self.initClusters()
        for iteration in range(self.noOfIterations):
            for item in self.data:
                minDistance = 999999
                for cluster in self.clusters:
                    clusterDistance = self.distance.euclidean_distance(cluster.centroid, item.features)
                    if clusterDistance < minDistance:
                        item.clusterId = cluster.id  # Assign item to the nearest cluster
                        minDistance = clusterDistance
                # Get all data points assigned to the current cluster
                clusterData = [x for x in self.data if x.clusterId == item.clusterId]
                self.clusters[item.clusterId].update(clusterData)  # Update the cluster with new data points

        return self.clusters  # Return the final clusters

# Function to load the Iris dataset
def LoadData():
    dataset = load_iris()
    data = []
    for item in dataset['data']:
        dataItem = DataItem(item)  # Wrap each data point in a DataItem
        data.append(dataItem)
    return data

# Function to perform clustering using built-in KMeans
def kmeans_builtIn():
    iris = load_iris()
    X = iris.data
    y = iris.target
    km = KMeans(n_clusters=3)
    km.fit(X)
    centers = km.cluster_centers_
    print(centers)

# Function to perform clustering using custom KMeans implementation
def kmeans_custom():
    dataset = LoadData()
    clustering = Clustering_kmeans(dataset, 3, 50)
    clusters = clustering.getClusters()
    for cluster in clusters:
        print(cluster.centroid)

print('built in')
kmeans_builtIn()  # Run built-in KMeans
print('custom')
kmeans_custom()  # Run custom KMeans

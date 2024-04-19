import math
import multiprocessing as mp
import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


def processing_points(args):
    """
    Calculate the average distance of points in a cluster to the cluster center.
    """
    points, center = args
    res = 0
    for p in points:
        res += math.sqrt((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2)
    return res / len(points)


def plot(data, centers):
    """
    Visualize the clustering results using a scatter plot.
    """
    plt.figure(figsize=(12, 9))
    plt.scatter(data["Platelets_mean"], data["Temp_mean"], s=12, c=data["Cluster"], linewidths=0.3, edgecolors="black")
    plt.scatter(centers[:, 0], centers[:, 1], s=100, c="red", edgecolors="black")
    plt.show()


def calculating_index(df, k, n_threads):
    """
    Calculate the Davies-Bouldin index for a given clustering solution.
    """
    centers = kmeans.cluster_centers_
    clusters = range(k)
    cluster_points = []
    for cluster in clusters:
        cluster_points.append(df[df["Cluster"] == cluster].drop(columns="Cluster").to_numpy())

    with mp.Pool(n_threads) as p:
        cluster_diameters = p.map(processing_points, zip(cluster_points, centers))

    center_distances = euclidean_distances(centers, centers)
    db_index = 0
    for cluster_i in clusters:
        max_index_i = 0
        for cluster_j in clusters:
            if cluster_i != cluster_j:
                index_i_j = (cluster_diameters[cluster_i] + cluster_diameters[cluster_j]) / center_distances[cluster_i][
                    cluster_j]
                if index_i_j > max_index_i:
                    max_index_i = index_i_j
        db_index += max_index_i
    db_index /= k
    return db_index


if __name__ == "__main__":
    # Load and preprocess the dataset
    df = pd.read_csv("BD-patients.csv")[["Platelets_mean", "Temp_mean"]]
    df = df.dropna()
    df = (df - df.mean(axis=0)) / df.std(axis=0)
    df = df[(abs(df["Platelets_mean"]) < 10) & (abs(df["Temp_mean"]) < 10)]

    # Define the range of threads and sample sizes to test
    test_threads = list(range(2, 17, 2))
    test_size = [1000, 3000, 5000]

    # Iterate over different values of K and sample sizes
    for k in range(3, 6):
        for size in test_size:
            new_df = df[:size].copy()

            # Perform K-means clustering
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
            new_df["Cluster"] = kmeans.fit_predict(new_df)

            # Visualize the clustering results
            plot(new_df, kmeans.cluster_centers_)

            # Evaluate the performance for different numbers of threads
            for n_threads in test_threads:
                start = time.perf_counter()
                db_index = calculating_index(new_df, k, n_threads)
                finish = time.perf_counter()
                print(
                    f"Clusters: {k}, sample size: {size}, threads: {n_threads}, time {round(finish - start, 3)}, index: {round(db_index, 5)}")
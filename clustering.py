import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.cluster.hierarchy import dendrogram
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import squareform, pdist, cdist, cosine
from numpy.linalg import eigh
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from visualize import visualize_affinity_matrix_refinement


def construct_affinity_matrix(embeddings):
    """
    Constructs the affinity matrix from segment embeddings.

    Args:
        embeddings: A NumPy array of shape (n_segments, embedding_dim).

    Returns:
        The affinity matrix as a NumPy array.
    """

    n_segments = embeddings.shape[0]
    A = np.zeros((n_segments, n_segments))

    # Calculate cosine similarity (and hence, cosine distance)
    for i in range(n_segments):
        for j in range(n_segments):
            if i != j:
                A[i, j] = 1 - cosine(embeddings[i], embeddings[j]) / 2
            else:
                # Avoid calculating self-similarity
                A[i, j] = -1  # Placeholder, will be replaced with max

    # Replace diagonal with max values
    np.fill_diagonal(A, A.max(axis=1))
    return A


def spectral_clustering(embeddings, sigma=0.5, percentile=95, n_clusters=None, max_clusters=18, visualize=True,
                        visualizeCluster=True, random_state=0):
    n_clusters, new_embeddings = embedding_refinement(embeddings, max_clusters, n_clusters, percentile, sigma, visualize,
                                                      visualizeCluster)

    # Plot the original, unclustered data
    if visualizeCluster and new_embeddings.shape[1] == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(new_embeddings[:, 0], new_embeddings[:, 1], s=50)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title("Original Data")
        plt.show()

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=random_state)
    labels = kmeans.fit_predict(new_embeddings)

    if visualizeCluster and new_embeddings.shape[1] == 2:
        # Plot for clustered data
        plt.figure(figsize=(8, 6))
        plt.scatter(new_embeddings[:, 0], new_embeddings[:, 1], c=labels, s=50, cmap='viridis')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='*',
                    label='Centroids')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f"Spectral Clustering with {n_clusters} Clusters")
        plt.legend()
        plt.show()
    elif new_embeddings.shape[1] != 2:
        print("Embeddings were not reduced to 2 dimensions for visualization. Skipping visualization.")

    return labels




def embedding_refinement(embeddings, max_clusters, n_clusters, percentile, sigma, visualize, visualizeCluster):
    # visualise the embeddings using pca
    if visualizeCluster:
        pca = PCA(n_components=2)
        embeddings = pca.fit_transform(embeddings)
        plt.scatter(embeddings[:, 0], embeddings[:, 1])
        plt.title("PCA of Embeddings")
        plt.show()
    # Step 1: Construct the affinity matrix A based on cosine similarity
    A = construct_affinity_matrix(embeddings)
    if visualize:
        visualize_affinity_matrix_refinement(A, "Original Cosine Affinity")
    # Step 2a: Gaussian Blur (optional step)
    A = gaussian_filter(A, sigma=sigma)
    if visualize:
        visualize_affinity_matrix_refinement(A, "Gaussian Blur", sigma=sigma)
    # Step 2b: Row-wise Thresholding
    for i in range(A.shape[0]):
        threshold = np.percentile(A[i], percentile)
        A[i, A[i] < threshold] *= 0.001
        # A[i, A[i] < threshold] = 0
    if visualize:
        visualize_affinity_matrix_refinement(A, "Row-wise Thresholding", percentile=percentile)
    # Step 2c: Symmetrization
    A = np.maximum(A, A.T)
    if visualize:
        visualize_affinity_matrix_refinement(A, "Symmetrized")
    # Step 2d: Diffusion
    A = np.dot(A, A.T)
    if visualize:
        visualize_affinity_matrix_refinement(A, "Diffusion")
    # Step 2e: Row-wise Max Normalization
    A = A / A.max(axis=1)[:, np.newaxis]
    if visualize:
        visualize_affinity_matrix_refinement(A, "Row-wise Max Normalized")
    # Step 3: Eigen-Decomposition
    eigenvalues, eigenvectors = eigh(A)
    eigenvalues = eigenvalues[::-1]  # Reverse to have in descending order
    eigenvectors = eigenvectors[:, ::-1]
    # Debugging: Print eigenvalues to understand the eigen-gap
    # print("Eigenvalues:", eigenvalues)
    # Determine number of clusters using the maximal eigen-gap if not given
    if n_clusters is None:
        # Calculate the eigen-gap with lambda(k) / lambda(k+1) where k is the index and lambda is the eigenvalue
        eigen_gaps = np.abs(eigenvalues[:-1] / eigenvalues[1:])
        # DEBUG PRINT EIGEN GAPS
        # print("Eigen Gaps:", eigen_gaps)

        # reduce the eigen gaps to the first max_clusters
        eigen_gaps = eigen_gaps[1:max_clusters + 1]

        n_clusters = np.argmax(eigen_gaps) + 1
        print(f"Number of clusters: {n_clusters}")
    # Step 4: Replace segment embeddings with dimensions from the largest eigen-vectors
    new_embeddings = eigenvectors[:, :n_clusters]
    # Dimensionality reduction for visualization (if needed)
    if new_embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        new_embeddings = pca.fit_transform(new_embeddings)
    return n_clusters, new_embeddings


def offline_kmeans(embeddings, max_clusters=10, random_state=0, sigma=0.5, percentile=95, visualize=True):
    """Performs K-means clustering with elbow method for optimal k."""
    # n_clusters, new_embeddings = embedding_refinement(embeddings, max_clusters, None, percentile, sigma, False, False)
    # Step 1: Determine the optimal number of clusters (k) using the elbow method
    distortions = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=random_state)
        kmeans.fit(embeddings)
        distortions.append(kmeans.inertia_)  # Inertia is the sum of squared distances

    # Find the elbow point (hopefully)
    silhouettes = []
    for k in range(2, max_clusters):  # Try different k values (adjust the range as needed)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(embeddings)
        silhouettes.append(silhouette_score(embeddings, kmeans.labels_))

    optimal_k = np.argmax(silhouettes) + 2  # Add 2 because the range started from 2

    # Step 2: K-means clustering with the optimal k
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=random_state)
    labels = kmeans.fit_predict(embeddings)

    # Step 3 (Optional): Visualize the clusters
    if embeddings.shape[1] > 1:
        plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, s=50, cmap='viridis')
    else:
        plt.scatter(embeddings[:, 0], np.zeros_like(embeddings[:, 0]), c=labels, s=50, cmap='viridis')
        if kmeans.cluster_centers_.shape[1] > 1:
            plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='*',
                        label='Centroids')
        else:
            plt.scatter(kmeans.cluster_centers_[:, 0], np.zeros_like(kmeans.cluster_centers_[:, 0]), s=200, c='red',
                        marker='*', label='Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f"K-means Clustering with {optimal_k} Clusters")
    plt.legend()
    plt.show()

    return labels


def dbscan_clustering(embeddings, eps=7, min_samples=35, metric='euclidean', visualize=True, sigma=0.5, percentile=95):
    # n_clusters, new_embeddings = embedding_refinement(embeddings, 10, None, percentile, sigma, False, False)

    # Normalize the embeddings
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)

    # Reduce dimensionality if needed
    if normalized_embeddings.shape[1] > 10:
        pca = PCA(n_components=10)
        normalized_embeddings = pca.fit_transform(normalized_embeddings)

    # Use sklearn's DBSCAN implementation
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = dbscan.fit_predict(normalized_embeddings)

    # Visualize if requested
    if visualize:
        # Reduce dimensionality for visualization if needed
        if normalized_embeddings.shape[1] > 2:
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(normalized_embeddings)
        else:
            embeddings_2d = normalized_embeddings

        # Create a scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis')
        plt.title(f'DBSCAN Clustering Results (eps={eps}, min_samples={min_samples})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

        plot_nearest_neighbor_distances(normalized_embeddings, n_neighbors=min_samples + 1)

    return labels


# Analyze nearest neighbor distances to help choose eps
def plot_nearest_neighbor_distances(embeddings, n_neighbors=5):
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)

    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(normalized_embeddings)
    distances, _ = neigh.kneighbors(normalized_embeddings)

    plt.figure(figsize=(10, 6))
    plt.hist(distances[:, -1], bins=50)
    plt.title(f'Distribution of distances to the {n_neighbors}-th nearest neighbor')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.show()


# Only test if works add to thesis else only use the other clustering methods
def agglomerative_clustering(embeddings, n_clusters=None, distance_threshold=None, linkage='ward',
                             compute_full_tree='auto', visualize=True, sigma=0.5, percentile=95):

    # n_clusters, new_embeddings = embedding_refinement(embeddings, 10, None, percentile, sigma, False, False)

    # Normalize the embeddings
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)

    # If n_clusters is not specified, use silhouette score to find optimal number of clusters
    if n_clusters is None and distance_threshold is None:
        n_clusters = find_optimal_clusters(normalized_embeddings)

    # Perform Agglomerative Clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters,
                                         distance_threshold=distance_threshold,
                                         linkage=linkage,
                                         compute_full_tree=compute_full_tree,
                                         compute_distances=True)
    labels = clustering.fit_predict(normalized_embeddings)

    # Visualize if requested
    if visualize:
        visualize_agglomerative(normalized_embeddings, labels, clustering)

    return labels


def find_optimal_clusters(data, max_clusters=10):
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    print(f"Optimal number of clusters: {optimal_clusters}")
    return optimal_clusters


def visualize_agglomerative(embeddings, labels, clustering):
    # Reduce dimensionality for visualization if needed
    if embeddings.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis')

    plt.title('Agglomerative Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    # Create dendrogram
    plt.figure(figsize=(10, 7))
    plot_dendrogram(clustering, truncate_mode='level', p=3)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Number of points in node (or index of point if no parenthesis)')
    plt.show()


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    dendrogram(linkage_matrix, **kwargs)

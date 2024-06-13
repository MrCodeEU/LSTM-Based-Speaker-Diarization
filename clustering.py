import math

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import squareform, pdist
from numpy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from visualize import visualize_affinity_matrix_refinement


def custom_distance(X):
    cos_sim = 1 - squareform(pdist(X, metric='cosine'))
    return (1 - cos_sim) / 2


def spectral_clustering(embeddings, sigma=0.5, percentile=95, n_clusters=None, max_clusters=18, visualize=False,
                        visualizeCluster=True, random_state=0):
    # visualise the embeddings using pca
    if visualizeCluster:
        pca = PCA(n_components=2)
        embeddings = pca.fit_transform(embeddings)
        plt.scatter(embeddings[:, 0], embeddings[:, 1])
        plt.title("PCA of Embeddings")
        plt.show()
    # Step 1: Construct the affinity matrix A based on cosine similarity
    A = squareform(pdist(embeddings, metric=custom_distance))  # Compute pairwise cosine distances
    A = 1 - A  # Convert distances to similarities
    # np.fill_diagonal(A, A.max(axis=1))  # Set diagonal elements to the maximum of each row
    np.fill_diagonal(A, 0)  # Set diagonal elements to 0

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

        n_clusters = np.argmax(eigen_gaps) + 2
        print(f"Number of clusters: {n_clusters}")

    # Step 4: Replace segment embeddings with dimensions from the largest eigen-vectors
    new_embeddings = eigenvectors[:, :n_clusters]

    # Dimensionality reduction for visualization (if needed)
    if new_embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        new_embeddings = pca.fit_transform(new_embeddings)

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=random_state)
    labels = kmeans.fit_predict(new_embeddings)

    if visualizeCluster:
        # visualize the clusters
        plt.scatter(new_embeddings[:, 0], new_embeddings[:, 1], c=labels, s=50, cmap='viridis')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='*',
                    label='Centroids')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f"Spectral Clustering with {n_clusters} Clusters")
        plt.legend()
        plt.show()

    return labels


def offline_kmeans(embeddings, max_clusters=10, random_state=0):
    """Performs K-means clustering with elbow method for optimal k."""

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
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='*',
                label='Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f"K-means Clustering with {optimal_k} Clusters")
    plt.legend()
    plt.show()

    return labels

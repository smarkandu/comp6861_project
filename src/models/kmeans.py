"""
torch_kmeans.py

Simple PyTorch implementation of KMeans clustering.

Reference:
[3] von Luxburg, "A Tutorial on Spectral Clustering", 2007.
"""

import torch


def kmeans_torch(X, n_clusters, n_iters=20, device="cpu"):
    """
    Basic KMeans in PyTorch.

    Args:
        X: (N, D) tensor
        n_clusters: number of clusters
        n_iters: number of iterations

    Returns:
        labels: (N,) tensor of cluster assignments
    """
    X = X.to(device)
    N, D = X.shape

    # Random init
    indices = torch.randperm(N)[:n_clusters]
    centroids = X[indices]

    for _ in range(n_iters):
        # Compute distances
        dist = torch.cdist(X, centroids)  # (N, K)

        # Assign clusters
        labels = torch.argmin(dist, dim=1)

        # Update centroids
        new_centroids = []
        for k in range(n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) == 0:
                new_centroids.append(centroids[k])
            else:
                new_centroids.append(cluster_points.mean(dim=0))

        centroids = torch.stack(new_centroids)

    return labels.cpu().numpy()
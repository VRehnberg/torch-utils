import torch

def kmeans(points, k, global_start=1):
    '''Naive K-means with parallelized global start.
    
    Arguments:
        points (Tensor): Points to be clustered. Size([n_points, n_dim]).
        k (int): Number of clusters.
        global_start (int): Number of parallel random initializations.
            Default is 1.

    Returns:
        i_cluster (Tensor): Cluster labels for each point.

    '''
    device = points.device
    emb_dim = points.shape[-1]
    n_points = points.shape[0]
    
    # Initially centroids randomly from points
    rng_idx = torch.randperm(global_start * n_points, device=device) % n_points
    centroids = points[rng_idx].reshape(global_start, *points.shape)[:, :k, :]
    points = points.unsqueeze(0).tile((global_start, 1, 1))
    
    converged = False
    while not converged:
        # Assign to clusters
        distances = (points.unsqueeze(2) - centroids.unsqueeze(1)).pow(2).sum(-1)
        distances, i_cluster = distances.min(-1)
        i_cluster = i_cluster.unsqueeze(2).tile((1, 1, emb_dim))
        
        # Remove duplicate parallel searches
        i_cluster = torch.unique(i_cluster, dim=0)
        
        # Move centroids
        for i in range(k):
            is_in_cluster = (i_cluster == i)
            old_centroids = centroids.clone()
            centroids[:, i, :] = (
                (is_in_cluster.float() * points).sum(dim=1)
                / is_in_cluster[:, :, :1].sum(dim=1)
            )

            # Reinitialize unused centroids
            no_points = (is_in_cluster.sum(1) == 0).nonzero(as_tuple=True)
            assert len(no_points)==2, "This shape was unexpected."
            idx = (
                no_points[0],
                torch.randint(n_points, (len(no_points[0]),), device=device),
                no_points[1],
            )
            centroids[:, i, :][no_points] = points[idx]

        converged = (centroids == old_centroids).all()

    assert (i_cluster.diff(dim=2)==0).all(), "This should've been tiled. Hmm."

    # Pick best clusters
    i_best = distances.sum(-1).argmin()
    i_cluster = i_cluster[i_best, :, 0]
    return i_cluster


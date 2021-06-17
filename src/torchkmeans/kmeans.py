import torch


def kmeans(
    points,
    k,
    global_start=1,
    max_iter=None,
    reinitialize_empty_clusters=True,
    verbose=False,
):
    '''Naive K-means with parallelized global start.
    
    Arguments:
        points (Tensor): Number of points times number of features.
        k (int): Number of clusters.
        global_start (int): Different intializations that are run in parallel.
            Default 1.
        reinatialize_empty_clusters (bool): If empty clusters are reinitialized.
            Default True.
        verbose (bool): Controls verbosity. Default True. 
    '''
    if len(points.shape) > 2:
        raise NotImplementedError("Batches not implemented.")

    emb_dim = points.shape[-1]
    n_points = points.shape[0]
    device = points.device
    
    # Initialize centroids randomly from points
    rng_idx = torch.randperm(global_start * n_points, device=device) % n_points
    centroids = points[rng_idx].reshape(global_start, *points.shape)[:, :k, :]
    points = points.unsqueeze(0).tile((global_start, 1, 1))
    
    converged = False
    iter = 0
    while not converged:
        if verbose:
            print("Iter", iter, end=" ")
        if max_iter and (iter > max_iter):
            warnings.warn("Algorithm finished prematurely. Consider increasing max_iter.", RuntimeWarning)
            break
        # Assign to clusters
        distances = torch.cdist(points, centroids).pow(2)
        distances, i_cluster = distances.min(-1)
        i_cluster = i_cluster.unsqueeze(2).tile((1, 1, emb_dim))

        ## Remove duplicate parallel searches
        #i_cluster, ix = torch.unique(i_cluster, dim=0, return_index=True)
        #points = points[ix, :, :]
        #centroids = centroids[ix, :, :]

        # Move centroids
        for i in range(k):
            is_in_cluster = (i_cluster == i)
            #assert (is_in_cluster.all(dim=-1)==is_in_cluster.any(dim=-1)).all()
            old_centroids = centroids.clone()
            centroids[:, i, :] = (
                (is_in_cluster.float() * points).sum(dim=1)
                / is_in_cluster[:, :, :1].sum(dim=1)
            )

            # Reinitialize unused centroids
            if reinitialize_empty_clusters:
                i_empty_cluster = (is_in_cluster.sum((1, 2)) == 0).nonzero(
                    as_tuple=True
                )[0]
                n_new_centroids = i_empty_cluster.nelement()
                i_random_points = torch.randint(
                    n_points,
                    (n_new_centroids,),
                    device=device,
                )
                new_centroids = points[i_empty_cluster, i_random_points, :]
                centroids[i_empty_cluster, i, :] = new_centroids
    
        # Check convergence
        cluster_converged = torch.isclose(centroids, old_centroids).flatten(1).all(1)
        if iter > 0:
            ssd_converged = torch.isclose(distances.sum(1), old_distances.sum(1))
            assert ssd_converged.shape == cluster_converged.shape
            cluster_converged |= ssd_converged
        old_distances = distances.clone()

        n_converged = cluster_converged.int().sum().item()
        if verbose:
            print(f"{n_converged}/{global_start} have converged.")
        converged = (n_converged == global_start)
        iter += 1

    assert (i_cluster.diff(dim=2)==0).all(), "This should've been tiled. Hmm."

    # Pick best clusters
    i_best = distances.sum(-1).argmin()
    i_cluster = i_cluster[i_best, :, 0]
    return i_cluster


"""
Get KNN edge index with a maximum radius
"""

from torch_geometric.nn import knn


def knn_with_max_radius(x, y,
                        batch_x,
                        batch_y,
                        k,
                        max_radius):
    """
    Construct KNN graph and prune edges by radius.

    Args:
        x (Tensor)         : (num_points, num_features) node positions
        y (Tensor)         : (num_points, num_features) node positions
        batch_x (Tensor)   : (num_points, ) batch index
        batch_x (Tensor)   : (num_points, ) batch index
        k (int)            : number of neighbors
        max_radius (float) : maximum allowed distance for an edge

    Returns:
        edge_index (Tensor): (2, num_edges) pruned edge indices
    """
    edge_index = knn(x=x, y=y,
                     batch_x=batch_x,
                     batch_y=batch_y,
                     k=k)

    head, tail = edge_index
    dist = (x[head] - y[tail]).pow(2).sum(-1).sqrt()

    return edge_index[:, dist <= max_radius]

import torch
import torch.nn.functional as F


def pc_to_route(pcs):
    """
    Convert principal components to routes.
    :param pcs: (..., n_features, n_components) principal components in column vectors
    :return: (...., n_features, 2 * n_components) routes
    """

    positive_routes = pcs.clamp(min=0)
    negative_routes = (-pcs).clamp(min=0)
    routes = torch.cat([positive_routes, negative_routes], dim=-1)
    routes = F.normalize(routes, p=1, dim=-2)
    return routes
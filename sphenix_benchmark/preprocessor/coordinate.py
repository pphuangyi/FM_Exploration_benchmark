"""
Convert Cartesian coordinates to high-energy physics (HEP) coordinates
"""

import torch

# centers of the radius bins
BIN_CENTERS = torch.tensor(
    [31.3757, 31.9478, 32.5198, 33.0918, 33.6639, 34.2359, 34.8079, 35.3800,
     35.9520, 36.5240, 37.0961, 37.6681, 38.2402, 38.8122, 39.3842, 39.9562,
     41.6640, 42.6847, 43.7055, 44.7263, 45.7471, 46.7679, 47.7887, 48.8095,
     49.8303, 50.8511, 51.8718, 52.8926, 53.9134, 54.9342, 55.9550, 56.9758,
     58.9180, 60.0152, 61.1124, 62.2096, 63.3068, 64.4040, 65.5012, 66.5984,
     67.6956, 68.7928, 69.8900, 70.9872, 72.0844, 73.1816, 74.2788, 75.3760]
)


def get_radius_bin(radius):
    """
    Map a radius to the index of the closest bin center
    """
    return (radius.unsqueeze(-1) - BIN_CENTERS.to(radius.device).unsqueeze(0))\
            .abs().min(-1)[1].to(torch.float)


def cartesian_to_hep(cart_points,
                     binned_radius=True,
                     continuous_phi=True,
                     eta_normalizer=1.96,
                     radius_normalizer=48):
    """
    Input:
        cart_points: tensor of shape (N, 3) in x, y, z direction
    Output:
        tpc_points: tensor of shape (N, 3) in
            - eta (Pseudorapidity)
            - phi (azimuthal angle)
            - r (transverse distance, radius)
    """
    x_coord, y_coord, z_coord = cart_points.T

    # radius (transverse distance)
    radius_sq = x_coord ** 2 + y_coord ** 2
    # phi (azimuthal angle)
    phi = torch.arctan2(y_coord, x_coord)
    # eta (Pseudorapidity)
    eta = torch.arctanh(z_coord / torch.sqrt(radius_sq + z_coord ** 2))

    if eta_normalizer is not None:
        eta /= eta_normalizer

    # radius from the beam
    radius = torch.sqrt(radius_sq)

    if binned_radius:
        radius = get_radius_bin(radius)
        if radius_normalizer is not None:
            radius /= radius_normalizer

    if continuous_phi:
        return torch.stack([eta, torch.cos(phi), torch.sin(phi), radius]).T

    return torch.stack([eta, phi, radius]).T

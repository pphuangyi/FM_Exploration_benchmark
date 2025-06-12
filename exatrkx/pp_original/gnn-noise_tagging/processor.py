"""
"""
import os

import torch
from sphenix_benchmark.preprocessor.coordinate import cartesian_to_hep


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


class Processor:
    """
    """
    def __init__(self,
                 max_num_edges     = None,
                 noise_threshold   = .06,
                 continuous_phi    = True,
                 radius_normalizer = 48):

        if max_num_edges is not None:
            self.max_num_edges = max_num_edges
        else:
            self.max_num_edges = float('inf')
        self.noise_threshold = noise_threshold

        # data conversion parameters
        self.continuous_phi    = continuous_phi
        self.radius_normalizer = radius_normalizer


    def __call__(self,
                 points,
                 particles,
                 batch,
                 edge_index,
                 edge_batch):

        inputs = self.get_inputs(points)

        head_indices, tail_indices = self.get_edges(batch      = batch,
                                                    edge_index = edge_index,
                                                    edge_batch = edge_batch)

        noise_tag = self.get_noise_tag(particles)

        return inputs, head_indices, tail_indices, noise_tag


    def __convert(self, points):
        """
        Convert points in Cartesian coordinates to HEP coordinates
        """
        return cartesian_to_hep(points,
                                binned_radius     = True,
                                continuous_phi    = self.continuous_phi,
                                eta_normalizer    = 1.96,
                                radius_normalizer = self.radius_normalizer)

    def get_inputs(self, points):
        """
        Produce network inputs
        """
        energy, cart_coords = points[:, :1], points[:, 1:]
        return torch.hstack([energy, self.__convert(cart_coords)])


    def get_edges(self,
                  batch,
                  edge_index,
                  edge_batch):

        head_indices, tail_indices = [], []

        offset = 0
        for batch_id in range(batch.max() + 1):

            num_nodes = (batch == batch_id).sum().item()

            event_heads, event_tails = edge_index[edge_batch == batch_id].T + offset

            head_indices.append(event_heads)
            tail_indices.append(event_tails)

            offset += num_nodes

        head_indices = torch.cat(head_indices)
        tail_indices = torch.cat(tail_indices)

        num_edges = len(head_indices)
        if num_edges > self.max_num_edges:
            subset = torch.randperm(num_edges)[ : self.max_num_edges]
            head_indices = head_indices[subset]
            tail_indices = tail_indices[subset]

        return head_indices, tail_indices


    def get_noise_tag(self, particles):
        px, py = particles[:, :2].T
        pt = torch.sqrt(px ** 2 + py ** 2)
        return pt > self.noise_threshold

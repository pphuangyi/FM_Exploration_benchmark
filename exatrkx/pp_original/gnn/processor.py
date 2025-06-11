"""
Input processor for training the filter model
It seems that we need to use FilterBaseBalanced module in
in Tracking-ML-Exa.TrkX/Pipelines/TrackML_Example/LightningModules/Filter/filter_base.py

In the processor, we need to do the following:
1. Found a number (or all) postive pairs
2. Fix a cut of the hinge loss
2. Found same number of negative pairs with
"""
import os

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from sphenix_fm.preprocessor.coordinate import cartesian_to_hep
from sphenix_benchmark.datasets.tpc_dataset import TPCDataset


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


class Processor:
    """
    """
    def __init__(self,
                 max_num_edges     = None,
                 continuous_phi    = True,
                 radius_normalizer = 48):

        if max_num_edges is not None:
            self.max_num_edges = max_num_edges
        else:
            self.max_num_edges = float('inf')

        # data conversion parameters
        self.continuous_phi    = continuous_phi
        self.radius_normalizer = radius_normalizer


    def __call__(self,
                 points,
                 track_ids,
                 batch,
                 edge_index,
                 edge_batch):

        inputs = self.get_inputs(points)

        head_indices, tail_indices, labels, truncated \
            = self.get_edges(track_ids  = track_ids,
                             batch      = batch,
                             edge_index = edge_index,
                             edge_batch = edge_batch)

        return inputs, head_indices, tail_indices, labels, truncated


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
                  track_ids,
                  batch,
                  edge_index,
                  edge_batch):

        device = track_ids.device

        head_indices, tail_indices, labels = [], [], []

        offset = 0
        for batch_id in range(batch.max() + 1):

            num_nodes = (batch == batch_id).sum().item()

            event_heads, event_tails = edge_index[edge_batch == batch_id].T + offset

            head_indices.append(event_heads)
            tail_indices.append(event_tails)

            event_labels = track_ids[event_heads] == track_ids[event_tails]

            labels.append(event_labels)

            offset += num_nodes

        head_indices = torch.cat(head_indices)
        tail_indices = torch.cat(tail_indices)
        labels       = torch.cat(labels)

        truncated = False
        if len(labels) > self.max_num_edges:
            subset = torch.randperm(len(labels))[ : self.max_num_edges]
            head_indices = head_indices[subset]
            tail_indices = tail_indices[subset]
            labels       = labels[subset]
            truncated = True

        return (head_indices, tail_indices, labels, truncated)


#     def get_edges(self,
#                   points,
#                   track_ids,
#                   batch,
#                   filtering):
#
#         device = points.device
#
#         indices = torch.arange(points.size(0), device=device)
#
#         head_indices, tail_indices = [], []
#
#         for batch_id in range(batch.max() + 1):
#             event_indices = indices[batch == batch_id]
#
#             _h, _t = get_pairs(len(event_indices))
#
#             # run filtering model to get edges and target
#             start = 0
#
#             while start < len(_h):
#                 end = min(start + self.chunk, len(_h))
#
#                 event_head_indices = event_indices[_h[start: end]]
#                 event_tail_indices = event_indices[_t[start: end]]
#
#                 with torch.no_grad():
#                     logits = filtering(points[event_head_indices],
#                                        points[event_tail_indices])
#
#                 probs = F.sigmoid(logits).squeeze(-1)
#
#                 mask = probs > self.filtering_cut
#                 event_head_indices = event_head_indices[mask]
#                 event_tail_indices = event_tail_indices[mask]
#
#                 head_indices.append(event_head_indices)
#                 tail_indices.append(event_tail_indices)
#
#                 start = end
#
#         head_indices = torch.cat(head_indices)
#         tail_indices = torch.cat(tail_indices)
#
#         return head_indices, tail_indices

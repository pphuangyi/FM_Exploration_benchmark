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

from sphenix_fm.preprocessor.coordinate import cartesian_to_hep


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


def sample(inputs, num_samples, replace=False):
    """
    Sample with or without replacement
    """
    total = inputs.size(0)

    if replace:
        indices = torch.randint(0, total, size=(num_samples,))
    else:
        assert num_samples <= total, \
            (f'num of samples ({num_samples}) should '
             f'not be bigger than total ({total})')
        indices = torch.randperm(total)[ : num_samples]

    return inputs[indices]


def joint_sample(inputs_a, inputs_b, num_samples, replace=False):
    """
    Sample with or without replacement
    """
    assert inputs_a.size(0) == inputs_b.size(0)

    total = inputs_a.size(0)

    if replace:
        indices = torch.randint(0, total, size=(num_samples,))
    else:
        assert num_samples <= total, \
            (f'num of samples ({num_samples}) should '
             f'not be bigger than total ({total})')
        indices = torch.randperm(total)[ : num_samples]

    return inputs_a[indices], inputs_b[indices]


class Processor:
    """
    Processing input and construct queries in the Exa.TrkX way.
    The default parameters are those reported by the paper and from
    Tracking-ML-Exa.TrkX/Pipelines/TrackML_Example/LightningModules/Embeddingtrain_quickstart_embedding.yaml.
    Since we don't have the cell features, we just use x, y, z and energy.
    Besides, true edges for us are NOT just the neighboring edges.
    As long as two points are from the same track, there is an edge between them.
    """
    def __init__(self,
                 embedding_cut     = 2,
                 filtering_cut     = 1.5,
                 subset_size       = 200,
                 # distance power has to match with the power used in the
                 # hinge loss in the embedding step.
                 distance_power    = 2,
                 continuous_phi    = True,
                 radius_normalizer = 48):

        # parameters to choose pairs
        self.embedding_cut  = embedding_cut
        self.filtering_cut  = filtering_cut
        self.subset_size    = subset_size
        self.distance_power = distance_power

        # data conversion parameters
        self.continuous_phi    = continuous_phi
        self.radius_normalizer = radius_normalizer


    def __call__(self,
                 points,
                 track_ids,
                 batch,
                 embedding,
                 filtering,
                 use_embedding):

        inputs = self.get_inputs(points)

        head, tail, labels, pair_type = self.__get_query(inputs, track_ids, batch,
                                                         embedding     = embedding,
                                                         filtering     = filtering,
                                                         use_embedding = use_embedding)

        return inputs[head], inputs[tail], labels, pair_type


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

    def __get_query(self,
                    points,
                    track_ids,
                    batch,
                    embedding,
                    filtering,
                    use_embedding):

        """
        Generate true, hard and easy pairs.
        """

        device = points.device

        indices = torch.arange(points.size(0), device=device)

        head_indices, tail_indices, labels, pair_type = [], [], [], []

        for batch_id in range(batch.max() + 1):
            event_indices = indices[batch == batch_id]
            event_track_ids = track_ids[batch == batch_id]

            if len(event_indices) > self.subset_size:
                event_indices, event_track_ids = joint_sample(
                    event_indices,
                    event_track_ids,
                    num_samples = self.subset_size,
                    replace     = False)

            # Adjacency matrix
            # We use the matrix to get true edges
            # and hard-to-classify false positive.
            adj_matrix = \
                event_track_ids.unsqueeze(0) == event_track_ids.unsqueeze(1)

            # == True Edges ===================================================
            _h, _t = torch.where(adj_matrix)

            head_true = event_indices[_h]
            tail_true = event_indices[_t]

            # labels true should always be True, but let just calculate
            # it for sanity check
            label_true = event_track_ids[_h] == event_track_ids[_t]

            # number of true edges, we are going to sample equal
            # number of hard none edges and easy none edges.
            num_true = len(head_true)

            # == Easy non-edges ===============================================
            head_easy  = torch.tensor([], device=device, dtype=torch.long)
            tail_easy  = torch.tensor([], device=device, dtype=torch.long)
            label_easy = torch.tensor([], device=device, dtype=torch.bool)
            # They may or may not be really easy, they are just
            # randomly sampled
            if (~adj_matrix).sum() > 0:
                _h, _t = joint_sample(*torch.where(~adj_matrix),
                                      num_samples=num_true,
                                      replace=True)

                head_easy = event_indices[_h]
                tail_easy = event_indices[_t]

                # labels easy should always be False, but let just calculate
                # it for sanity check
                label_easy = event_track_ids[_h] == event_track_ids[_t]

            # == Hard non=edges ===============================================
            head_hard  = torch.tensor([], device=device, dtype=torch.long)
            tail_hard  = torch.tensor([], device=device, dtype=torch.long)
            label_hard = torch.tensor([], device=device, dtype=torch.bool)

            # Distance matrix
            # We use distance matrix to get false positive edge
            with torch.no_grad():
                emb = embedding(points[event_indices])
                distance = torch.pow(emb.unsqueeze(1) - emb.unsqueeze(0),
                                     self.distance_power).sum(-1)

            # False positive edge from embedding
            # efp stands for embedding false positive.
            head_efp, tail_efp = \
                torch.where((distance < self.embedding_cut) & (~adj_matrix))

            if len(head_efp) > 0:

                # If an embedding false positive edge also get high probability
                # from the filtering model, we train on it (as an hardness mining)
                with torch.no_grad():
                    head_points = points[event_indices[head_efp]]
                    tail_points = points[event_indices[tail_efp]]

                    if use_embedding:
                        head_points = embedding(head_points)
                        tail_points = embedding(tail_points)

                    probs = F.sigmoid(filtering(head_points, tail_points)).squeeze(-1)

                # Find hard pairs where filtering doesn't work well on
                _i = torch.where(probs > self.filtering_cut)[0]
                # we sample [num_true] many hard pairs
                if len(_i) > 0:
                    _i = sample(_i, num_samples=num_true, replace=True)

                    _h, _t = head_efp[_i], tail_efp[_i]

                    head_hard = event_indices[_h]
                    tail_hard = event_indices[_t]

                    # labels hard should always be False, but let just calculate
                    # it for sanity check
                    label_hard = event_track_ids[_h] == event_track_ids[_t]

            # gather true, hard, and esay pairs
            head_combined = torch.cat([head_true, head_hard, head_easy])
            tail_combined = torch.cat([tail_true, tail_hard, tail_easy])
            label_combined = torch.cat([label_true, label_hard, label_easy])
            p_type = torch.cat([torch.zeros_like(head_true),
                                torch.ones_like(head_hard),
                                torch.ones_like(head_easy) * 2])

            # print(label_combined.size(0) / label_combined.sum().item())

            head_indices.append(head_combined)
            tail_indices.append(tail_combined)
            labels.append(label_combined)
            pair_type.append(p_type)
            # print(list(p_type.cpu().numpy()))
            # print(len(p_type))
            # exit()

        return (torch.cat(head_indices),
                torch.cat(tail_indices),
                torch.cat(labels),
                torch.cat(pair_type))

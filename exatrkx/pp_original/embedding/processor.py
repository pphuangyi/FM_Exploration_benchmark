"""
Input processor for training embedding
"""
import os
from collections import defaultdict
from tqdm import tqdm
import yaml
import torch

from torch_geometric.loader import DataLoader
from torch_geometric.nn import knn

from sphenix_fm.datasets.tpc_dataset import TPCDataset
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


def sample_by_fraction(inputs, fraction, replace=False):
    """
    Sample a fraction of the inputs with or without replacement
    """
    total = inputs.size(0)
    num_samples = int(total * fraction)
    return sample(inputs, num_samples, replace=replace)


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
                 num_knn_neighbors     = 100,
                 knn_query_fraction    = .2,
                 random_query_fraction = .2,
                 continuous_phi        = True,
                 radius_normalizer     = 48):

        self.num_knn_neighbors     = num_knn_neighbors
        self.knn_query_fraction    = knn_query_fraction
        self.random_query_fraction = random_query_fraction

        self.continuous_phi    = continuous_phi
        self.radius_normalizer = radius_normalizer

    def __call__(self, points, track_ids, batch, projection, return_stat=False):

        inputs = self.get_inputs(points)

        # Get paired queries
        knn_head, knn_tail, knn_labels = \
            self.__get_knn_queries(inputs, track_ids, batch, projection)

        if return_stat:
            stat = {'knn_pos_ratio': (knn_labels.sum() / knn_labels.numel()).item(),
                    'num_knn_queries': len(knn_labels)}

        random_head, random_tail, random_labels \
            = self.__get_random_queries(inputs, track_ids, batch)

        if return_stat:
            stat['random_pos_ratio'] = (random_labels.sum() / random_labels.numel()).item()
            stat['num_random_queries'] = len(random_labels)

        head = torch.cat([knn_head, random_head])
        tail = torch.cat([knn_tail, random_tail])
        labels = torch.cat([knn_labels, random_labels])

        if return_stat:
            return inputs[head], inputs[tail], labels, stat

        return inputs[head], inputs[tail], labels

    def get_inputs(self, points):
        energy, cart_coords = points[:, :1], points[:, 1:]
        return torch.hstack([energy, self.__convert(cart_coords)])
    
    def __convert(self, points):
        """
        Convert points in Cartesian coordinates to HEP coordinates
        """
        return cartesian_to_hep(points,
                                binned_radius     = True,
                                continuous_phi    = self.continuous_phi,
                                eta_normalizer    = 1.96,
                                radius_normalizer = self.radius_normalizer)


    def __get_knn_queries(self,
                          points,
                          track_ids,
                          batch,
                          projection,
                          fraction=None):
        """
        Generate hard queries. A query is formed by points that
        are close to each other.
        """
        if projection is not None:
            with torch.no_grad():
                head, tail = knn(x       = projection(points),
                                 y       = projection(points),
                                 batch_x = batch,
                                 batch_y = batch,
                                 k       = self.num_knn_neighbors)
        else:
            head, tail = knn(x       = points,
                             y       = points,
                             batch_x = batch,
                             batch_y = batch,
                             k       = self.num_knn_neighbors)

        if fraction is None:
            fraction = self.knn_query_fraction

        indices = sample_by_fraction(torch.arange(0, len(head)),
                                     fraction = fraction,
                                     replace  = False)

        head = head[indices]
        tail = tail[indices]
        labels = track_ids[head] == track_ids[tail]

        return head, tail, labels

    def get_gnn_inputs(self, points, batch):

        # convert to HEP coordinates
        points_hep = self.__convert(points)

        # Get inputs
        inputs = self.__get_inputs(points_hep, batch)

        # get knn neighbors
        head, tail = knn(x       = points,
                         y       = points,
                         batch_x = batch,
                         batch_y = batch,
                         k       = self.num_knn_neighbors)

        return inputs, head, tail


    def get_knn_queries(self,
                        points,
                        track_ids,
                        batch,
                        fraction       = None,
                        return_indices = False):

        # convert to HEP coordinates
        points_hep = self.__convert(points)

        # Get inputs
        inputs = self.__get_inputs(points_hep, batch)

        # Get paired queries
        head, tail, labels = self.__get_knn_queries(points_hep,
                                                    track_ids,
                                                    batch,
                                                    fraction=fraction)

        if return_indices:
            return inputs[head], inputs[tail], labels, head, tail

        return inputs[head], inputs[tail], labels

    def __get_random_queries(self, points, track_ids, batch, fraction=None):
        """
        For each event, generate a fraction of random queries.
        """

        batch_size = batch.max() + 1
        indices = torch.arange(points.size(0), device=points.device)

        head = []
        tail = []
        labels = []

        if fraction is None:
            fraction = self.random_query_fraction

        for batch_id in range(batch_size):
            event_indices = indices[batch == batch_id]

            event_head = sample_by_fraction(event_indices,
                                            fraction = fraction,
                                            replace  = False)
            event_tail = sample_by_fraction(event_indices,
                                            fraction = fraction,
                                            replace  = False)
            event_labels = track_ids[event_head] == track_ids[event_tail]

            head.append(event_head)
            tail.append(event_tail)
            labels.append(event_labels)

        return (torch.cat(head, dim=0),
                torch.cat(tail, dim=0),
                torch.cat(labels, dim=0))

    def get_random_queries(self, points, track_ids, batch, fraction=None):
        points_hep = self.__convert(points)

        # Get inputs
        inputs = self.__get_inputs(points_hep, batch)

        # Get paired queries
        head, tail, labels = self.__get_random_queries(points_hep,
                                                       track_ids,
                                                       batch,
                                                       fraction=fraction)

        return inputs[head], inputs[tail], labels


def make_undirected(head, tail, mode):
    """
    Expand directed edges to undirected edges
    """
    edges = torch.hstack([torch.vstack([head, tail]),
                          torch.vstack([tail, head])])

    # Remove duplicates
    edges, count = torch.unique(edges, dim=1, return_counts=True)

    if mode == 'union':
        return edges

    if mode == 'intersection':
        return edges[:, count == 2]

    raise KeyError(f'Unknown mode ({mode})')



def test():
    """
    test input processor
    """
    with open('config.yaml', 'r', encoding='UTF-8') as handle:
        config = yaml.safe_load(handle)

    # load data

    data_config = config['data']
    batch_size = config['train']['batch_size']
    train_dataset = TPCDataset(**data_config, split='test')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # set device
    torch.cuda.set_device(1)

    # process data
    processor_config = config['data_processor']
    processor = Processor(**processor_config)

    pbar = tqdm(train_loader)
    summary = defaultdict(list)
    for data in pbar:
        points_car = data['features'].x[:, 1:].cuda()
        track_ids  = data['seg_target'].x.cuda()
        batch      = data['features'].batch.cuda()

        _, _, _, stat = processor(points_car,
                                  track_ids,
                                  batch,
                                  return_stat=True)

        for key, val in stat.items():
            summary[key].append(val)

        pbar.set_postfix(**stat)

    for key, val in summary.items():
        mean = torch.tensor(val, dtype=torch.float).mean().item()
        print(f'{key}: {mean:.3f}')


if __name__ == '__main__':
    test()

"""
Construct KNN-with-max-radius graph
"""

import torch
from sphenix_benchmark.preprocessor import (cartesian_to_hep,
                                            knn_with_max_radius)


class Processor:
    """
    """
    def __init__(self,
                 k                  = 20,
                 max_radius         = .1,
                 target             = 'noise_tagging',
                 noise_threshold    = None,
                 continuous_phi     = True,
                 radius_normalizer  = 48,
                 device             = 'cuda'):

        # neighborhood parameters
        self.k          = k
        self.max_radius = max_radius

        # classification target
        possible_targets = {'noise_tagging' : 2,
                            'pid'           : 5,
                            'mid'           : 2}

        assert target in possible_targets, \
            f'Unknown target {target}! Choose target from {possible_targets}.'

        if target == 'noise_tagging':
            assert noise_threshold is not None, \
                'noise threshold must be given if target is noise_tagging'

            self.noise_threshold = noise_threshold

        self.target = target
        self.num_classes = possible_targets[target]

        # data conversion parameters
        self.continuous_phi    = continuous_phi
        self.radius_normalizer = radius_normalizer

        self.device = device


    def __call__(self, data):

        points = data['features'].x.to(self.device)
        batch  = data['features'].batch.to(self.device)

        # nodes
        inputs = self.get_inputs(points)

        # edges
        coords = inputs[:, 1:]
        edge_index = knn_with_max_radius(x          = coords,
                                         y          = coords,
                                         batch_x    = batch,
                                         batch_y    = batch,
                                         k          = self.k,
                                         max_radius = self.max_radius)

        # node target
        if self.target == 'noise_tagging':
            assert 'reg_target' in data, \
                'reg_target must be given if target is noise_tagging'

            particles = data['reg_target'].x.to(self.device)
            targets = self.get_noise_tag(particles)

        elif self.target == 'pid':
            assert 'pid_target' in data, \
                'pid_target must be loaded if target is pid'

            pid = data['pid_target'].x.to(self.device)
            targets = self.get_pid_label(pid)

        elif self.target == 'mid':
            assert 'mid_target' in data, \
                'mid_target must be loaded if target is mid'

            mid = data['mid_target'].x.to(self.device)
            targets = self.get_mid_label(mid)

        else:
            raise KeyError(f'Unknown target {self.target}!')

        results = {'inputs'     : inputs,
                   'edge_index' : edge_index,
                   'targets'    : targets}

        return results


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


    def get_noise_tag(self, particles):
        """
        tag the noise as particles with pT < noise_threshold
        """
        px, py = particles[:, :2].T
        pt = torch.sqrt(px ** 2 + py ** 2)
        tag = pt < self.noise_threshold
        return tag.to(torch.long)


    def get_pid_label(self, pid):
        """
        Extract pid label from pid_target
        - pion abs(pid) == 211
        - kaon abs(pid) == 321
        - proton abs(pid) == 2212
        - electron abs(pid) == 11
        """

        pid_class = torch.zeros_like(pid, dtype=torch.long)
        pid_class[pid.abs() == 211]  = 1
        pid_class[pid.abs() == 321]  = 2
        pid_class[pid.abs() == 2212] = 3
        pid_class[pid.abs() == 11]   = 4

        return pid_class


    def get_mid_label(self, mid):
        """
        Define weak decay mother IDs
        """

        weak_decay_class = torch.zeros_like(mid, dtype=torch.long)
        mask = (mid == 130) | (mid == 310)
        weak_decay_class[mask] = 1

        return weak_decay_class


def test(config_fname, gpu_id):

    import os
    import yaml
    from tqdm import tqdm
    from torch_geometric.loader import DataLoader
    from sphenix_benchmark.datasets.tpc_dataset_with_edge import TPCDataset
    from sphenix_benchmark.utils import Cumulator

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

    device = 'cuda'
    torch.cuda.set_device(gpu_id)

    with open(config_fname, 'r', encoding='UTF-8') as handle:
        config = yaml.safe_load(handle)

    dataset        = TPCDataset(split='pretrain', **config['data'])
    dataloader     = DataLoader(dataset, batch_size=32, shuffle=True)
    data_processor = Processor(**config['data_processor'], device=device)

    pbar = tqdm(enumerate(dataloader, start=1), total=len(dataloader))

    cumulator = Cumulator()

    for idx, data in pbar:
        results = data_processor(data)

        nodes      = results['inputs']
        edge_index = results['edge_index']
        targets    = results['targets']

        batch = data['features'].batch.to(device)
        track_ids  = data['seg_target'].x.to(device)

        adj = (track_ids.unsqueeze(0) == track_ids.unsqueeze(1)) \
              & (batch.unsqueeze(0) == batch.unsqueeze(1))
        head, tail = edge_index

        true = adj.sum()
        tp = adj[head, tail].sum()

        false = adj.numel() - true
        fp = (~adj[head, tail]).sum()

        recall = (tp / true).item()
        precision = 0 if len(head) == 0 else (tp / len(head)).item()
        false_pos_rate = 0 if false == 0 else (fp / false).item()

        cumulator.update({'recall'         : recall,
                          'precision'      : precision,
                          'false_pos_rate' : false_pos_rate})

        pbar.set_postfix(cumulator.get_average())


if __name__ == '__main__':

    test('configs/config_uniform.yaml', gpu_id=0)

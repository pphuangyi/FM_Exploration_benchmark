"""
Pretraining by solving jigsaw puzzle
"""

import os
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
import numpy as np

# == torch ===============================================
import torch
import torch.nn.functional as F

# == torch.geometric =====================================
from torch_geometric.loader import DataLoader

# == user defined ========================================
from sphenix_benchmark.utils import Cumulator
from sphenix_benchmark.datasets.tpc_dataset import TPCDataset
from sphenix_benchmark.preprocessor.data_processor import make_undirected

from sphenix_fm.preprocessor.coordinate import cartesian_to_hep



os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


def get_pairs(num_samples):
    """
    Generate pair of indices
    """
    indices = torch.arange(num_samples)
    return torch.cartesian_prod(indices, indices).T


class Processor:
    """
    """
    def __init__(self,
                 chunk             = 20000,
                 filtering_cut     = .675,
                 continuous_phi    = True,
                 radius_normalizer = 48):

        # parameters to choose pairs
        self.chunk = chunk
        self.filtering_cut = filtering_cut

        # data conversion parameters
        self.continuous_phi    = continuous_phi
        self.radius_normalizer = radius_normalizer


    def __call__(self,
                 points,
                 batch,
                 filtering):

        inputs = self.get_inputs(points)

        head, tail = self.get_edges(inputs, batch, filtering)

        return inputs, head, tail


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
                  points,
                  batch,
                  filtering):

        device = points.device

        indices = torch.arange(points.size(0), device=device)

        head_indices, tail_indices = [], []

        for batch_id in range(batch.max() + 1):
            event_indices = indices[batch == batch_id]

            _h, _t = get_pairs(len(event_indices))

            # run filtering model to get edges and target
            start = 0

            while start < len(_h):
                end = min(start + self.chunk, len(_h))

                event_head_indices = event_indices[_h[start: end]]
                event_tail_indices = event_indices[_t[start: end]]

                with torch.no_grad():
                    logits = filtering(points[event_head_indices],
                                       points[event_tail_indices])

                probs = F.sigmoid(logits).squeeze(-1)

                mask = probs > self.filtering_cut
                event_head_indices = event_head_indices[mask]
                event_tail_indices = event_tail_indices[mask]

                head_indices.append(event_head_indices)
                tail_indices.append(event_tail_indices)

                start = end

        head_indices = torch.cat(head_indices)
        tail_indices = torch.cat(tail_indices)

        return head_indices, tail_indices


def run_epoch(data_processor,
              filtering,
              dataloader, *,
              device,
              mode,
              output_folder):
    """
    run one epoch on a data loader
    """

    cumulator = Cumulator()

    pbar = tqdm(enumerate(dataloader, start=1), total=len(dataloader), desc='inference')

    for idx, data in pbar:

        points    = data['features'].x.to(device)
        track_ids = data['seg_target'].x.to(device)
        batch     = data['features'].batch.to(device)

        # processing filter_input
        _, head_indices, tail_indices = data_processor(
            points    = points,
            batch     = batch,
            filtering = filtering
        )
        num_undirected_edges = len(head_indices)

        # make undirected edges
        head_indices, tail_indices = make_undirected(head_indices, tail_indices, mode=mode)
        labels = track_ids[head_indices] == track_ids[tail_indices]

        # bookkeeping
        cumulator.update({'num_points': len(points),
                          'num_undirected_edges': num_undirected_edges,
                          'num_directed_edges': len(head_indices),
                          'positive_rate': labels.sum().item() / len(labels)})

        metrics = cumulator.get_average()
        pbar.set_postfix(metrics)

        edge_index = torch.stack([head_indices, tail_indices])
        edge_index = edge_index.detach().cpu().numpy()

        fname = Path(output_folder)/f'event_{idx}'
        np.savez_compressed(fname, edge_index=edge_index)

    return metrics


def get_parameters():
    """
    Get training configuration and training device
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config',
                        type    = str,
                        default = 'config.yaml',
                        help    = 'path to config file | config.yaml')
    parser.add_argument('--device',
                        type    = str,
                        default = 'cuda',
                        choices = ('cuda', 'cpu'),
                        help    = 'device to train the model on | default = cuda')
    parser.add_argument('--gpu-id',
                        type    = int,
                        default = 0,
                        help    = 'GPU to train the model on | default = 0')
    parser.add_argument('--split',
                        type    = str,
                        help    = 'split')
    parser.add_argument('--mode',
                        type    = str,
                        choices = ('union', 'intersection'),
                        default = 'union',
                        help    = 'method to make a graph undirected')

    args = parser.parse_args()

    with open(args.config, 'r', encoding='UTF-8') as handle:
        config = yaml.safe_load(handle)

    return config, args.device, args.gpu_id, args.split, args.mode


def infer():
    """
    Load config, initialize data loaders and models, and run training.
    """

    config, device, gpu_id, split, mode = get_parameters()

    if device == 'cuda':
        torch.cuda.set_device(gpu_id)

    checkpoint_path = Path('checkpoints_raw')

    # load filtering mode
    filtering_path = checkpoint_path/'ckpt_last.script'
    filtering = torch.jit.load(str(filtering_path))
    filtering = filtering.to(device)

    # output folder
    output_folder = checkpoint_path/f'edge_index-{split}'
    if not output_folder.exists():
        output_folder.mkdir()

    data_processor = Processor()

    # data loader
    ds  = TPCDataset(split=split, **config['data'])
    ldr = DataLoader(ds, batch_size=1, shuffle=False)

    with torch.no_grad():
        run_epoch(data_processor,
                  filtering,
                  ldr,
                  device        = device,
                  mode          = mode,
                  output_folder = output_folder)


if __name__ == '__main__':
    infer()

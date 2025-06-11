"""
NumPy Time Projection Chamber Dataset API
"""

from pathlib import Path
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GNNDataLoader

from mmap_ninja import RaggedMmap


class TPCDataset(Dataset):
    """
    Load mmap_ninja data with optional filtering on multiplicity
    """
    def __init__(self,
                 mmap_root,
                 split,
                 target = None,
                 gnn    = True,
                 npart  = None):

        mmap_root = Path(mmap_root)

        # loading memory map data.
        self.feature = RaggedMmap(mmap_root/f'features_{split}')

        # filtering by multiplicity
        if npart is not None:
            meta_fname = mmap_root/'meta.csv'
            assert meta_fname.exists(), \
                ('meta.csv must exist for filtering '
                 'by number of particles!')
            meta = pd.read_csv(mmap_root/'meta.csv')
            meta = meta[ meta.split == split ].reset_index(drop=True)

            meta['npart'] = meta.Npart_proj + meta.Npart_targ
            meta = meta[ meta.npart < npart ].reset_index()
            self.indices = meta['index']
        else:
            self.indices = range(len(self.feature))


        if target is not None:

            if isinstance(target, str):
                targets = [target]
            elif isinstance(target, tuple) or isinstance(target, list):
                targets = target
            else:
                raise ValueError('target can only be a list or a tuple!')

            self.target = {}
            for tgt in targets:
                if tgt in ('seg', 'reg'):
                    key = f'{tgt}_target'
                    self.target[key] = RaggedMmap(mmap_root/f'{key}_{split}')
                else:
                    raise(f"Unknown target {tgt}!")
        else:
            self.target = None

        if gnn:
            self.map_data = lambda data: Data(x=torch.tensor(data))
        else:
            self.map_data = lambda data: torch.tensor(data)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        _index = self.indices[index]

        data = {'features': self.map_data(self.feature[_index])}

        if self.target is not None:
            for key, val in self.target.items():
                data[key] = self.map_data(val[_index])

        return data


def collate_fn(batch):

    keys = batch[0].keys()

    result = {key: torch.cat([item[key] for item in batch], dim=0)
              for key in keys}

    result['batch'] = torch.hstack(
        [torch.full((item['features'].size(0),), idx)
         for idx, item in enumerate(batch)]
    )
    return result


if __name__ == '__main__':

    mmap_root = '/home/sphenix_fm/data/pp_100k_mmap-with_charge'
    # mmap_root = '/home/sphenix_fm/data/pp_100k_mmap'
    split     = 'pretrain'
    # split     = 'test'
    # split     = 'train'
    # target    = None
    targets = ['seg', 'reg']

    # == GNN dataloader ===================================
    print('\nGNN dataloader')
    dataset = TPCDataset(mmap_root = mmap_root,
                         split     = split,
                         target    = targets,
                         gnn       = True)
    print(f'number of event in the {split} split = {len(dataset)}')

    loader = GNNDataLoader(dataset,
                           batch_size = 64,
                           shuffle    = False)

    data = next(iter(loader))
    print('\tfeatures shape: ', data['features'].x.shape)
    print('\tbatch shape: ', data['features'].batch.shape)

    for target in targets:
        print(f'\t{target} target shape: ', data[f'{target}_target'].x.shape)

    # == Torch DataLoader + collate =======================
    print('\nTorch dataloader + collate_fn')
    dataset = TPCDataset(mmap_root = mmap_root,
                         split     = split,
                         target    = targets,
                         gnn       = False)
    print(f'number of event in the {split} split = {len(dataset)}')

    loader = DataLoader(dataset,
                        batch_size = 64,
                        shuffle    = False,
                        collate_fn = collate_fn)

    data = next(iter(loader))
    print('\tfeatures shape: ', data['features'].shape)
    print('\tbatch shape: ', data['batch'].shape)
    for target in targets:
        print(f'\t{target} target shape: ', data[f'{target}_target'].shape)

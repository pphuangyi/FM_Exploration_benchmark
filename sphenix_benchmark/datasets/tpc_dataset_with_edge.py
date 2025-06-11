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
                 target    = None,
                 gnn       = False,
                 load_edge = False,
                 **kwargs):

        mmap_root = Path(mmap_root)

        # load memory-map data.
        self.feature = RaggedMmap(mmap_root/f'features_{split}')

        # load metadata
        meta_fname = mmap_root/'meta.csv'
        assert meta_fname.exists(), \
            ('meta.csv must exist for filtering '
             'by number of particles!')
        meta = pd.read_csv(mmap_root/'meta.csv')
        meta = meta[ meta.split == split ].reset_index(drop=True)
        meta['index'] = meta.index

        # filter data
        for key, val in kwargs.items():
            field, boundary = key.split('_')
            assert field in meta.columns, \
                f'meta data does not have field {field}'

            if boundary == 'min':
                meta = meta[ meta[field] >= val ]
            elif boundary == 'max':
                meta = meta[ meta[field] < val ]
            else:
                raise ValueError(f'value boundary should be "min" or "max"')

        self.indices = meta.index
        print(f'\nLoaded {len(self.indices)} events from {str(mmap_root)}!\n')

        # load target
        if target is not None:

            if isinstance(target, str):
                targets = [target]
            elif isinstance(target, (tuple, list)):
                targets = target
            else:
                raise ValueError('target can only be a list or a tuple!')

            self.target = {}
            for tgt in targets:
                key = f'{tgt}_target'
                mmap_path = mmap_root/f'{key}_{split}'
                assert mmap_path.exists(), f'{tgt} target does not exist!'
                self.target[key] = RaggedMmap(mmap_path)
        else:
            self.target = None

        # load edge index
        if load_edge:
            mmap_path = mmap_root/f'edge_index_{split}'
            assert mmap_path.exists(), \
                f'edge index for {split} split does not exists!'
            self.edge_index = RaggedMmap(mmap_path)
        else:
            self.edge_index = None

        if gnn:
            self.map_data = lambda data: Data(x=torch.tensor(data))
        else:
            self.map_data = lambda data: torch.tensor(data)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        _index = self.indices[index]

        # Load input features
        data = {'features': self.map_data(self.feature[_index])}

        # Load targets
        if self.target is not None:
            for key, val in self.target.items():
                data[key] = self.map_data(val[_index])

        # Load edge index
        if self.edge_index is not None:
            # The GNN Dataloader concat along the 0-dimension, and
            # hence we need to transpose the edge index data.
            data['edge_index'] = self.map_data(self.edge_index[_index].T)

        return data


def collate_fn(batch):

    keys = batch[0].keys()

    result = {key: torch.cat([item[key] for item in batch], dim=0)
              for key in keys}

    result['batch'] = torch.hstack(
        [torch.full((item['features'].size(0),), idx)
         for idx, item in enumerate(batch)]
    )

    if 'edge_index' in keys:
        result['edge_batch'] = torch.hstack(
            [torch.full((item['edge_index'].size(0),), idx)
            for idx, item in enumerate(batch)]
        )

    return result


def test(mmap_root, split, target, batch_size, **kwargs):

    dataset = TPCDataset(mmap_root = mmap_root,
                         split     = split,
                         target    = target,
                         **kwargs)

    is_gnn = ('gnn' in kwargs) and kwargs['gnn']

    if is_gnn:
        loader = GNNDataLoader(dataset,
                               batch_size = batch_size,
                               shuffle    = True)
    else:
        loader = DataLoader(dataset,
                            batch_size = batch_size,
                            shuffle    = True,
                            collate_fn = collate_fn)

    data = next(iter(loader))

    print(f'Batch (of size {batch_size})')
    if is_gnn:
        features = data['features'].x
        batch = data['features'].batch
        print('\tfeatures shape: ', data['features'].x.shape)
        print('\tbatch shape: ', data['features'].batch.shape)
    else:
        features = data['features']
        batch = data['batch']
        print('\tfeatures shape: ', data['features'].shape)
        print('\tbatch shape: ', data['batch'].shape)

    target_dict = {}
    for tgt in target:
        if is_gnn:
            target_dict[tgt] = data[f'{tgt}_target'].x
            print(f'\t{tgt} target shape: ', data[f'{tgt}_target'].x.shape)
        else:
            target_dict[tgt] = data[f'{tgt}_target']
            print(f'\t{tgt} target shape: ', data[f'{tgt}_target'].shape)

    if 'edge_index' in data:
        if is_gnn:
            edge_index = data['edge_index'].x
            edge_batch = data['edge_index'].batch
            print(f'\tedge index shape: ', data[f'edge_index'].x.shape)
        else:
            edge_index = data['edge_index']
            edge_batch = data['edge_batch']
            print(f'\tedge index shape: ', data[f'edge_index'].shape)

        # sanity check for edge_index:
        batch_size = batch.max() + 1
        for batch_idx in range(batch_size):
            event_features = features[batch == batch_idx]
            event_edge_index = edge_index[edge_batch == batch_idx]

            node_idx_min = event_edge_index.min()
            node_idx_max = event_edge_index.max()

            print(f'event {batch_idx}: ')
            print(f'\tnumber of nodes = {len(event_features)}')
            print(f'\tnode index min  = {node_idx_min}')
            print(f'\tnode index max  = {node_idx_max}')

            if 'seg' in target:
                event_track_ids = target_dict['seg'][batch == batch_idx]
                head, tail = event_edge_index.T
                labels = event_track_ids[head] == event_track_ids[tail]
                pos_rate = (labels.sum() / len(labels)).item()
                print(f'\tpostive rate = {pos_rate: .2f}')



if __name__ == '__main__':

    test(mmap_root  = '/home/sphenix_fm/data/pp_100k_mmap-with_charge',
         split      = 'pretrain',
         target     = ['seg', 'reg'],
         batch_size = 4,
         gnn        = True,
         load_edge  = True)

    # test(mmap_root = '/home/sphenix_fm/data/pp_100k_mmap-index',
    #      split     = 'pretrain',
    #      target    = ['seg', 'reg', 'track_index'])

    # test(mmap_root = '/home/sphenix_fm/data/auau_100k_mmap',
    #      split     = 'pretrain',
    #      target    = ['seg', 'reg'],
    #      multiplicity_min=1000,
    #      multiplicity_max=30000,
    #      Npart_max=32)

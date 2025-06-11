"""
Convert edge_index information saved as npz to memory map file
"""
from pathlib import Path
import numpy as np

from mmap_ninja.ragged import RaggedMmap

if __name__ == '__main__':

    split = 'pretrain'
    edge_folder = Path(f'checkpoints_raw/edge_index-{split}')
    npzs = list(edge_folder.glob('*npz'))

    # The event edge information is generated in the same
    # order as the original data.
    npzs = sorted(npzs, key=lambda x: int(x.stem.split('_')[-1]))

    # load data
    print('Loading data ...', flush=True)
    data = [np.load(npz)['edge_index'] for npz in npzs]

    mmap_folder = Path('/home/sphenix_fm/data/pp_100k_mmap-with_charge/')
    out_dir = mmap_folder/f'edge_index_{split}'

    # save the memory map
    print('Saving Mmap ...', flush=True)
    RaggedMmap.from_generator(
        out_dir          = out_dir,
        sample_generator = data,
        batch_size       = 64,
        verbose          = True
    )

    # validation
    print('Validating ...', flush=True)
    mmap = RaggedMmap(out_dir)
    try:
        length = len(mmap)
        for datum_id in range(length):
            mmap[datum_id]
        print(f'\033[32mmmap in {str(out_dir)} looks good!')
    except RuntimeError:
        print(f'\033[31mconversion failed on {str(out_dir)}')

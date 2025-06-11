"""
Pretraining by solving jigsaw puzzle
"""

import os
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
import pandas as pd

# == torch ===============================================
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

# == torch.geometric =====================================
from torch_geometric.loader import DataLoader

# == user defined ========================================
from sphenix_benchmark.utils import (get_lr,
                                     count_parameters,
                                     Checkpointer,
                                     Cumulator)
from sphenix_benchmark.datasets.tpc_dataset import TPCDataset

from processor import Processor
from models import ExaTrkXFilter


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


def run_epoch(processor,
              embedding,
              filtering,
              use_embedding,
              loss_fn,
              dataloader, *,
              device,
              optimizer = None,
              desc      = None):
    """
    run one epoch on a data loader
    """

    cumulator = Cumulator()

    pbar = tqdm(enumerate(dataloader, start=1), total=len(dataloader), desc=desc)

    for idx, data in pbar:

        points    = data['features'].x.to(device)
        track_ids = data['seg_target'].x.to(device)
        batch     = data['features'].batch.to(device)

        # processing input
        head, tail, labels, pair_type = processor(
            points        = points,
            track_ids     = track_ids,
            batch         = batch,
            embedding     = embedding,
            filtering     = filtering,
            use_embedding = use_embedding
        )

        # for h, t, l in zip(head[:100], tail[:100], labels[:100]):
        #     print(f'{l}\n{h.detach().cpu().numpy()}\n{t.detach().cpu().numpy()}\n\n')
        # exit()

        if use_embedding:
            with torch.no_grad():
                head = embedding(head)
                tail = embedding(tail)

        logits = filtering(head, tail).squeeze(-1)

        a = torch.where(torch.isnan(logits))[0]

        loss = loss_fn(logits, labels.float())

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            probs = F.sigmoid(logits)
            true_score = probs[pair_type == 0].sum().item()
            hard_score = probs[pair_type == 1].sum().item()
            easy_score = probs[pair_type == 2].sum().item()

            true_score /= (pair_type == 0).sum()
            if hard_score > 0:
                hard_score /= (pair_type == 1).sum()
            if easy_score > 0:
                easy_score /= (pair_type == 2).sum()


        # bookkeeping
        cumulator.update({'loss': loss.item(),
                          'true_score': true_score,
                          'hard_score': hard_score,
                          'easy_score': easy_score})

        metrics = cumulator.get_average()
        pbar.set_postfix(metrics)

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

    args = parser.parse_args()

    with open(args.config, 'r', encoding='UTF-8') as handle:
        config = yaml.safe_load(handle)

    return config, args.device, args.gpu_id


def train():
    """
    Load config, initialize data loaders and models, and run training.
    """

    config, device, gpu_id = get_parameters()

    if device == 'cuda':
        torch.cuda.set_device(gpu_id)

    # checkpointing parameters
    checkpoint_path = config['checkpointing']['checkpoint_path']
    save_frequency  = config['checkpointing']['save_frequency']
    resume          = config['checkpointing']['resume']

    # training parameters
    num_epochs = config['train']['num_epochs']
    batch_size = config['train']['batch_size']

    # process TPC data to get training input and target
    processor = Processor(**config['processor'])

    embedding = torch.jit.load(config['embedding'])
    embedding = embedding.to(device)
    use_embedding = config['use_embedding']

    # model parameters
    filtering = ExaTrkXFilter(**config['filtering'])
    loss_fn = nn.BCEWithLogitsLoss()

    # optimizer and schedular
    # the hyper-parameter values are from the repo:
    # Tracking-ML-Exa.TrkX/Pipelines/TrackML_Example/LightningModules/\
    # Filter/train_quickstart_filter.yaml
    optimizer = AdamW(
        filtering.parameters(),
        lr=0.0001,
        betas=(0.9, 0.999),
        eps=1e-08,
        amsgrad=True,
    )
    scheduler = StepLR(optimizer, step_size=8, gamma=.3)

    # Checkpointer
    checkpointer = Checkpointer(filtering,
                                optimizer       = optimizer,
                                scheduler       = scheduler,
                                checkpoint_path = checkpoint_path,
                                save_frequency  = save_frequency,
                                jit             = 'script')
    # resume if necessary
    resume_epoch = 0
    if resume:
        resume_epoch = checkpointer.load(device=device)

    filtering = filtering.to(device)

    # model sizes
    model_size_dict = {
        'model'   : count_parameters(filtering) / (1024 ** 2),
    }
    for key, val in model_size_dict.items():
        print(f'{key}: {val:.4f}M')

    model_size_dataframe = pd.DataFrame({'model' : model_size_dict.keys(),
                                         'size'  : model_size_dict.values()})
    model_size_dataframe.to_csv(Path(checkpoint_path)/'model_size.csv', index=False)


    # data loader
    train_ds  = TPCDataset(split='pretrain', **config['data'])
    valid_ds  = TPCDataset(split='test', **config['data'])
    train_ldr = DataLoader(train_ds,
                           batch_size = batch_size,
                           shuffle    = True)
    valid_ldr = DataLoader(valid_ds,
                           batch_size = batch_size,
                           shuffle    = True)

    # training
    train_log = Path(checkpoint_path)/'train_log.csv'
    valid_log = Path(checkpoint_path)/'valid_log.csv'
    for epoch in range(resume_epoch + 1, num_epochs + 1):

        current_lr = get_lr(optimizer)
        print(f'current learning rate = {current_lr:.10f}')

        # train
        desc = f'Train Epoch {epoch} / {num_epochs}'
        filtering.train()
        train_stat = run_epoch(processor,
                               embedding,
                               filtering,
                               use_embedding,
                               loss_fn,
                               train_ldr,
                               desc      = desc,
                               optimizer = optimizer,
                               device    = device)

        # validation
        with torch.no_grad():
            filtering.eval()
            desc = f'Validation Epoch {epoch} / {num_epochs}'
            valid_stat = run_epoch(processor,
                                   embedding,
                                   filtering,
                                   use_embedding,
                                   loss_fn,
                                   valid_ldr,
                                   desc   = desc,
                                   device = device)

        # update learning rate
        scheduler.step()

        # save checkpoints
        checkpointer.save(epoch)

        # log the results
        for log, stat in zip([train_log, valid_log],
                             [train_stat, valid_stat]):

            stat.update({'lr': current_lr, 'epoch': epoch})
            dataframe = pd.DataFrame(data=stat, index=[1])
            dataframe.to_csv(log,
                             index        = False,
                             float_format = '%.6f',
                             mode         = 'a' if log.exists() else 'w',
                             header       = not log.exists())


if __name__ == '__main__':
    train()

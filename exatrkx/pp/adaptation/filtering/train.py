"""
Pretraining by solving jigsaw puzzle
"""

import os
from itertools import chain
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
import pandas as pd

# == torch ===============================================
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR

# == torch.geometric =====================================
from torch_geometric.loader import DataLoader

# == user defined ========================================
from sphenix_benchmark.utils import (get_lr,
                                     count_parameters,
                                     Checkpointer,
                                     Cumulator)
from sphenix_benchmark.datasets.tpc_dataset import TPCDataset
from sphenix_benchmark.preprocessor.data_processor import Processor
from sphenix_benchmark.models.mlp import MLP
from sphenix_benchmark.metrics import compute_roc, compute_pr


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


def run_epoch(data_processor,
              model,
              loss_fn,
              dataloader, *,
              device,
              batches_per_epoch = None,
              optimizer         = None,
              desc              = None):
    """
    run one epoch on a data loader
    """

    if batches_per_epoch is None:
        batches_per_epoch = float('inf')

    cumulator = Cumulator()

    total_batches = min(len(dataloader), batches_per_epoch)

    pbar = tqdm(enumerate(dataloader, start=1), total=total_batches, desc=desc)

    for idx, data in pbar:

        if idx > batches_per_epoch:
            break

        points    = data['features'].x[:, 1:].to(device)
        track_ids = data['seg_target'].x.to(device)
        batch     = data['features'].batch.to(device)

        # processing input
        head, tail, labels = data_processor(
            points    = points,
            track_ids = track_ids,
            batch     = batch
        )

        pred = model(head, tail).squeeze(-1)

        loss = loss_fn(pred.squeeze(-1), labels.float())

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            prob = torch.sigmoid(pred)
            _, _, auc = compute_roc(prob, labels, vmin=0, vmax=1, reverse=False)
            _, _, avg_precision = compute_pr(prob, labels, vmin=0, vmax=1, reverse=False)

            # False positive rate with a fixed threshold (.5)
            threshold = .5
            false_pos      = ((prob > threshold) & (~labels)).sum()
            true_neg       = ((prob < threshold) & (~labels)).sum()
            false_pos_rate = false_pos / (false_pos + true_neg)

        # bookkeeping
        cumulator.update({
            'loss'                        : loss.item(),
            'auc'                         : auc,
            'avg_precision'               : avg_precision,
            f'false_pos_rate({threshold})': false_pos_rate.item()
        })

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


class Model(nn.Module):
    """
    Wrapping embedding and filter models into one model
    """
    def __init__(self, embedding_model, filter_model):
        super().__init__()
        self.embedding_model = embedding_model
        self.filter_model    = filter_model

    def forward(self, head, tail):
        head_emb = self.embedding_model(head)
        tail_emb = self.embedding_model(tail)
        return self.filter_model(torch.cat([head_emb, tail_emb], dim=-1))


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
    num_epochs        = config['train']['num_epochs']
    num_warmup_epochs = config['train']['num_warmup_epochs']
    batch_size        = config['train']['batch_size']
    batches_per_epoch = config['train']['batches_per_epoch']
    learning_rate     = config['train']['learning_rate']
    sched_steps       = config['train']['sched_steps']
    sched_gamma       = config['train']['sched_gamma']

    # process TPC data to get training input and target
    processor = Processor(**config['data_processor'])

    # model parameters
    embedding_model = MLP(**config['embedding_model'])

    config['filter_model']['input_features'] = \
        config['embedding_model']['output_features'] * 2
    filter_model = MLP(**config['filter_model'])

    model = Model(embedding_model, filter_model)

    loss_fn = nn.BCEWithLogitsLoss()

    # optimizer and schedular
    optimizer = AdamW(chain(embedding_model.parameters(),
                            filter_model.parameters()),
                      lr=learning_rate)

    sched_milestones = range(num_warmup_epochs, num_epochs, sched_steps)
    scheduler = MultiStepLR(optimizer,
                            milestones=sched_milestones,
                            gamma=sched_gamma)

    # Checkpointers
    checkpointer = Checkpointer(model,
                                optimizer       = optimizer,
                                scheduler       = scheduler,
                                checkpoint_path = checkpoint_path,
                                save_frequency  = save_frequency,
                                jit             = 'script')

    # resume if necessary
    resume_epoch = 0
    if resume:
        resume_epoch = checkpointer.load(device=device)

    model = model.to(device)

    # model sizes
    mega = 1024 ** 2
    model_size_dict = {
        'embedding_model' : count_parameters(embedding_model) / mega,
        'filter_model'    : count_parameters(filter_model) / mega,
        'model'           : count_parameters(model) / mega
    }
    for key, val in model_size_dict.items():
        print(f'{key}: {val:.4f}M')

    model_size_dataframe = pd.DataFrame({'model' : model_size_dict.keys(),
                                         'size'  : model_size_dict.values()})
    model_size_dataframe.to_csv(
        Path(checkpoint_path)/'model_size.csv',
        index=False
    )


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
        model.train()
        train_stat = run_epoch(processor,
                               model,
                               loss_fn,
                               train_ldr,
                               desc              = desc,
                               optimizer         = optimizer,
                               batches_per_epoch = batches_per_epoch,
                               device            = device)

        # validation
        with torch.no_grad():
            model.eval()
            desc = f'Validation Epoch {epoch} / {num_epochs}'
            valid_stat = run_epoch(processor,
                                   model,
                                   loss_fn,
                                   valid_ldr,
                                   desc              = desc,
                                   batches_per_epoch = batches_per_epoch,
                                   device            = device)

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

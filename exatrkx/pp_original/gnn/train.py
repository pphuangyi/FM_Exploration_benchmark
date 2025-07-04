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
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

# == torch.geometric =====================================
from torch_geometric.loader import DataLoader

# == user defined ========================================
from sphenix_benchmark.utils import (get_lr,
                                     count_parameters,
                                     Checkpointer,
                                     Cumulator)
from sphenix_benchmark.datasets.tpc_dataset_with_edge import TPCDataset
from sphenix_benchmark.metrics import compute_roc, compute_pr

# local import (in fact update)
from processor import Processor
from models import assemble_gnn


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


def run_epoch(data_processor,
              gnn_model,
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

    total_truncated = 0
    for idx, data in pbar:

        if idx > batches_per_epoch:
            break

        points    = data['features'].x.to(device)
        track_ids = data['seg_target'].x.to(device)
        batch     = data['features'].batch.to(device)

        edge_index = data['edge_index'].x.to(device)
        edge_batch = data['edge_index'].batch.to(device)

        # data processing
        inputs, head_indices, tail_indices, labels, truncated = data_processor(
            points     = points,
            track_ids  = track_ids,
            batch      = batch,
            edge_index = edge_index,
            edge_batch = edge_batch
        )
        total_truncated += truncated

        # gnn model
        logits = gnn_model(inputs, head_indices, tail_indices)
        loss = loss_fn(logits.squeeze(-1), labels.float())

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # with torch.no_grad():
        #     probs = torch.sigmoid(logits)
        #     _, _, auc = compute_roc(probs, labels, vmin=0, vmax=1, reverse=False)
        #     _, _, avg_precision = compute_pr(probs, labels, vmin=0, vmax=1, reverse=False)

        #     # true_pos = pred * labels
        #     # recall = true_pos.sum() / labels.sum()
        #     # precision = true_pos.sum() / pred.sum()
        #     threshold = .5
        #     false_neg = ((probs > threshold) * (~labels)).sum()
        #     neg = (~labels).sum()
        #     if neg == 0:
        #         false_neg_rate = 0
        #     else:
        #         false_neg_rate = (false_neg / neg).item()

        # bookkeeping
        cumulator.update({
            'loss'                         : loss.item(),
            # 'recall'         : recall.item(),
            # 'precision'      : precision.item(),
            # 'auc'                          : auc,
            # 'avg_precision'                : avg_precision,
            # f'false_neg_rate({threshold})' : false_neg_rate
        })

        metrics = cumulator.get_average()
        pbar.set_postfix(metrics)
        # pbar.set_postfix({'total truncated': total_truncated})

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
    processor = Processor(**config['data_processor'])

    # model parameters
    gnn_model = assemble_gnn(config['gnn_model'])
    loss_fn = BCEWithLogitsLoss()

    optimizer = AdamW(
        gnn_model.parameters(),
        lr=0.0001,
        betas=(0.9, 0.999),
        eps=1e-08,
        amsgrad=True,
    )
    scheduler = StepLR(optimizer, step_size=10, gamma=.3)


    # Checkpointer
    checkpointer = Checkpointer(gnn_model,
                                optimizer       = optimizer,
                                scheduler       = scheduler,
                                checkpoint_path = checkpoint_path,
                                save_frequency  = save_frequency,
                                jit             = None)

    # resume if necessary
    resume_epoch = 0
    if resume:
        resume_epoch = checkpointer.load(device=device)

    gnn_model = gnn_model.to(device)

    # model sizes
    model_size_dict = {
        'gnn_model' : count_parameters(gnn_model) / (1024 ** 2),
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
        gnn_model.train()
        train_stat = run_epoch(processor,
                               gnn_model,
                               loss_fn,
                               train_ldr,
                               desc      = desc,
                               optimizer = optimizer,
                               device    = device)

        # validation
        with torch.no_grad():
            gnn_model.eval()
            desc = f'Validation Epoch {epoch} / {num_epochs}'
            valid_stat = run_epoch(processor,
                                   gnn_model,
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

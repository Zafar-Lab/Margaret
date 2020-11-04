import ast
import copy
import gc
import logging
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import tqdm as tq

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from util.config import ConfigLoader, configure_device, get_optimizer, get_loss, configure_model_params, \
    get_lr_scheduler, save_checkpoint, load_checkpoint
from util.criterion import *


__all__ = ['train_on_gpu']


# TODO: Support MultiGPU training
def train_on_gpu(file_path, save_dir, checkpoint_path, net, dataset, val_dataset, remove_previous_chkpt=True):
    # Set a deterministic CuDNN backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = configure_device('gpu')

    # Load the param config
    config_loader = ConfigLoader(file_path)

    # Create the loader from the dataset
    data_params = config_loader.get_param_value('data')
    batch_size = int(data_params.get('batch_size', 32))
    eval_batch_size = int(data_params.get('val_batch_size', 32))
    num_workers = int(data_params.get('workers', 0))
    
    # Data Loaders
    ignore_index = getattr(dataset, 'IGNORE_INDEX', None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, num_workers=num_workers, drop_last=False, pin_memory=True)

    # Transfer the model to device
    net = net.to(device)

    # Training Hyperparameters
    train_params = config_loader.get_param_value('train')
    seed = int(train_params['random_seed'])
    random.seed(seed)
    num_epochs = int(train_params['num_epochs'])
    log_step = int(train_params['log_step'])
    
    # Loss params
    loss_params = config_loader.get_param_value('loss')
    loss_name = loss_params.pop('loss')
    kwargs = {k: v for k, v in loss_params.items()}
    kwargs['ignore_index'] = ignore_index
    loss_function = get_loss(loss_name, **kwargs).to(device)
    print('---------- Loss Params ---------')
    print(f'Using Loss Criterion: {loss_function}')
    print(f'Ignore Index: {loss_function.ignore_index}')

    # Optimizer params
    optim_params = config_loader.get_param_value('optim')
    optim_name = optim_params.pop('optimizer')
    lr = float(optim_params.pop('learning_rate'))
    kwargs = {k: v for k, v in optim_params.items()}
    optimizer = get_optimizer(optim_name, net, lr, **kwargs)

    sched_params = config_loader.get_param_value('scheduler')
    sched_type = sched_params.pop('type')
    kwargs = {k: v for k, v in sched_params.items()}
    lr_scheduler = get_lr_scheduler(optimizer, num_epochs, sched_type=sched_type, **kwargs)
    print('---------- Optimizer Params ---------')
    print(f'Using optimizer: {optimizer}')
    print(f'Using LR Scheduler: {lr_scheduler}')

    # Load any saved state checkpoints
    iter_val = 0
    loss_profile = []
    if checkpoint_path:
        chkpt = load_checkpoint(checkpoint_path, net, optimizer, lr_scheduler, device)
        iter_val = chkpt['iter']
        loss_profile = chkpt['loss_profile']
        net = chkpt['model']
        optimizer = chkpt['optimizer']
        lr_scheduler = chkpt['scheduler']
    
    # Log all the parameters before training starts
    print('---------- Running training with the following params ---------')
    print(f'Random Seed: {seed}')
    print(f'Num epochs: {num_epochs}')
    print(f'Log step: {log_step}')
    print(f'Batch Size: {batch_size}')
    print(f'Device: {device}')
    
    best_auc = 0.0
    save_dir_best = os.path.join(save_dir, 'best')
    # Main Training Loop
    tk0 = tq.tqdm(range(iter_val, num_epochs))
    for epoch_idx in tk0:
        avg_epoch_loss = _train_one_epoch(
            epoch_idx, loader, val_loader, net, loss_function, optimizer,
            log_step=log_step, device=device
        )
        lr_scheduler.step()
        loss_profile.append(avg_epoch_loss)

        # Evaluate auc score on the val set
        avg_auc_score = evaluate(val_loader, net, device, ignore_index=ignore_index)
        tk0.set_postfix({'Epoch': epoch_idx + 1, 'Epoch Train Loss': avg_epoch_loss, 'Avg AUC score': avg_auc_score})

        # Logging and State saving
        if best_auc <= avg_auc_score:
            best_auc = avg_auc_score
            save_checkpoint(epoch_idx, file_path, save_dir_best, 'gpu', net, optimizer, lr_scheduler, loss_profile)
        
        if epoch_idx % 10 == 0:
            save_checkpoint(epoch_idx, file_path, save_dir, 'gpu', net, optimizer, lr_scheduler, loss_profile)
        
        tk0.set_postfix({'Epoch': epoch_idx + 1, 'Epoch Train Loss': avg_epoch_loss, 'Avg AUC score': avg_auc_score, 'Best AUC': best_auc})

        # Garbage collection
        gc.collect()


def evaluate(val_loader, net, device, ignore_index=-1):
    net.eval()
    tk0 = val_loader
    with torch.no_grad():
        targets = []
        preds = []
        for data_batch, target_batch in tk0:
            data_batch = data_batch.to(device)
            val_pred_batch = torch.sigmoid(net(data_batch))
            targets.append(target_batch)
            preds.append(val_pred_batch)
        targets = torch.cat(targets, dim=0).cpu().numpy()
        preds = torch.cat(preds, dim=0).cpu().numpy()
        targets_ = targets[targets != ignore_index]
        preds_ = preds[targets != ignore_index]
        # Compute the ROC AUC metric
        auc_score = roc_auc_score(targets_, preds_)
    return auc_score


def mixup_data(x, y, device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def smooth_labels(target, epsilon=0.1):
    target[target == 0] = epsilon
    target[target == 1] = 1- epsilon
    return target


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Single epoch training loop
def _train_one_epoch(epoch_idx, data_loader, val_loader, net, loss_criterion, optimizer, log_step=50, device='cpu', eval=False):
    val_data_batch, val_target_batch = next(iter(val_loader))
    epoch_loss = 0
    loss = 0
    for batch_idx, (data_batch, target_batch) in enumerate(data_loader):
        # Zero out gradients
        optimizer.zero_grad()

        # Move the training sample to device
        net.train()
        data_batch = data_batch.to(device)
        target_batch = target_batch.to(device)

        # Apply label smoothing here
        # target_batch = smooth_labels(target_batch, epsilon=0.1)

        # Compute Loss and optimize
        do_mixup = random.random() > 0.5
        if do_mixup:
            # Loss computation on the mixed up batch
            mixed_data_batch, targets_a, targets_b, lam = mixup_data(data_batch, target_batch, device, alpha=2.0)
            out = net(mixed_data_batch)
            loss = mixup_criterion(loss_criterion, out, targets_a, targets_b, lam)
            loss.backward()
        else:
            # Loss computation on non mixed up batch
            out = net(data_batch)
            loss = loss_criterion(out, target_batch)
            loss.backward()
        optimizer.step()

        # Gather loss statistics
        loss_ = loss.detach().cpu().numpy()
        epoch_loss += loss_

        # Check validation loss on a batch
        if eval:
            val_data_batch = val_data_batch.to(device)
            val_target_batch = val_target_batch.to(device)
            net.eval()
            with torch.no_grad():
                val_pred_batch = net(val_data_batch)
                val_loss = loss_criterion(val_pred_batch, val_target_batch)
                tk0.set_postfix_str(s=f'Validation Loss: {val_loss.cpu().numpy()}')

    return epoch_loss / len(data_loader)


if __name__ == "__main__":
    train()

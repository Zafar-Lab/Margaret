import ast
import copy
import gc
import logging
import os
import torch
import torch.nn.functional as F
import tqdm.notebook as tq

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from util.config import ConfigLoader, configure_device, get_optimizer, get_loss, configure_model_params, \
    save_checkpoint, load_checkpoint


try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
except ImportError:
    raise Exception('PyTorch XLA not installed!')


__all__ = ['train_on_tpu']


def train_on_tpu(rank, file_path, save_dir, checkpoint_path, net, dataset, val_dataset, remove_previous_chkpt=True):
    device = configure_device('tpu')

    # Load the param config
    config_loader = ConfigLoader(file_path)

    # Create the loader from the dataset
    sampler=None
    data_params = config_loader.get_param_value('data')
    batch_size = int(data_params.get('batch_size', 4))
    eval_batch_size = int(data_params.get('val_batch_size', 32))
    num_workers = int(data_params.get('workers', 0))

    # TPU specific data samplers and Loaders
    num_classes = dataset.NUM_CLASSES
    ignore_index = getattr(dataset, 'IGNORE_INDEX', None)
    # class_weights = dataset.get_label_weights()
    sampler = DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=True, pin_memory=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, sampler=val_sampler, num_workers=num_workers, drop_last=False, pin_memory=True)

    # Transfer the model to device
    net = net.to(device)

    # Training Hyperparameters
    train_params = config_loader.get_param_value('train')
    seed = int(train_params['random_seed'])
    num_epochs = int(train_params['num_epochs'])
    log_step = int(train_params['log_step'])
    
    # Loss params
    loss_params = config_loader.get_param_value('loss')
    loss_name = loss_params.pop('loss')
    kwargs = {k: v for k, v in loss_params.items()}
    loss_function = get_loss(loss_name, **kwargs).to(device)
    kwargs['ignore_index'] = ignore_index
    xm.master_print('---------- Loss Params ---------')
    xm.master_print(f'Using Loss Criterion: {loss_function}')
    xm.master_print(f'Class weights: {loss_function.weight}')
    xm.master_print(f'Ignore Index: {loss_function.ignore_index}')

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
    xm.master_print('---------- Optimizer Params ---------')
    xm.master_print(f'Using optimizer: {optimizer}')
    xm.master_print(f'Using LR Scheduler: {lr_scheduler}')

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
    xm.master_print('---------- Running training with the following params ---------')
    xm.master_print(f'Random Seed: {seed}')
    xm.master_print(f'Num epochs: {num_epochs}')
    xm.master_print(f'Log step: {log_step}')
    xm.master_print(f'Batch Size: {batch_size}')
    xm.master_print(f'Device: {device}')
    
    best_auc = 0.0
    save_dir_best = os.path.join(save_dir, 'best')
    # Main Training Loop
    tk0 = tq.tqdm(range(iter_val, num_epochs)) if xm.is_master_ordinal() else range(iter_val, num_epochs)
    for epoch_idx in tk0:
        avg_epoch_loss = _train_one_epoch(
            epoch_idx, loader, val_loader, net, loss_function, optimizer,
            log_step=log_step, device=device
        )
        lr_scheduler.step()
        loss_profile.append(avg_epoch_loss)

        # Evaluate auc on the val set
        avg_auc_score = evaluate(val_loader, net, device, ignore_index=ignore_index)
        avg_auc_score = xm.mesh_reduce('auc_reduce', avg_auc_score, lambda vals: sum(vals) / len(vals))
        if xm.is_master_ordinal():
            tk0.set_postfix({'Epoch': epoch_idx + 1, 'Epoch Train Loss': avg_epoch_loss, 'Avg AUC score': avg_auc_score})

        # Logging and State saving
        if xm.is_master_ordinal():
            if best_auc < avg_auc_score:
                best_iou = avg_auc_score
                save_checkpoint(epoch_idx, file_path, save_dir_best, 'tpu', net, optimizer, lr_scheduler, loss_profile)
            
            if epoch_idx % 10 == 0:
                save_checkpoint(epoch_idx, file_path, save_dir, 'tpu', net, optimizer, lr_scheduler, loss_profile)

        # Garbage collection
        gc.collect()


def evaluate(val_loader, net, device, ignore_index=-1):
    net.eval()
    loader = pl.ParallelLoader(val_loader, [device]).per_device_loader(device)
    tk0 = loader
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


def _train_one_epoch(epoch_idx, data_loader, val_loader, net, loss_criterion, optimizer, log_step=50, device='cpu', eval=False):
    loader = pl.ParallelLoader(data_loader, [device]).per_device_loader(device)
    tk0 = loader
    val_img_batch, val_target_batch = next(iter(val_loader))
    epoch_loss = 0
    for batch_idx, (img_batch, target_batch) in enumerate(tk0):
        # Zero out gradients
        optimizer.zero_grad()

        # Move the training sample to device
        net.train()
        data_batch = data_batch.to(device)
        target_batch = target_batch.to(device)

        # Compute Loss and optimize
        # out, decoded_preds, kl_loss = net(data_batch)
        out = net(data_batch)
        clf_loss = loss_criterion(out, target_batch)
        # decoder_loss = F.binary_cross_entropy_with_logits(decoded_preds, data_batch)
        # loss = clf_loss + decoder_loss + kl_loss
        # loss += (decoder_loss + kl_loss)
        loss += clf_loss
        loss.backward()
        xm.optimizer_step(optimizer)

        # Gather loss statistics from all cores
        loss_ = xm.mesh_reduce('loss_reduce', loss, lambda vals: sum(vals) / len(vals))
        loss_ = loss_.detach().cpu().numpy()
        epoch_loss += loss_

        # Check validation loss on a batch
        if eval:
            val_img_batch = val_img_batch.to(device)
            val_target_batch = val_target_batch.to(device)
            net.eval()
            with torch.no_grad():
                val_pred_batch = net(val_img_batch)['decoder']
                val_loss = loss_criterion(val_pred_batch, val_target_batch)
                val_loss = xm.mesh_reduce('single_val_reduce', val_loss, lambda vals: sum(vals) / len(vals))
                if xm.is_master_ordinal():
                    tk0.set_postfix_str(s=f'Validation Loss: {val_loss.cpu().numpy()}')

    return epoch_loss / len(data_loader)


if __name__ == "__main__":
    train()

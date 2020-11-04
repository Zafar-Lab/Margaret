# Parses the config file used for hyperparameter specification
import ast
import copy
import configparser
import gc
import numpy as np
import os
import re
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T

from ast import literal_eval
from PIL import Image
from tqdm import tqdm

from models import *
from util.datastore import get_dataset
from util.criterion import *

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.utils.serialization as xser
    _xla_available = True
except ImportError:
    _xla_available = False


_SUPPORTED_DEVICES = ['cpu', 'gpu', 'tpu']


class ConfigLoader:
    SUPPORTED_MODES = ['mandatory', 'optional']

    def __init__(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError('The config file could not be found at the specified path')

        config = configparser.ConfigParser()
        config.read(path)

        self.config = config

    def get_param_value(self, section, parameter=None, mode='mandatory'):
        """
        Returns the hyperparameter corresponding to the title and parameter in
        the config file
        """
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f'Unsupported Mode: `{mode}`')
        if section not in self.config:
            if mode == 'optional':
                return None
            raise ValueError(f'The section: `{section}` must be present in the config file')
        if parameter is None:
            # Return the list of all parameters in the section
            return self.config[section]

        if mode == 'optional':
            return self.config[section].get(parameter, None)
        else:
            try:
                return self.config[section][parameter]
            except:
                raise ValueError(f'The parameter `{parameter}` must be specified in section: `{section}`')


def configure_device(device):
    if device not in _SUPPORTED_DEVICES:
        raise NotImplementedError(f'The device type `{device}` is not supported')

    if device == 'gpu':
        if not torch.cuda.is_available():
            raise Exception('CUDA support is not available on your platform. Re-run using CPU or TPU mode')
        return 'cuda'

    if device == 'tpu':
        if _xla_available:
            return xm.xla_device()
        raise Exception('Install PyTorch XLA to use TPUs')
    return 'cpu'


def configure_model_params(config, infeatures, num_classes):
    # Log information about the Model params
    print('---------- Creating Model with the following specs ---------')
    kwargs = {}
    clf_params = config.get_param_value('clf')
    nodes = ast.literal_eval(clf_params.get('nodes', '[256, 256, 256]'))
    # net = WDNNResistancePredictor(infeatures, num_classes, nodes)
    net = DeepCNNResistancePredictor(infeatures, num_classes)
    print(f'Net: {net}')
    return net


def get_optimizer(name, net, lr, **kwargs):
    optim_cls = getattr(optim, name, None)
    kwargs = _eval_kwargs(kwargs)
    if optim_cls is None:
        raise ValueError(
            f"""The optimizer {name} is not supported by torch.optim.
            Refer to https://pytorch.org/docs/stable/optim.html#algorithms
            for an overview of the algorithms supported"""
        )
    return optim_cls(
        [{'params': filter(lambda p: p.requires_grad, net.parameters()), 'lr': lr }],
        lr=lr, **kwargs
    )


def get_loss(name, **kwargs):
    kwargs = _eval_kwargs(kwargs)
    weight = None
    if name == 'maskedbce':
        loss = MaskedBCEWithLogitsLoss(**kwargs)
    elif name == 'ohemmaskedbce':
        loss = OhemMaskedBCEWithLogitsLoss(**kwargs)
    elif name == 'maskedmargin':
        loss = MaskedMarginLoss(**kwargs)
    elif name == 'maskedmse':
        loss = MaskedMSELoss(**kwargs)
    elif name == 'maskedl1':
        loss = MaskedMSELoss(**kwargs)
    else:
        raise NotImplementedError(f'The loss {name} has not been implemented yet!')
    return loss


def get_lr_scheduler(optimizer, num_epochs, sched_type='poly', **kwargs):
    kwargs = _eval_kwargs(kwargs)
    if sched_type == 'poly':
        # A poly learning rate scheduler
        lambda_fn = lambda i: pow((1 - i / num_epochs), 0.9)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_fn)
    elif sched_type == 'step':
        # A Step learning rate scheduler
        step_size = kwargs['step_size']
        gamma = kwargs['gamma']
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_type == 'cosine':
        # Cosine learning rate annealing with Warm restarts
        T_0 = kwargs['t0']
        T_mul = kwargs.get('tmul', 1)
        eta_min = 0
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0, T_mult=T_mul, eta_min=eta_min
        )
    else:
        raise ValueError(f'The lr_scheduler type {sched_type} has not been implemented yet')


def load_checkpoint(checkpoint_path, net, device, optimizer=None, lr_scheduler=None):
    log = print
    if _xla_available and device not in ['cuda', 'cpu']:
        log = xm.master_print
    state_dict = torch.load(checkpoint_path)
    iter_val = state_dict.get('epoch', 0)
    loss_profile = state_dict.get('loss_profile', [])
    if 'model' in state_dict:
        log('Restoring Model state')
        net.load_state_dict(state_dict['model'])

    if optimizer is not None and 'optimizer' in state_dict:
        log('Restoring Optimizer state')
        optimizer.load_state_dict(state_dict['optimizer'])
        # manually move the optimizer state vectors to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    if lr_scheduler is not None and 'scheduler' in state_dict:
        log('Restoring Learning Rate scheduler state')
        lr_scheduler.load_state_dict(state_dict['scheduler'])

    return {
        'iter': iter_val,
        'loss_profile': loss_profile,
        'model': net,
        'optimizer': optimizer,
        'scheduler': lr_scheduler
    }


def save_checkpoint(epoch_id, file_path, save_dir, dev_name, model, optimizer, scheduler, loss_profile):
    config_loader = ConfigLoader(file_path)

    # Checkpoint Naming
    dataset = config_loader.get_param_value('data', 'dataset')
    optim = config_loader.get_param_value('optim', 'optimizer')
    epoch = epoch_id + 1
    checkpoint_name = f"chkpt_{dataset}_{optim}_e{epoch}"
    checkpoint_path = os.path.join(save_dir, f'{checkpoint_name}.pt')

    state_dict = {}
    model_state = copy.deepcopy(model.state_dict())
    model_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items()}
    optim_state = copy.deepcopy(optimizer.state_dict())
    for state in optim_state['state'].values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cpu()

    state_dict['model'] = model_state
    state_dict['optimizer'] = optim_state
    state_dict['scheduler'] = scheduler.state_dict()
    state_dict['epoch'] = epoch_id + 1
    state_dict['loss_profile'] = loss_profile
    state_dict['config'] = config_loader

    os.makedirs(save_dir, exist_ok=True)
    for f in os.listdir(save_dir):
        if f.endswith('.pt'):
            os.remove(os.path.join(save_dir, f))
    torch.save(state_dict, checkpoint_path)

    del model_state, optim_state
    gc.collect()


def _eval_kwargs(kwargs):
    for k, v in kwargs.items():
        try:
            kwargs[k] = ast.literal_eval(v)
        except ValueError:
            # A value error is thrown if the value is a str.
            continue
    return kwargs

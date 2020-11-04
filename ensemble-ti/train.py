# TODO: Integrate tensorboard here

import ast
import click
import copy
import gc
import logging
import numpy as np
import os
import torch
import warnings

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from util.datastore import get_dataset
from util.config import ConfigLoader, configure_device, get_optimizer, get_loss, configure_model_params, \
    save_checkpoint
from train_gpu import train_on_gpu


try:
    import torch_xla.distributed.xla_multiprocessing as xmp
    from train_tpu import train_on_tpu
    _xla_available = True
except ImportError:
    _xla_available = False

warnings.filterwarnings('ignore', category=UserWarning)


__all__ = ['train']


logging.basicConfig()
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)


@click.command()
@click.option('--checkpoint-path', '-c', help='Path to the checkpoint file to restore training state')
@click.option('--remove-previous-chkpt', '-r', default=True)
@click.option('--nprocs', '-n', default=8)
@click.argument('conf_path')
@click.argument('save_dir')
def train(conf_path, save_dir, checkpoint_path, remove_previous_chkpt=True, nprocs=8):
    """Trains a Mutation Resistance Prediction network

    Args:

        conf_path ([type]): Path to the model conf file

        save_dir ([type]): The directory in which to save the models after training ends
    """
    # Set tensor type
    torch.set_default_tensor_type('torch.FloatTensor')

    # Load the config
    config_loader = ConfigLoader(conf_path)
    dev_name = config_loader.get_param_value('train').get('device', 'cpu')

    # Set the random seed
    seed = int(config_loader.get_param_value('train').get('random_seed', 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if dev_name == 'tpu' and not _xla_available:
        raise Exception('Please install PyTorch XLA to use TPUs')

    # Load the dataset
    data_params = config_loader.get_param_value('data')
    dataset_name = data_params.pop('dataset').lower()
    root = data_params.pop('root')
    mode = data_params.pop('mode')
    eval_mode = data_params.pop('eval_mode')
    kwargs = {k: v for k, v in data_params.items()}
    dataset, num_classes = get_dataset(dataset_name, root, mode=mode, **kwargs)
    val_dataset, _ = get_dataset(dataset_name, root, mode=eval_mode, **kwargs)

    # Load the model
    net = configure_model_params(config_loader, dataset.shape[1], num_classes)

    # If the device type is TPU we use xmp.spawn to setup mutli-core tpu training
    # Otherwise we simply call the training function
    if dev_name == 'tpu':
        net = xmp.MpModelWrapper(net)
        args = [conf_path, save_dir, checkpoint_path, net, dataset, val_dataset, remove_previous_chkpt]
        xmp.spawn(
            train_on_tpu,
            args=(*args,),
            nprocs=nprocs,
            start_method='fork'
        )
        return
    args = [conf_path, save_dir, checkpoint_path, net, dataset, val_dataset]
    kwargs = {
        'remove_previous_chkpt': remove_previous_chkpt,
    }
    train_on_gpu(*args, **kwargs)


if __name__ == "__main__":
    train()

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

from ast import literal_eval
from PIL import Image
from tqdm import tqdm

from models import *
from utils.criterion import *


_SUPPORTED_DEVICES = ["cpu", "cuda"]


def configure_device(device):
    if device not in _SUPPORTED_DEVICES:
        raise NotImplementedError(f"The device type `{device}` is not supported")

    if device == "cuda":
        if not torch.cuda.is_available():
            raise Exception(
                "CUDA support is not available on your platform. Re-run using CPU or TPU mode"
            )
        return "cuda"

    if device == "tpu":
        if _xla_available:
            return xm.xla_device()
        raise Exception("Install PyTorch XLA to use TPUs")
    return "cpu"


def get_optimizer(name, net, lr, **kwargs):
    optim_cls = getattr(optim, name, None)
    if optim_cls is None:
        raise ValueError(
            f"""The optimizer {name} is not supported by torch.optim.
            Refer to https://pytorch.org/docs/stable/optim.html#algorithms
            for an overview of the algorithms supported"""
        )
    return optim_cls(
        [{"params": filter(lambda p: p.requires_grad, net.parameters()), "lr": lr}],
        lr=lr,
        **kwargs,
    )


def get_loss(name, **kwargs):
    weight = None
    if name == "mse":
        loss = nn.MSELoss(**kwargs)
    elif name == "vae":
        loss = VAELoss(**kwargs)
    else:
        raise NotImplementedError(f"The loss {name} has not been implemented yet!")
    return loss


def get_lr_scheduler(optimizer, num_epochs, sched_type="poly", **kwargs):
    if sched_type == "poly":
        # A poly learning rate scheduler
        lambda_fn = lambda i: pow((1 - i / num_epochs), 0.9)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_fn)
    elif sched_type == "step":
        # A Step learning rate scheduler
        step_size = kwargs["step_size"]
        gamma = kwargs["gamma"]
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif sched_type == "cosine":
        # Cosine learning rate annealing with Warm restarts
        T_0 = kwargs["t0"]
        T_mul = kwargs.get("tmul", 1)
        eta_min = 0
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0, T_mult=T_mul, eta_min=eta_min
        )
    else:
        raise ValueError(
            f"The lr_scheduler type {sched_type} has not been implemented yet"
        )

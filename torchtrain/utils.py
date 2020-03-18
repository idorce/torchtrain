import itertools
import random
from datetime import datetime

import numpy as np
import torch


def filter_dict(d, to_save):
    """Filter configs to save as tensorboard hparams."""

    def can_store(k):
        return type(k) in {int, float, str, bool, torch.Tensor}

    return (
        {k: v for k, v in d.items() if ((k in set(to_save)) and can_store(k))}
        if to_save
        else {k: v for k, v in d.items() if can_store(k)}
    )


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def is_list_tuple(var):
    return isinstance(var, list) or isinstance(var, tuple)


def to_device(var, device):
    if is_list_tuple(var):
        var = list(t.to(device) if torch.is_tensor(t) else t for t in var)
    else:
        var = var.to(device)
    return var


def distribute_model(model, device, cuda_list):
    if device != "cpu":
        model = torch.nn.DataParallel(model, device_ids=eval(cuda_list))
        model = model.to(device)
    return model


def get_device(cuda_list):
    """Receive a string like '2,3'."""
    return (
        "cuda:" + cuda_list[0] if (torch.cuda.is_available() and cuda_list) else "cpu"
    )


def now(format="%Y%m%d_%H_%M_%S"):
    return datetime.now().strftime(format)


def repeat(data_loader):
    """Repeat dataloader. Create infinite loop."""
    for loader in itertools.repeat(data_loader):
        for batch in loader:
            yield batch


def every_n_steps(step, n):
    """Step starts from 0."""
    return (step + 1) % n == 0

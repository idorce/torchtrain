import numpy as np
import torch


def filter_dict(d, to_save):
    return {k: v for k, v in d.items() if k in to_save} if to_save else dict(d)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_random_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def one_if_not_set(config, names):
    for name in names:
        config[name] = 1 if config[name] < 1 else int(config[name])
    return config

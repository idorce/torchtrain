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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_random_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def one_if_not_set(config, names):
    for name in names:
        config[name] = 1 if config[name] < 1 else int(config[name])
    return config

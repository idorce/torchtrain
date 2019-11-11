import numpy as np
import torch


def distribute_model(model, config):
    if config["device"] != "cpu":
        model = torch.nn.DataParallel(
            model, device_ids=eval(config["cuda_list"])
        )
        model = model.to(config["device"])
    return model


def filter_dict(d, to_save):
    return {k: v for k, v in d.items() if k in to_save} if to_save else dict(d)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_random_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

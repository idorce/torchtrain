import numpy as np
import torch


def load_model(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    return model


def prepare_model(model, config):
    """Distribute and load model."""
    if config["device"] != "cpu":
        model = torch.nn.DataParallel(
            model, device_ids=eval(config["cuda_list"])
        )
        model = model.to(config["device"])
    if config["start_epoch"] > 1 or config["start_ckp_path"]:
        model = load_model(model, config["start_ckp_path"])
    return model


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

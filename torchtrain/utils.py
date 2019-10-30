import os
from collections import defaultdict
from datetime import datetime

import torch


def append_config(config):
    config = defaultdict(bool, config)
    config["device"] = (
        "cuda:" + config["cuda_list"][0]
        if (torch.cuda.is_available() and config["cuda_list"])
        else "cpu"
    )
    config["save_path"] = os.path.join(
        config["save_path"], datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    )
    config["checkpoint_path"] = os.path.join(
        config["save_path"], "checkpoint.pt"
    )
    return config


def distribute_model(model, config):
    if config["device"] != "cpu":
        model = torch.nn.DataParallel(
            model, device_ids=eval(config["cuda_list"])
        )
        model = model.to(config["device"])
    return model


def filter_dict(d, to_save):
    return {k: v for k, v in d.items() if k in to_save} if to_save else dict(d)

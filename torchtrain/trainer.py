import os
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .callbacks import EarlyStop


class Trainer:
    """Supervised trainer.

    Args
    ----

    config (`dict`):
        'max_train_epoch' (int):
        'early_stop_patience' (int):
        'cuda_list' (str):
            e.g. '1,3'.
        'save_path' (str):
            Create a subfolder using current datetime.
            Best checkpoint and tensorboard logs are saved inside.
        'early_stop_verbose' (bool, optional):
            If True, print verbose message. Default to False.
        'tqdm' (bool, optional):
            If True, tqdm progress bar for batch iteration. Default to False.

    data_iter (`dict`):
        'train', 'val', 'test' (iterator):

    model, optimizer, scheduler (`torch`):
        PyTorch model, optimizer, scheduler (optional).

    criteria (`dict`): Other criterions will be calculated as well.
        'loss' (function):
            Calculate loss for `backward()`.
    """

    def __init__(
        self, config, data_iter, model, optimizer, criteria, scheduler=None
    ):
        self.config = self.append_config(config)
        self.data_iter = data_iter
        self.model = self.distribute_model(model)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criteria = criteria
        self.writer = SummaryWriter(self.config["save_path"])

    def append_config(self, config):
        config = defaultdict(bool, config)
        config["device"] = (
            "cuda:" + config["cuda_list"][0]
            if (torch.cuda.is_available() and config["cuda_list"])
            else "cpu"
        )
        config["save_path"] = os.path.join(
            config["save_path"], datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        )
        config["checkpoint_best_path"] = os.path.join(
            config["save_path"], "checkpoint_best.pt"
        )
        config["checkpoint_end_path"] = os.path.join(
            config["save_path"], "checkpoint_end.pt"
        )
        return config

    def distribute_model(self, model):
        if self.config["device"] != "cpu":
            model = nn.DataParallel(
                model, device_ids=eval(self.config["cuda_list"])
            )
            model = model.to(self.config["device"])
        return model

    def iter_batch(self, phase, epoch=1):
        is_train = phase == "train"
        self.model.train(is_train)
        metrics_sum = {name: 0 for name, criterion in self.criteria.items()}
        iter_count = 0
        t = self.data_iter[phase]
        if self.config["tqdm"]:
            t = tqdm(self.data_iter[phase], desc=phase)
        for batch in t:
            iter_count += 1
            if is_train:
                self.optimizer.zero_grad()
            x, y = batch
            x = x.to(self.config["device"])
            y = y.to(self.config["device"])
            metrics = {}
            desc = f" epoch: {epoch:3d} "
            with torch.set_grad_enabled(is_train):
                pred = self.model(x)
                for name, criterion in self.criteria.items():
                    metric = criterion(pred, y)
                    metrics[name] = metric
                    metrics_sum[name] += metric.item()
                    desc += f"{name}_{phase:5s}: {metric.item():.6f} "
            if is_train:
                metrics["loss"].backward()
                self.optimizer.step()
            if self.config["tqdm"]:
                t.set_description(desc)
        metrics_avg = {}
        desc = f" epoch: {epoch:3d} "
        for name, v in metrics_sum.items():
            metric_avg = v / iter_count
            metrics_avg[f"{name}/{phase}"] = metric_avg
            desc += f"{name}_{phase:5s}: {metric_avg:.6f} "
            self.writer.add_scalar(f"{name}/{phase}", metric_avg, epoch)
        self.writer.close()
        if self.config["tqdm"]:
            t.set_description(desc)
        return metrics_avg

    def train(self):
        early_stopper = EarlyStop(
            self.config["early_stop_patience"],
            verbose=self.config["early_stop_verbose"],
        )
        metrics_train_best = {}
        metrics_val_best = {}
        for epoch in range(1, self.config["max_train_epoch"] + 1):
            metrics_train = self.iter_batch("train", epoch)
            metrics_val = self.iter_batch("val", epoch)
            loss_val = metrics_val["loss/val"]
            self.writer.add_scalar(
                "lr",
                [group["lr"] for group in self.optimizer.param_groups][0],
                epoch,
            )
            if self.scheduler:
                self.scheduler.step(loss_val)
            signal = early_stopper.check(loss_val)
            if signal == "stop":
                break
            elif signal == "best":
                torch.save(
                    self.model.state_dict(),
                    self.config["checkpoint_best_path"],
                )
                metrics_train_best = metrics_train
                metrics_val_best = metrics_val
        torch.save(self.model.state_dict(), self.config["checkpoint_end_path"])
        self.writer.add_hparams(
            dict(self.config), {**metrics_train_best, **metrics_val_best}
        )
        self.writer.close()

    def test(self, checkpoint_path=None):
        if not checkpoint_path:
            checkpoint_path = self.config["checkpoint_best_path"]
        self.model.load_state_dict(torch.load(checkpoint_path))
        metrics_test = self.iter_batch("test")
        return metrics_test

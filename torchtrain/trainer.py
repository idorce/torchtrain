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
        'save_end_state' (bool, optional):
            If True, model end state will be saved. Default to False.
    data_iter (`dict`):
        'train', 'val', 'test' (iterator):
            Data iterators should be on the right device beforehand.
    model, optimizer, scheduler (`torch`):
        PyTorch model, optimizer, scheduler (optional).
    criteria (`dict`): Other criterions will be calculated as well.
        'loss' (function):
            Calculate loss for `backward()`.
    hparams_to_save, metrics_to_save (`list[str]`):
        Save to tensorboard hparams. Default to save all of config.
    """

    def __init__(
        self,
        config,
        data_iter,
        model,
        optimizer,
        criteria,
        scheduler=None,
        hparams_to_save=None,
        metrics_to_save=None,
    ):
        self.data_iter = data_iter
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criteria = criteria
        self.hparams_to_save = hparams_to_save
        self.metrics_to_save = metrics_to_save
        self.config = self.append_config(config)
        self.model = self.distribute_model(model)
        self.writer = SummaryWriter(self.config["save_path"])

    def append_config(self, config):
        config = defaultdict(bool, config)
        config["device"] = (
            "cuda:" + config["cuda_list"][0]
            if (torch.cuda.is_available() and config["cuda_list"])
            else "cpu"
        )
        run_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if self.hparams_to_save:
            run_name += "_" + "_".join(
                [str(config[hparam]) for hparam in self.hparams_to_save]
            )
        config["save_path"] = os.path.join(config["save_path"], run_name)
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
            inputs, labels = batch
            metrics = {}
            desc = f" epoch: {epoch:3d} "
            with torch.set_grad_enabled(is_train):
                pred = self.model(inputs)
                for name, criterion in self.criteria.items():
                    metric = criterion(pred, labels)
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

    def filter_dict(self, d, to_save):
        return (
            {k: v for k, v in d.items() if k in to_save}
            if to_save
            else dict(d)
        )

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
        if self.config["save_end_state"]:
            torch.save(
                self.model.state_dict(), self.config["checkpoint_end_path"]
            )
        metrics_best = {**metrics_val_best, **metrics_train_best}
        self.writer.add_hparams(
            self.filter_dict(self.config, self.hparams_to_save),
            self.filter_dict(metrics_best, self.metrics_to_save),
        )
        self.writer.close()

    def test(self, checkpoint_path=None):
        if not checkpoint_path:
            checkpoint_path = self.config["checkpoint_best_path"]
        self.model.load_state_dict(torch.load(checkpoint_path))
        metrics_test = self.iter_batch("test")
        return metrics_test

import os
from collections import defaultdict
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from . import utils
from .callbacks import EarlyStop


class Trainer:
    """Supervised trainer.

    Parameters
    ----------
    config : dict
        'max_train_epoch' : int
        'early_stop_patience' : int
        'watching_metric' : str
            Metric to monitor for early stop and lr scheduler.
        'watch_mode' : str, ['min', 'max']
        'cuda_list' : str
            E.g. '1,3', ','. Will be used like `config["cuda_list"][0]` and
            `eval(config["cuda_list"])`.
        'save_path' : str
            Create a subfolder using current datetime.
            Best checkpoint and tensorboard logs are saved inside.
        'early_stop_verbose' : bool, optional
            If True, early stop print verbose message. Default to False.
        'tqdm' : bool, optional
            If True, tqdm progress bar for batch iteration. Default to False.
        'grad_accumulate_batch' : int, optional
           Accumulate gradient for given batches, then backward. Default to 1.
        'train_one_epoch' : bool, optional
            If True, only train one epoch for testing code. Default to False.
        'start_epoch', 'start_ckp_path' : int, str, optional
            start_epoch is default to 1, otherwise must specify start_ckp_path.
        'grad_clip_norm' : float, optional
            If greater than 0, apply gradient clipping. Default to 0.
    data_iter : dict
        'train', 'val', 'test' : iterator
            Data iterators should be on the right device beforehand.
    model, optimizer : torch
        PyTorch model, optimizer.
    criteria : dict
        Other criterions will be calculated as well.
        'loss' : callable
            Calculate loss for `backward()`.
    scheduler : torch, optional
        PyTorch scheduler.
    hparams_to_save, metrics_to_save : list[str]
        Save to tensorboard hparams. Default to not save hparams.
    batch_to_xy : callable
        Will be used as `inputs, labels = self.batch_to_xy(batch, phase)`.
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
        batch_to_xy=lambda batch, phase: batch,
    ):
        self.config = config
        self.data_iter = data_iter
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criteria = criteria
        self.hparams_to_save = hparams_to_save
        self.metrics_to_save = metrics_to_save
        self.batch_to_xy = batch_to_xy
        self.config = self.configure(self.config)
        self.model = utils.prepare_model(self.model, self.config)
        self.writer = SummaryWriter(self.config["save_path"])

    def configure(self, config):
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
        if config["train_one_epoch"]:
            config["max_train_epoch"] = 1
        config = utils.one_if_not_set(
            config, ["grad_accumulate_batch", "start_epoch"]
        )
        config["n_parameters"] = utils.count_parameters(self.model)
        return config

    def current_stats(
        self, phase, epoch, tqdm_wrapper, reset=False, write=False
    ):
        metrics = {}
        desc = f" epoch: {epoch:3d} "
        for name, criterion in self.criteria.items():
            metric = criterion.get_value(reset)
            metrics[f"{name}/{phase}"] = metric
            desc += f"{name}_{phase:5s}: {metric:.6f} "
            if write:
                self.writer.add_scalar(f"{name}/{phase}", metric, epoch)
        self.writer.flush()
        tqdm_wrapper.set_description(desc)
        return metrics

    def optim_step(self):
        if self.config["grad_clip_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config["grad_clip_norm"]
            )
        self.optimizer.step()
        self.optimizer.zero_grad()

    def iter_batch(self, phase, epoch=1):
        is_train = phase == "train"
        self.model.train(is_train)
        data = tqdm(self.data_iter[phase], disable=not self.config["tqdm"])
        self.optimizer.zero_grad()
        for batch in data:
            inputs, labels = self.batch_to_xy(batch, phase)
            with torch.set_grad_enabled(is_train):
                outputs = self.model(inputs)
                for criterion in self.criteria.values():
                    criterion.update(outputs, labels)
            if is_train:
                self.criteria["loss"].get_batch_score().backward()
                if (data.n + 1) % self.config["grad_accumulate_batch"] == 0:
                    self.optim_step()
            self.current_stats(phase, epoch, data)
        self.optim_step()
        return self.current_stats(phase, epoch, data, reset=True, write=True)

    def schedule_lr(self, epoch, metrics):
        if self.scheduler:
            metric = metrics[self.config["watching_metric"]]
            self.writer.add_scalar(
                "lr",
                [group["lr"] for group in self.optimizer.param_groups][0],
                epoch,
            )
            self.scheduler.step(metric)

    def save_hparams(self, best_metrics):
        if self.hparams_to_save:
            self.writer.add_hparams(
                utils.filter_dict(self.config, self.hparams_to_save),
                utils.filter_dict(best_metrics, self.metrics_to_save),
            )
            self.writer.flush()

    def train(self):
        early_stopper = EarlyStop(self.config, self.model)
        for epoch in range(
            self.config["start_epoch"], self.config["max_train_epoch"] + 1
        ):
            metrics = {
                **self.iter_batch("train", epoch),
                **self.iter_batch("val", epoch),
            }
            if early_stopper.check(metrics):
                break
            self.schedule_lr(epoch, metrics)
        self.save_hparams(early_stopper.best_metrics)

    def test(self, checkpoint_path=None):
        if not checkpoint_path:
            checkpoint_path = self.config["checkpoint_path"]
        self.model = utils.load_model(self.model, checkpoint_path)
        metrics_test = self.iter_batch("test")
        self.writer.add_hparams(
            utils.filter_dict(self.config, self.hparams_to_save), metrics_test
        )
        self.writer.flush()
        return metrics_test

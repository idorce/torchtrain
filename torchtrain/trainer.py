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

    Args:
        config (dict):
            'max_train_epoch' (int):
            'early_stop_patience' (int):
            'watching_metric' (str): Metric to monitor for early stop and lr
                scheduler.
            'watch_mode' (str): 'min' or 'max'.
            'cuda_list' (str): Will be used like `config["cuda_list"][0]` and
                `eval(config["cuda_list"])`. E.g. '1,3', ','.
            'save_path' (str): Create a subfolder using current datetime.
                Best checkpoint and tensorboard logs are saved inside.
            'early_stop_verbose' (bool, optional): If True, early stop print
                verbose message. Default: False.
            'tqdm' (bool, optional): If True, tqdm progress bar for batch
                iteration. Default: False.
            'grad_accumulate_batch' (int, optional): Accumulate gradient for
                given batches, then backward. Default: 1.
            'train_one_epoch' (bool, optional): If True, only train one epoch
                for testing code. Default: False.
            'start_ckp_path' (str, optional): If specified, load checkpoint at
                the beginning.
            'grad_clip_norm' (float, optional): If greater than 0, apply
                gradient clipping. Default: 0.
        data_iter (dict): Keys are 'train', 'val', 'test'. Values are
            iterators. Data iterators should be on the right device beforehand.
        model (torch):
        optimizer (torch):
        criteria (dict): 'loss' (callable): Calculate loss for `backward()`.
            Other criterions will be calculated as well.
        scheduler (torch, optional):
        hparams_to_save (list[str], optional):
        metrics_to_save (list[str], optional): Save to tensorboard hparams.
            Default: not save hparams.
        batch_to_xy (callable, optional): Will be used as
            `inputs, labels = self.batch_to_xy(batch, phase)`.
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
        self.configure()
        self.writer = SummaryWriter(self.config["save_path"])
        if self.config["start_ckp_path"]:
            self.load_state_dict(self.config["start_ckp_path"])
        self.distribute_model()

    def configure(self):
        self.config = defaultdict(bool, self.config)
        self.config["device"] = (
            "cuda:" + self.config["cuda_list"][0]
            if (torch.cuda.is_available() and self.config["cuda_list"])
            else "cpu"
        )
        self.config["save_path"] = os.path.join(
            self.config["save_path"],
            datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
        )
        self.config["checkpoint_path"] = os.path.join(
            self.config["save_path"], "checkpoint.pt"
        )
        if self.config["train_one_epoch"]:
            self.config["max_train_epoch"] = 1
        self.config = utils.one_if_not_set(
            self.config, ["grad_accumulate_batch", "start_epoch"]
        )
        self.config["n_parameters"] = utils.count_parameters(self.model)

    def distribute_model(self):
        if self.config["device"] != "cpu":
            self.model = torch.nn.DataParallel(
                self.model, device_ids=eval(self.config["cuda_list"])
            )
            self.model.to(self.config["device"])

    def save_state_dict(self, epoch):
        state_dict = {
            "epoch": epoch,
            "model": self.model.module.state_dict(),  # DataParallel.module
            "optimizer": self.optimizer.state_dict(),
        }
        state_dict["scheduler"] = (
            self.scheduler.state_dict() if self.scheduler else None
        )
        for phase, data in self.data_iter.items():
            state_dict[phase] = (
                data.state_dict() if hasattr(data, "state_dict") else None
            )
        torch.save(state_dict, self.config["checkpoint_path"])

    def load_state_dict(self, checkpoint_path, model_only=False):
        def state_to_device(obj):
            for state in obj.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.config["device"])

        if not checkpoint_path:
            checkpoint_path = self.config["checkpoint_path"]
        state_dict = torch.load(checkpoint_path)
        self.model.load_state_dict(state_dict["model"])
        if model_only:
            return
        self.config["start_epoch"] = state_dict["epoch"]
        self.optimizer.load_state_dict(state_dict["optimizer"])
        state_to_device(self.optimizer)
        if self.scheduler:
            self.scheduler.load_state_dict(state_dict["scheduler"])
            state_to_device(self.scheduler)
        for phase, data in self.data_iter.items():
            if hasattr(data, "load_state_dict"):
                self.data_iter[phase].load_state_dict(state_dict[phase])

    def iter_batch(self, phase, epoch=1):
        """Iterate batches for one epoch."""

        def optim_step():
            if self.config["grad_clip_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["grad_clip_norm"]
                )
            self.optimizer.step()
            self.optimizer.zero_grad()

        def current_stats(reset=False, write=False):
            metrics = {}
            desc = f" epoch: {epoch:3d} "
            for name, criterion in self.criteria.items():
                metric = criterion.get_value(reset)
                metrics[f"{name}/{phase}"] = metric
                desc += f"{name}_{phase:5s}: {metric:.6f} "
                if write:
                    self.writer.add_scalar(f"{name}/{phase}", metric, epoch)
            self.writer.flush()
            data.set_description(desc)
            return metrics

        def one_batch(batch):
            inputs, labels = self.batch_to_xy(batch, phase)
            with torch.set_grad_enabled(is_train):
                outputs = self.model(inputs)
                for criterion in self.criteria.values():
                    criterion.update(outputs, labels)
            if is_train:
                self.criteria["loss"].get_batch_score().backward()
                if (data.n + 1) % self.config["grad_accumulate_batch"] == 0:
                    optim_step()
            current_stats()

        is_train = phase == "train"
        self.model.train(is_train)
        data = tqdm(self.data_iter[phase], disable=not self.config["tqdm"])
        self.optimizer.zero_grad()
        for batch in data:
            one_batch(batch)
        optim_step()
        return current_stats(reset=True, write=True)

    def train(self):
        """Train and validate."""

        def schedule_lr(epoch, metrics):
            if self.scheduler:
                metric = metrics[self.config["watching_metric"]]
                self.writer.add_scalar(
                    "lr",
                    [group["lr"] for group in self.optimizer.param_groups][0],
                    epoch,
                )
                self.scheduler.step(metric)

        def save_hparams():
            if self.hparams_to_save:
                self.writer.add_hparams(
                    utils.filter_dict(self.config, self.hparams_to_save),
                    utils.filter_dict(
                        early_stopper.best_metrics, self.metrics_to_save
                    ),
                )
                self.writer.flush()

        early_stopper = EarlyStop(self)
        for epoch in range(
            self.config["start_epoch"], self.config["max_train_epoch"] + 1
        ):
            metrics = {
                **self.iter_batch("train", epoch),
                **self.iter_batch("val", epoch),
            }
            if early_stopper.check(epoch, metrics):
                break
            schedule_lr(epoch, metrics)
        save_hparams()

    def test(self, checkpoint_path=None):
        self.load_state_dict(checkpoint_path, model_only=True)
        metrics_test = self.iter_batch("test")
        self.writer.add_hparams(
            utils.filter_dict(self.config, self.hparams_to_save), metrics_test
        )
        self.writer.flush()
        return metrics_test

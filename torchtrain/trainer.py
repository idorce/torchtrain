import os
import warnings
from collections import defaultdict

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from . import utils
from .callbacks import EarlyStop


class Trainer:
    """Supervised trainer.

    Args:
        model (torch): PyTorch nn.Module.
        data_iter (dict): Keys are 'train', 'val', 'test'. Value is iterator.
        criteria (dict): 'loss' (callable): Calculate loss for `backward()`.
            Other criterions will be calculated as well.
        cfg (dict, optional):
            'max_n' (int, optional): Max training epoch or step. Default: 1.
            'val_step' (int, optional): Validate on full val set every given
                steps. Default: 1.
            'save_path' (str, optional): Create a subfolder using current
                datetime. Best checkpoint and tensorboard logs are saved
                inside. Default: 'runs'
            'model_name' (str, optional): Create a subfolder using this.
                Default: No such subfolder created.
            'cuda_list' (str, optional): Used like `config["cuda_list"][0]` and
                `eval(config["cuda_list"])`. E.g. '1,3', ','. Default: ''.
            'patience' (int, optional): Patience for early stop. Default: inf.
            'watch_metric' (str, optional): Metric to monitor for early stop
                and lr scheduler. Default: 'loss/val'.
            'watch_mode' (str, optional): 'min' or 'max'. Default: 'min'.
            'verbose' (bool, optional): If True, print verbose message.
                Default: False.
            'tqdm' (bool, optional): If True, tqdm progress bar for batch
                iteration. Default: True.
            'grad_accum_batch' (int, optional): Accumulate gradient for
                given batches, then backward. Default: 1.
            'train_few' (bool, optional): If True, only train one epoch and
                few examples for testing code. Default: False.
            'start_ckp_path' (str, optional): If specified, load checkpoint at
                the beginning.
            'grad_clip_norm' (float, optional): If greater than 0, apply
                gradient clipping. Default: 0.
        optimizer (torch, optional): Default: Adam.
        scheduler (torch, optional): Default: None.
        hparams_to_save (list[str], optional): Save to tensorboard hparams tab.
            Default: Save all.
        metrics_to_save (list[str], optional): Save to tensorboard hparams tab
            as metrics. Default: Save all.
        batch_to_xy (callable, optional): Will be used as
            `inputs, labels = self.batch_to_xy(batch, phase)`.
    """

    def __init__(
        self,
        model,
        data_iter=None,
        criteria={},
        cfg={},
        optimizer=None,
        scheduler=None,
        hparams_to_save=None,
        metrics_to_save=None,
        batch_to_xy=lambda batch, phase: batch,
    ):
        self.cfg = cfg
        self.data_iter = data_iter
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criteria = criteria
        self.hparams_to_save = hparams_to_save
        self.metrics_to_save = metrics_to_save
        self.batch_to_xy = batch_to_xy
        self._configure()
        self.load_state_dict(self.cfg["start_ckp_path"])
        if self.data_iter:
            self.writer = SummaryWriter(self.cfg["save_path"])
            self._save_hparams()
        self.model = utils.distribute_model(
            self.model, self.cfg["device"], self.cfg["cuda_list"]
        )

    def _default(self, name, default=1):
        if name not in self.cfg:
            self.cfg[name] = default

    def _join_path(self, path1, append, path2=None):
        if path2 is None:
            path2 = path1
        self.cfg[path1] = os.path.join(self.cfg[path2], append)

    def _configure(self):
        self.cfg = defaultdict(bool, self.cfg)
        self._default("max_n", 1)
        self._default("val_step", 1)
        self._default("save_path", "runs")
        if self.cfg["train_few"]:
            self.cfg["max_n"] = 3
            self.cfg["val_step"] = 1
            self._join_path("save_path", "test")
        if "model_name" in self.cfg:
            self._join_path("save_path", self.cfg["model_name"])
        self._join_path("save_path", utils.now())
        self._join_path("ckp_path", "ckp", "save_path")
        self._default("cuda_list", "")
        self.cfg["device"] = utils.get_device(self.cfg["cuda_list"])
        self._default("patience", float("inf"))
        self._default("watch_metric", "loss/val")
        self._default("watch_mode", "min")
        self._default("tqdm", True)
        self._default("grad_accum_batch", 1)
        self.cfg["start_n"] = 1
        self.cfg["n_params"] = utils.count_params(self.model)
        if self.optimizer is None:
            self.optimizer = Adam(self.model.parameters())

    def _save_state_dict(self, n):
        state_dict = {
            "n": n,
            "model": self.model.module.state_dict(),  # DataParallel.module
            "optimizer": self.optimizer.state_dict(),
        }
        if self.scheduler:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Please also save or load the state of the optimzer when saving or loading the scheduler.",
                    category=UserWarning,
                )
                state_dict["scheduler"] = self.scheduler.state_dict()
        for phase, data in self.data_iter.items():
            if hasattr(data, "state_dict"):
                state_dict[phase] = data.state_dict()
        torch.save(state_dict, f"{self.cfg['ckp_path']}_{utils.now()}.pt")

    def load_state_dict(self, ckp_path=None, model_only=False, map_location=None):
        if not ckp_path:
            return
        state_dict = (
            torch.load(ckp_path, map_location=map_location)
            if torch.cuda.is_available()
            else torch.load(ckp_path, map_location="cpu")
        )
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(state_dict["model"])
        else:
            self.model.load_state_dict(state_dict["model"])
        if model_only:
            return self.model
        if "n" in state_dict:
            self.cfg["start_n"] = state_dict["n"] + 1
        if "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.cfg["device"])
        if self.scheduler and ("scheduler" in state_dict):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Please also save or load the state of the optimzer when saving or loading the scheduler.",
                    warnings=UserWarning,
                )
                self.scheduler.load_state_dict(state_dict["scheduler"])
        for phase, data in self.data_iter.items():
            if hasattr(data, "load_state_dict") and (phase in state_dict):
                self.data_iter[phase].load_state_dict(state_dict[phase])
        # Change save_path
        p = os.path.dirname(self.cfg["start_ckp_path"])
        if len(os.path.basename(p)) == len(utils.now()):
            self.cfg["save_path"] = p

    def _optim_step(self):
        # rescale grad
        if self.batch_size_accum != 0:
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad /= self.batch_size_accum
            self.batch_size_accum = 0
        if self.cfg["grad_clip_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg["grad_clip_norm"]
            )
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _one_batch(self, batch, phase, step):
        """Process one batch and propagate loss."""
        try:
            inputs, labels = self.batch_to_xy(batch, phase)
            inputs = utils.to_device(inputs, self.cfg["device"])
            labels = utils.to_device(labels, self.cfg["device"])
            self.model.train(phase == "train")
            with torch.set_grad_enabled(phase == "train"):
                outputs = (
                    self.model(*inputs)
                    if utils.is_list_tuple(inputs)
                    else self.model(inputs)
                )
            for name, criterion in self.criteria.items():
                with torch.set_grad_enabled(phase == "train" and name == "loss"):
                    criterion.update(outputs, labels)
            if phase == "train":
                if self.cfg["grad_accum_batch"] <= 1:
                    self.criteria["loss"].get_batch_score().backward()
                    self._optim_step()
                else:
                    batch_size = (
                        self.criterion.get_batch_size(labels)
                        if hasattr(self.criterion, "get_batch_size")
                        else len(labels)
                    )
                    self.batch_size_accum += batch_size
                    loss = self.criteria["loss"].get_batch_score() * batch_size
                    loss.backward()
                    if utils.every_n_steps(step, self.cfg["grad_accum_batch"]):
                        self._optim_step()
        except RuntimeError as e:
            if "out of memory" in str(e):
                self.optimizer.zero_grad()
                self.oom_batch_count[phase] += 1
            else:
                raise e

    def _stats_now(self, phase, n, tqdm_iter, reset=False, write=False):
        metrics = {}
        desc = f" n: {n:3d} "  # n is epoch or step
        for name, criterion in self.criteria.items():
            metric = criterion.get_value(reset)
            metrics[f"{name}/{phase}"] = metric
            desc += f"{name}/{phase:5s}: {metric:.6f} "
            if write:
                self.writer.add_scalar(f"{name}/{phase}", metric, n)
        if write:
            self.writer.add_scalar(f"oom/{phase}", self.oom_batch_count[phase], n)
        self.writer.flush()
        desc += f"oom: {self.oom_batch_count[phase]:3d} "
        tqdm_iter.set_description(desc)
        return metrics

    def _one_epoch(self, phase, epoch=1, leave_tqdm=True):
        """Iterate batches for one epoch."""
        self.optimizer.zero_grad()
        self.batch_size_accum = 0
        self.oom_batch_count[phase] = 0  # out of memory
        data = tqdm(
            self.data_iter[phase], leave=leave_tqdm, disable=not self.cfg["tqdm"]
        )
        for batch in data:
            if self.cfg["train_few"] and data.n >= 3:
                break
            self._one_batch(batch, phase, data.n)
            self._stats_now(phase, epoch, data)
        self._optim_step()
        return self._stats_now(phase, epoch, data, reset=True, write=True)

    def _schedule_lr(self, n, schedule_input=None):
        if self.scheduler:
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("lr", lr, n)
            if isinstance(self.scheduler, ReduceLROnPlateau):
                if schedule_input is not None:
                    self.scheduler.step(schedule_input)
            else:
                self.scheduler.step()

    def _save_hparams(self, metrics=None):
        hparams = utils.filter_dict(self.cfg, self.hparams_to_save)
        if metrics is None:
            cfg = "|cfg|value|\n|-|-|\n"
            cfg += "\n".join(f"|{k}|{v}|" for k, v in hparams.items())
            self.writer.add_text("cfg", cfg, 0)
        else:
            metrics = utils.filter_dict(metrics, self.metrics_to_save)
            self.writer.add_hparams(hparams, metrics)
        self.writer.flush()

    def train(self, stepwise=False):
        """Train and validate."""
        if stepwise:
            return self.train_stepwise()
        best_metrics = {}
        early_stopper = EarlyStop(
            self.cfg["patience"], self.cfg["watch_mode"], self.cfg["verbose"]
        )
        self.oom_batch_count = {"train": 0, "val": 0, "test": 0}
        for n in range(self.cfg["start_n"], self.cfg["max_n"] + 1):
            metrics = {**self._one_epoch("train", n), **self._one_epoch("val", n)}
            early_stopper.check(metrics[self.cfg["watch_metric"]])
            if early_stopper.best:
                self._save_state_dict(n)
                best_metrics = metrics
            elif early_stopper.stop:
                break
            self._schedule_lr(n, metrics[self.cfg["watch_metric"]])
        self._save_hparams(best_metrics)
        return best_metrics

    def train_stepwise(self):
        best_metrics = {}
        early_stopper = EarlyStop(
            self.cfg["patience"], self.cfg["watch_mode"], self.cfg["verbose"]
        )
        self.optimizer.zero_grad()
        self.batch_size_accum = 0
        self.oom_batch_count = {"train": 0, "val": 0, "test": 0}  # out of memory
        data = tqdm(
            utils.repeat(self.data_iter["train"]),
            total=self.cfg["max_n"],
            disable=not self.cfg["tqdm"],
            initial=self.cfg["start_n"] - 1,
        )
        for batch in data:
            if data.n >= self.cfg["max_n"]:
                break
            self._one_batch(batch, "train", data.n)
            metrics_train = self._stats_now(
                "train", data.n + 1, data, reset=True, write=True
            )
            if utils.every_n_steps(data.n, self.cfg["val_step"]):
                metrics_val = self._one_epoch("val", data.n + 1, leave_tqdm=None)
                metrics = {**metrics_train, **metrics_val}
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self._schedule_lr(data.n + 1, metrics[self.cfg["watch_metric"]])
                if self.cfg["verbose"]:
                    data.write(f"Val on step {data.n + 1:6d}: " + str(metrics))
                early_stopper.check(metrics[self.cfg["watch_metric"]])
                if early_stopper.best:
                    self._save_state_dict(data.n + 1)
                    best_metrics = metrics
                elif early_stopper.stop:
                    break
            self._schedule_lr(data.n + 1)
        self._optim_step()
        self._save_hparams(best_metrics)
        return best_metrics

    def test(self, ckp_path, dataset="val"):
        self.load_state_dict(ckp_path, model_only=True)
        self.oom_batch_count = {"train": 0, "val": 0, "test": 0}
        metrics_test = self._one_epoch(dataset)
        self._save_hparams(metrics_test)
        return metrics_test

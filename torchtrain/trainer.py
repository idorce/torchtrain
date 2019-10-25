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

    args (`dict`):
        'max_train_epoch' (int):
        'early_stop_patience' (int):
        'cuda_list' (list[int]):
        'save_path' (str):
            Create a subfolder using current datetime.
            Best checkpoint and tensorboard logs are saved inside.
        'early_stop_verbose' (bool, optional):
        'torchtext' (bool, optional):
            If True, `batch_to_xy()` will `return batch.text, batch.label`.

    data_iter (`dict`):
        'train', 'val', 'test' (iterator):

    criteria (`dict`): Other criterions will be calculated as well.
        'loss' (function):
            Calculate loss for `backward()`.
    """

    def __init__(
        self, args, data_iter, model, optimizer, criteria, scheduler=None
    ):
        self.args = self.append_args(args)
        self.data_iter = data_iter
        self.model = self.distribute_model(model)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criteria = criteria
        self.writer = SummaryWriter(self.args["save_path"])

    def append_args(self, args):
        args = defaultdict(bool, args)
        args["device"] = (
            "cuda:" + str(args["cuda_list"][0])
            if (torch.cuda.is_available() and args["cuda_list"])
            else "cpu"
        )
        args["save_path"] = os.path.join(
            args["save_path"], datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        )
        args["checkpoint_best_path"] = os.path.join(
            args["save_path"], "checkpoint_best.pt"
        )
        args["checkpoint_end_path"] = os.path.join(
            args["save_path"], "checkpoint_end.pt"
        )
        return args

    def distribute_model(self, model):
        if self.args["device"] != "cpu":
            model = nn.DataParallel(model, device_ids=self.args["cuda_list"])
            model = model.to(self.args["device"])
        return model

    def iter_batch(self, phase, epoch=1):
        is_train = phase == "train"
        metrics_sum = {name: 0 for name, criterion in self.criteria.items()}
        iter_count = 0
        t = tqdm(self.data_iter[phase], desc=phase)
        for batch in t:
            iter_count += 1
            if is_train:
                self.optimizer.zero_grad()
            # x, y = self.bacth_to_xy(batch)
            x, y = batch
            x = x.to(self.args["device"])
            y = y.to(self.args["device"])
            self.model.train(is_train)
            metrics = {}
            desc = " "
            with torch.set_grad_enabled(is_train):
                pred = self.model(x)
                for name, criterion in self.criteria.items():
                    metric = criterion(pred, y)
                    metrics[name] = metric
                    metrics_sum[name] += metric.item()
                    desc += f"{name}_{phase}: {metric.item():.6f} "
            if is_train:
                metrics["loss"].backward()
                self.optimizer.step()
            t.set_description(desc)
        metrics_avg = {}
        desc = " "
        for name, v in metrics_sum.items():
            metric_avg = v / iter_count
            metrics_avg[f"{name}_{phase}"] = metric_avg
            desc += f"{name}_{phase}: {metric_avg:.6f} "
            self.writer.add_scalar(f"{name}/{phase}", metric_avg, epoch)
        t.set_description(desc)
        return metrics_avg

    def train(self):
        early_stopper = EarlyStop(
            self.args["early_stop_patience"],
            verbose=self.args["early_stop_verbose"],
        )
        metrics_train_best = {}
        metrics_val_best = {}
        for epoch in range(1, self.args["max_train_epoch"] + 1):
            metrics_train = self.iter_batch("train", epoch)
            metrics_val = self.iter_batch("val", epoch)
            loss_val = metrics_val["loss_val"]
            self.writer.add_scalar(
                "lr",
                [group["lr"] for group in self.optimizer.param_groups][0],
                epoch,
            )
            self.scheduler.step(loss_val)
            signal = early_stopper.check(loss_val)
            if signal == "stop":
                break
            elif signal == "best":
                torch.save(
                    self.model.state_dict(), self.args["checkpoint_best_path"]
                )
                metrics_train_best = metrics_train
                metrics_val_best = metrics_val
        torch.save(self.model.state_dict(), self.args["checkpoint_end_path"])
        self.writer.add_hparams(
            dict(self.args), {**metrics_train_best, **metrics_val_best}
        )

    def test(self, checkpoint_path=None):
        if not checkpoint_path:
            checkpoint_path = self.args["checkpoint_best_path"]
        self.model.load_state_dict(torch.load(checkpoint_path))
        metrics_test = self.iter_batch("test")
        return metrics_test

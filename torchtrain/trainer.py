import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from . import utils
from .callbacks import EarlyStop


class Trainer:
    """Supervised trainer.
    Args
    ----
    config (`dict`):
        'max_train_epoch' (int):
        'early_stop_patience' (int):
        'early_stop_metric' (str):
        'early_stop_mode' (str, ['min', 'max']):
        'cuda_list' (str):
            e.g. '1,3'.
        'save_path' (str):
            Create a subfolder using current datetime.
            Best checkpoint and tensorboard logs are saved inside.
        'early_stop_verbose' (bool, optional):
            If True, early stop print verbose message. Default to False.
        'tqdm' (bool, optional):
            If True, tqdm progress bar for batch iteration. Default to False.
    data_iter (`dict`):
        'train', 'val', 'test' (iterator):
            Data iterators should be on the right device beforehand.
    model, optimizer, scheduler (`torch`):
        PyTorch model, optimizer, scheduler (optional).
    criteria (`dict`): Other criterions will be calculated as well.
        'loss' (function):
            Calculate loss for `backward()`.
    hparams_to_save, metrics_to_save (`list[str]`):
        Save to tensorboard hparams. Default to not save hparams.
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
        self.config = utils.append_config(config)
        self.model = utils.distribute_model(model, config)
        self.writer = SummaryWriter(self.config["save_path"])

    def iter_batch(self, phase, epoch=1):
        iter_count = 0
        is_train = phase == "train"
        self.model.train(is_train)
        metrics_sum = {name: 0 for name, criterion in self.criteria.items()}
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
        self.writer.flush()
        if self.config["tqdm"]:
            t.set_description(desc)
        return metrics_avg

    def train(self):
        early_stopper = EarlyStop(
            self.config["early_stop_patience"],
            self.config["early_stop_mode"],
            verbose=self.config["early_stop_verbose"],
        )
        metrics_best = {}
        for epoch in range(1, self.config["max_train_epoch"] + 1):
            metrics_train = self.iter_batch("train", epoch)
            metrics_val = self.iter_batch("val", epoch)
            metrics = {**metrics_train, **metrics_val}
            metric = metrics[self.config["early_stop_metric"]]
            signal = early_stopper.check(metric)
            if signal == "stop":
                break
            elif signal == "best":
                torch.save(
                    self.model.state_dict(), self.config["checkpoint_path"]
                )
                metrics_best = metrics
            if self.scheduler:
                self.writer.add_scalar(
                    "lr",
                    [group["lr"] for group in self.optimizer.param_groups][0],
                    epoch,
                )
                self.scheduler.step(metric)
        if self.hparams_to_save:
            self.writer.add_hparams(
                utils.filter_dict(self.config, self.hparams_to_save),
                utils.filter_dict(metrics_best, self.metrics_to_save),
            )
        self.writer.flush()

    def test(self, checkpoint_path=None):
        if not checkpoint_path:
            checkpoint_path = self.config["checkpoint_path"]
        self.model.load_state_dict(torch.load(checkpoint_path))
        metrics_test = self.iter_batch("test")
        self.writer.add_hparams(
            utils.filter_dict(self.config, self.hparams_to_save), metrics_test
        )
        self.writer.flush()
        return metrics_test

# 🔥 torchtrain 💪

A small tool for PyTorch training.

## Features

- Avoid boilerplate code for training.
- Stepwise training.
- Automatic TensorBoard logging, and tqdm bar.
- Count model parameters and save hyperparameters.
- DataParallel.
- Early stop.
- Save and load checkpoint. Continue training.
- Catch out of memory exceptions to avoid breaking training.
- Gradient accumulation.
- Gradient clipping.
- Only run few epochs, steps and batches for code test.

## Install

```
pip install torchtrain
```

## Example

Check doc string of [`Trainer` class](https://github.com/idorce/torchtrain/blob/master/torchtrain/trainer.py) for detailed configurations.

An incomplete minimal example:

```python
data_iter = get_data()
model = Bert()
optimizer = Adam(model.parameters(), lr=cfg["lr"])
criteria = {"loss": AverageAggregator(BCELoss())}
trainer = Trainer(cfg, data_iter, model, optimizer, criteria)
trainer.train(stepwise=True)
```

Or this version:

```python
from argparse import ArgumentParser

from sklearn.model_selection import ParameterGrid
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModel, BertTokenizer

from data.load import get_batch_size, get_data
from metrics import BCELoss
from models import BertSumExt
from torchtrain import Trainer
from torchtrain.metrics import AverageAggregator
from torchtrain.utils import set_random_seeds


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=233666)
    parser.add_argument("--run_ckp", default="")
    parser.add_argument("--run_dataset", default="val")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10000)
    parser.add_argument("--stepwise", action="store_false")
    # torchtrain cfgs
    parser.add_argument("--max_n", type=int, default=50000)
    parser.add_argument("--val_step", type=int, default=1000)
    parser.add_argument("--save_path", default="/tmp/runs")
    parser.add_argument("--model_name", default="BertSumExt")
    parser.add_argument("--cuda_list", default="2,3")
    parser.add_argument("--grad_accum_batch", type=int, default=1)
    parser.add_argument("--train_few", action="store_true")
    return vars(parser.parse_args())


def get_param_grid():
    param_grid = [
        {"pretrained_model_name": ["voidful/albert_chinese_tiny"], "lr": [6e-5]},
    ]
    return ParameterGrid(param_grid)


def get_cfg(args={}, params={}):
    cfg = {**args, **params}
    # other cfgs
    return cfg


def run(cfg):
    set_random_seeds(cfg["seed"])
    tokenizer = BertTokenizer.from_pretrained(cfg["pretrained_model_name"])
    bert = AutoModel.from_pretrained(cfg["pretrained_model_name"])
    data_iter = get_data(
        cfg["batch_size"], tokenizer, bert.config.max_position_embeddings
    )
    model = BertSumExt(bert)
    optimizer = Adam(model.parameters(), lr=cfg["lr"])
    scheduler = LambdaLR(
        optimizer,
        lambda step: min(step ** (-0.5), step * (cfg["warmup"] ** (-1.5)))
        if step > 0
        else 0,
    )
    criteria = {"loss": AverageAggregator(BCELoss())}
    trainer = Trainer(
        cfg,
        data_iter,
        model,
        optimizer,
        criteria,
        scheduler,
        get_batch_size=get_batch_size,
    )
    if cfg["run_ckp"]:
        return trainer.test(cfg["run_ckp"], cfg["run_dataset"])
    return trainer.train(stepwise=cfg["stepwise"])


def main():
    param_grid = get_param_grid()
    for i, params in enumerate(param_grid):
        print("Config", str(i + 1), "/", str(len(param_grid)))
        cfg = get_cfg(get_args(), params)
        metrics = run(cfg)
        print("Best metrics:", metrics)


if __name__ == "__main__":
    main()
```
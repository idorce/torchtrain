import torch


class EarlyStop:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.patience = self.config["early_stop_patience"]
        self.best_metric = (
            float("inf")
            if self.config["watch_mode"] == "min"
            else float("-inf")
        )
        self.best_metrics = {}

    def save_model(self):
        torch.save(self.model.state_dict(), self.config["checkpoint_path"])

    def check(self, metrics):
        metric = metrics[self.config["watching_metric"]]
        if self.config["watch_mode"] == "min":
            best = metric < self.best_metric
        elif self.config["watch_mode"] == "max":
            best = metric > self.best_metric
        else:
            raise ValueError("watch_mode can only be 'min' or 'max'.")
        if best:
            self.best_metric = metric
            self.patience = self.config["early_stop_patience"]
            if self.config["early_stop_verbose"]:
                print("Save best-so-far model state_dict...")
            self.save_model()
            self.best_metrics = metrics
        else:
            self.patience -= 1
        stop = self.patience == 0
        if stop and self.config["early_stop_verbose"]:
            print(f"Early stopped! Patience is {self.patience}.")
        return stop

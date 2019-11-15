class EarlyStop:
    def __init__(self, trainer):
        self.trainer = trainer
        self.config = trainer.config
        self.patience = self.config["early_stop_patience"]
        self.best_metric = (
            float("inf")
            if self.config["watch_mode"] == "min"
            else float("-inf")
        )
        self.best_metrics = {}

    def check(self, epoch, metrics):
        metric = metrics[self.config["watching_metric"]]
        best = (
            metric < self.best_metric
            if self.config["watch_mode"] == "min"
            else metric > self.best_metric
        )
        if best:
            self.best_metric = metric
            self.patience = self.config["early_stop_patience"]
            if self.config["early_stop_verbose"]:
                print("Save best-so-far model state_dict...")
            self.trainer.save_state_dict(epoch)
            self.best_metrics = metrics
        else:
            self.patience -= 1
        stop = self.patience == 0
        if stop and self.config["early_stop_verbose"]:
            print(f"Early stopped! Patience is {self.patience}.")
        return stop

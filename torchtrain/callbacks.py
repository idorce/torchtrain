class EarlyStop:
    def __init__(self, patience, mode, verbose=False):
        super().__init__()
        self.patience_orignal = patience
        self.mode = mode
        self.patience = patience
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.verbose = verbose

    def check(self, metric):
        signal = ""
        if self.mode == "min":
            best = metric < self.best_metric
        else:
            best = metric > self.best_metric
        if best:
            self.best_metric = metric
            self.patience = self.patience_orignal
            signal = "best"
            if self.verbose:
                print("Save best-so-far model state_dict...")
        else:
            self.patience -= 1
        stop = self.patience == 0
        if stop:
            signal = "stop"
            msg = (
                "Early stopped! Training loss has not decreased for"
                + str(self.patience)
                + "epoches."
            )
            if self.verbose:
                print(msg)
        return signal

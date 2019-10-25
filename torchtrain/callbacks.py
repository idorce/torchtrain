class EarlyStop:
    def __init__(self, patience, verbose=False):
        super().__init__()
        self.patience_orignal = patience
        self.patience = patience
        self.min_loss = float("inf")
        self.verbose = verbose

    def check(self, loss):
        signal = ""
        if loss < self.min_loss:
            self.min_loss = loss
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

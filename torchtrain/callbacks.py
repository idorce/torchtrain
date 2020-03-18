class EarlyStop:
    def __init__(self, patience=5, mode="min", verbose=False):
        self.patience = patience
        self.patience_now = patience
        self.mode = mode
        self.verbose = verbose
        self.best_score = float("inf") if self.mode == "min" else float("-inf")
        self.last_score = self.best_score
        self.best = True
        self.stop = False

    def is_better(self, now, before):
        return now < before if self.mode == "min" else now > before

    def check(self, score):
        self.best = self.is_better(score, self.best_score)
        if self.best:
            self.best_score = score
            if self.verbose:
                print("Save best-so-far model state_dict...")
        better = self.best or self.is_better(score, self.last_score)
        self.last_score = score
        self.patience_now = self.patience if better else (self.patience_now - 1)
        self.stop = self.patience_now == 0
        if self.stop and self.verbose:
            print(f"Early stop! Patience is {self.patience}.")

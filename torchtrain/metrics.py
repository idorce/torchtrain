class AverageAggregator:
    def __init__(self, criterion):
        self.criterion = criterion
        self.reset()

    def reset(self):
        self.value = 0
        self.sum = 0
        self.count = 0

    def update(self, outputs, labels):
        self.batch_score = self.criterion(outputs, labels)
        batch_value = (
            self.batch_score.item()
            if hasattr(self.batch_score, "item")
            else self.batch_score
        )
        batch_size = (
            self.criterion.get_batch_size(labels)
            if hasattr(self.criterion, "get_batch_size")
            else len(labels)
        )
        self.sum += batch_value * batch_size
        self.count += batch_size
        self.value = self.sum / self.count

    def get_value(self, reset=False):
        value = self.value
        if reset:
            self.reset()
        return value

    def get_batch_score(self):
        batch_score = self.batch_score
        del self.batch_score
        return batch_score

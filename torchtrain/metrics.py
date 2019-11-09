class AverageAggregator:
    def __init__(self, criterion):
        self.criterion = criterion
        self.reset()

    def reset(self):
        self.value = 0
        self.sum = 0
        self.count = 0
        self.batch_score = None

    def update(self, outputs, labels):
        self.batch_score = self.criterion(outputs, labels)
        batch_value = self.batch_score.item()
        batch_size = labels.size(0)
        self.sum += batch_value * batch_size
        self.count += batch_size
        self.value = self.sum / self.count

    def value(self, reset=False):
        value = self.value
        if reset:
            self.reset()
        return value

    def get_batch_score(self):
        return self.batch_score

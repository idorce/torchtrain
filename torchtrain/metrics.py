class AverageMeter:
    def __init__(self, criterion):
        self.criterion = criterion
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.batch_loss = None

    def update(self, outputs, labels):
        batch_size = labels.size(0)
        self.batch_loss = self.criterion(outputs, labels)
        batch_value = self.batch_loss.item()
        self.sum += batch_value * batch_size
        self.count += batch_size
        self.avg = self.sum / self.count

    def value(self, reset=False):
        value = self.avg
        if reset:
            self.reset()
        return value

    def batch_loss(self):
        return self.batch_loss

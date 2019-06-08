import tensorflow as tf


class Trainer:

    def __init__(self, model, optimizer, train_dataset, loss_func, epoch):
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.loss_func = loss_func
        self.epoch = epoch

    def run(self):
        for epoch in range(self.epoch):
            epoch_loss = self._train_epoch()
            print(f'Epoch {epoch + 1} Loss {epoch_loss:.4f}')

    def _train_epoch(self):
        total_loss = 0.
        for iteration, (src, tgt) in enumerate(self.train_dataset):
            outputs = self.model(src, tgt)
            from IPython import embed; embed()
            total_loss += batch_loss
        return total_loss

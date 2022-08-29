import torch
from torch.optim.lr_scheduler import _LRScheduler


class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r
                for base_lr in self.base_lrs]


class IteratorWrapper:
    def __init__(self, iterator):
        self.iterator = iterator
        self._iterator = iter(iterator)

    def __next__(self):
        try:
            inputs, labels = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterator)
            inputs, labels, *_ = next(self._iterator)

        return inputs, labels

    def get_batch(self):
        return next(self)


class LRFinder:
    def __init__(self, model, optimizer, criterion, device, save_path, model_name):
        """
        Learning rate finder tool
        :param model: Optimized model
        :type model: torch.nn.Module
        :param optimizer: Optimizer object
        :type optimizer: torch.optim.Optim
        :param criterion: Loss criterion object
        :type criterion: torch.nn.Loss
        :param device: Utilized device (cuda or cpu)
        :type device: torch.device.Device
        :param save_path: Base path for model saving
        :type save_path: str
        :param model_name: Name of the model
        :type model_name: str
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.save_path = save_path
        self.model_name = model_name

        self.save()

    def __get_path(self, name):
        return f"{self.save_path}/{self.model_name}/{name}.pt"

    def range_test(self, iterator, end_lr=10, num_iter=100, smooth_f=0.05, diverge_th=5):

        lrs = []
        losses = []
        best_loss = float('inf')

        lr_scheduler = ExponentialLR(self.optimizer, end_lr, num_iter)
        iterator = IteratorWrapper(iterator)

        for iteration in range(num_iter):
            loss = self._train_batch(iterator)
            lrs.append(lr_scheduler.get_last_lr()[0])

            # update lr
            lr_scheduler.step()

            if iteration > 0:
                loss = smooth_f * loss + (1 - smooth_f) * losses[-1]

            if loss < best_loss:
                best_loss = loss

            losses.append(loss)
            if loss > diverge_th * best_loss:
                print("LR finder early stopping. The loss has diverged.")
                break

        # reset model to initial parameters
        self.load()
        return lrs, losses

    def _train_batch(self, iterator):
        self.model.train()

        self.optimizer.zero_grad()
        x, y = iterator.get_batch()

        x = x.to(self.device)
        y = y.to(self.device)

        y_pred, _ = self.model(x)

        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self):
        torch.save(self.model.state_dict(), self.__get_path("init_params"))

    def load(self):
        self.model.load_state_dict(torch.load(self.__get_path("init_params")))

import torch
import torch.nn.functional as F
import torchmetrics.functional as fn
import time

from tqdm.auto import tqdm
from tqdm.notebook import tqdm as tqdm_notebook


class ModelProcessor:
    def __init__(self, model, criterion, optimizer, device, epochs, **config):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs

        self.is_launched_in_notebook = config.get("is_launched_in_notebook", False)
        self.tqdm_function = tqdm if not self.is_launched_in_notebook else tqdm_notebook

    @staticmethod
    def __epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def train(self, data_loader):
        epoch_loss = 0
        epoch_acc = 0

        self.model.train()
        for (x, y) in self.tqdm_function(data_loader, desc="Training", leave=False):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            y_pred, _ = self.model(x)

            loss = self.criterion(y_pred, y)
            acc = fn.accuracy(y_pred, y)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(data_loader), epoch_acc / len(data_loader)

    def evaluate(self, data_loader):
        epoch_loss = 0
        epoch_acc = 0

        self.model.eval()
        with torch.no_grad():
            for (x, y) in self.tqdm_function(data_loader, desc="Evaluating", leave=False):
                x = x.to(self.device)
                y = y.to(self.device)

                y_pred, _ = self.model(x)

                loss = self.criterion(y_pred, y)
                acc = fn.accuracy(y_pred, y)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(data_loader), epoch_acc / len(data_loader)

    def process(self, train_loader, valid_loader, test_loader):
        best_valid_loss = float('inf')
        iterator = range(self.epochs) if not self.is_launched_in_notebook else self.tqdm_function(range(self.epochs))
        for epoch in iterator:

            start_time = time.monotonic()

            train_loss, train_acc = self.train(train_loader)
            valid_loss, valid_acc = self.evaluate(valid_loader)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.model.save()

            end_time = time.monotonic()

            epoch_mins, epoch_secs = self.__epoch_time(start_time, end_time)

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

        self.model.load()
        test_loss, test_acc = self.evaluate(test_loader)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

    def get_predictions(self, data_loader):
        self.model.eval()

        images = []
        labels = []
        probs = []

        with torch.no_grad():
            for (x, y) in data_loader:
                x = x.to(self.device)

                y_pred, _ = self.model(x)
                y_prob = F.softmax(y_pred, dim=-1)

                images.append(x.cpu())
                labels.append(y.cpu())
                probs.append(y_prob.cpu())

        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)
        probs = torch.cat(probs, dim=0)

        return images, labels, probs

    def get_representations(self, data_loader):
        self.model.eval()

        outputs = []
        intermediates = []
        labels = []

        with torch.no_grad():
            for (x, y) in self.tqdm_function(data_loader, desc="Getting representations", leave=False):
                x = x.to(self.device)

                y_pred, h = self.model(x)

                outputs.append(y_pred.cpu())
                intermediates.append(h.cpu())
                labels.append(y)

        outputs = torch.cat(outputs, dim=0)
        intermediates = torch.cat(intermediates, dim=0)
        labels = torch.cat(labels, dim=0)

        return outputs, intermediates, labels

    def imagine_digit(self, digit, shape, n_iterations=50_000):
        self.model.eval()

        best_prob = 0
        best_image = None

        with torch.no_grad():
            for _ in self.tqdm_function(range(n_iterations), desc="Digit imagining", leave=False):
                x = torch.randn(shape).to(self.device)

                y_pred, _ = self.model(x)

                preds = F.softmax(y_pred, dim=-1)
                _best_prob, index = torch.max(preds[:, digit], dim=0)

                if _best_prob > best_prob:
                    best_prob = _best_prob
                    best_image = x[index]

        return best_image.view(shape[-2:]), best_prob

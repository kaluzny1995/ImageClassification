import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics

import utils.calculation
from utils.util import create_dir_if_not_exists


class Visualization:
    def __init__(self, save_path, model_name, is_saved=False, is_shown=False):
        """
        Image and plots visualizer
        :param save_path: Base path for images saving
        :type save_path: str
        :param model_name: Name of the model
        :type model_name: str
        :param is_saved: Are visualizations saved
        :type is_saved: bool
        :param is_shown: Are visualizations shown
        :type is_shown: bool
        """
        self.save_path = save_path
        self.model_name = model_name
        self.is_saved = is_saved
        self.is_shown = is_shown
        if self.is_saved:
            create_dir_if_not_exists(f"{self.save_path}/{self.model_name}")

    def __get_image_path(self, name):
        return f"{self.save_path}/{self.model_name}/{name}.png"

    def plot_image(self, image, **config):
        """
        Single image visualization
        :param image: Image
        :type image: tensor[float] | np.array[float]
        """
        mpl.rcParams.update(mpl.rcParamsDefault)
        title = config.get("title", "Image")
        name = config.get("name", "image")

        fig = plt.figure()
        fig.suptitle(title)
        ax = fig.add_subplot(111)
        ax.imshow(image.permute(1, 2, 0).cpu().numpy(), cmap='bone')
        ax.axis('off')

        if self.is_saved:
            plt.savefig(self.__get_image_path(name))
        if self.is_shown:
            plt.show()

    def plot_images(self, images, labels=None, **config):
        """
        Sample of images visualization
        :param images: Sample of images
        :type images: List[tensor[float]]
        :param labels: Image labels (image class names)
        :type labels: List[str] | type(None)
        """
        mpl.rcParams.update(mpl.rcParamsDefault)
        title = config.get("title", "Sample of images")
        name = config.get("name", "sample_images")
        figsize = config.get("figsize", (10, 10))

        n_images = len(images)
        rows = int(np.sqrt(n_images))
        cols = int(np.sqrt(n_images))

        fig = plt.figure(figsize=figsize)
        fig.suptitle(title)
        for i in range(rows * cols):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.imshow(images[i].permute(1, 2, 0).cpu().numpy(), cmap="bone")
            if labels is not None:
                ax.set_title(labels[i])
            ax.axis("off")

        if self.is_saved:
            plt.savefig(self.__get_image_path(name))
        if self.is_shown:
            plt.show()

    def plot_filter(self, images, filtered_images, **config):
        """
        Images with their respective convolutionally filtered images visualization
        :param images: Sample of images
        :type images: List[tensor[float]]
        :param filtered_images: Respective convolutionally filtered images
        :type filtered_images: List[tensor[float]]
        """
        mpl.rcParams.update(mpl.rcParamsDefault)
        title = config.get("title", "Convolutionally filtered images")
        name = config.get("name", "filtered_images")
        figsize = config.get("figsize", (20, 5))

        n_images = images.shape[0]

        fig = plt.figure(figsize=figsize)
        fig.suptitle(title)
        for i in range(n_images):
            ax = fig.add_subplot(2, n_images, i + 1)
            ax.imshow(images[i].permute(1, 2, 0), cmap='bone')
            ax.set_title('Original')
            ax.axis('off')

            ax = fig.add_subplot(2, n_images, n_images + i + 1)
            ax.imshow(filtered_images[i].permute(1, 2, 0), cmap='bone')
            ax.set_title('Filtered')
            ax.axis('off')

        if self.is_saved:
            plt.savefig(self.__get_image_path(name))
        if self.is_shown:
            plt.show()

    def plot_pool(self, images, pooled_images, **config):
        """
        Images with their respective pooled images visualization
        :param images: Sample of images
        :type images: List[tensor[float]]
        :param pooled_images: Respective pooled images
        :type pooled_images: List[tensor[float]]
        """
        mpl.rcParams.update(mpl.rcParamsDefault)
        title = config.get("title", "Pooled images")
        name = config.get("name", "pooled_images")
        figsize = config.get("figsize", (20, 5))

        n_images = images.shape[0]

        fig = plt.figure(figsize=figsize)
        fig.suptitle(title)
        for i in range(n_images):
            ax = fig.add_subplot(2, n_images, i + 1)
            ax.imshow(images[i].permute(1, 2, 0), cmap='bone')
            ax.set_title('Original')
            ax.axis('off')

            ax = fig.add_subplot(2, n_images, n_images + i + 1)
            ax.imshow(pooled_images[i].permute(1, 2, 0), cmap='bone')
            ax.set_title('Pooled')
            ax.axis('off')

        if self.is_saved:
            plt.savefig(self.__get_image_path(name))
        if self.is_shown:
            plt.show()

    def plot_lr_finder(self, lrs, losses, skip_start=5, skip_end=5, **config):
        """
        Optimal learning rate finding visualization
        :param lrs: List of examined learning rates
        :type lrs: List[float]
        :param losses: List of observed losses
        :type losses: List[float]
        :param skip_start: Offset from start
        :type skip_start: int
        :param skip_end: Offset from end
        :type skip_end: int
        """
        mpl.rcParams.update(mpl.rcParamsDefault)
        title = config.get("title", "Optimal learning rate finding")
        name = config.get("name", "optimal_lr_finding")
        figsize = config.get("figsize", (16, 8))

        if skip_end == 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        else:
            lrs = lrs[skip_start:-skip_end]
            losses = losses[skip_start:-skip_end]

        fig = plt.figure(figsize=figsize)
        fig.suptitle(title)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(lrs, losses)
        ax.set_xscale('log')
        ax.set_xlabel('Learning rate')
        ax.set_ylabel('Loss')
        ax.grid(True, 'both', 'x')

        if self.is_saved:
            plt.savefig(self.__get_image_path(name))
        if self.is_shown:
            plt.show()

    def plot_confusion_matrix(self, labels, pred_labels, **config):
        """
        Confusion matrix visualization
        :param labels: Ground truth labels
        :type labels: tensor[float]
        :param pred_labels: Predicted labels
        :type pred_labels: tensor[float]
        """
        mpl.rcParams.update(mpl.rcParamsDefault)
        title = config.get("title", "Confusion matrix")
        name = config.get("name", "confusion_matrix")
        figsize = config.get("figsize", (10, 10))
        labels_range = config.get("labels_range", 10)

        fig = plt.figure(figsize=figsize)
        fig.suptitle(title)
        ax = fig.add_subplot(1, 1, 1)
        cm = metrics.confusion_matrix(labels, pred_labels)
        cm = metrics.ConfusionMatrixDisplay(cm, display_labels=range(labels_range))
        cm.plot(values_format='d', cmap='Blues', ax=ax)

        if self.is_saved:
            plt.savefig(self.__get_image_path(name))
        if self.is_shown:
            plt.show()

    def plot_most_incorrect(self, incorrect, n_images, **config):
        """
        Most incorrectly classified examples visualization
        :param incorrect: Incorrectly classified images with their labels and probabilities
        :type incorrect: List[tensor[float]]
        :param n_images: Number of images
        :type n_images: int
        """
        mpl.rcParams.update(mpl.rcParamsDefault)
        title = config.get("title", "Most incorrect classifications")
        name = config.get("name", "most_incorrect")
        class_names = config.get("class_names", None)
        figsize = config.get("figsize", (25, 20))

        rows = int(np.sqrt(n_images))
        cols = int(np.sqrt(n_images))

        fig = plt.figure(figsize=figsize)
        fig.suptitle(title)
        for i in range(rows * cols):
            ax = fig.add_subplot(rows, cols, i + 1)
            image, true_label, probs = incorrect[i]
            true_prob = probs[true_label]
            incorrect_prob, incorrect_label = probs.max(dim=0)
            if class_names is not None:
                true_label = class_names[true_label]
                incorrect_label = class_names[incorrect_label]
            ax.imshow(image.permute(1, 2, 0).cpu().numpy(), cmap='bone')
            ax.set_title(f'true label: {true_label} ({true_prob:.3f})\n'
                         f'pred label: {incorrect_label} ({incorrect_prob:.3f})')
            ax.axis('off')
        fig.subplots_adjust(hspace=0.5)

        if self.is_saved:
            plt.savefig(self.__get_image_path(name))
        if self.is_shown:
            plt.show()

    def plot_representations(self, data, labels, **config):
        """
        Data points representation diagram visualization
        :param data: Data points
        :type data: List[tensor[float]] | List[float]
        :param labels: Data labels
        :type labels: List[str]
        """
        mpl.rcParams.update(mpl.rcParamsDefault)
        title = config.get("title", "Dataset representation")
        name = config.get("name", "dataset_representation")
        n_examples = config.get("n_examples", None)
        figsize = config.get("figsize", (10, 10))

        if n_examples is not None:
            data = data[:n_examples]
            labels = labels[:n_examples]
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title)
        ax = fig.add_subplot(111)
        scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10')
        handles, labels = scatter.legend_elements()
        ax.legend(handles=handles, labels=labels)

        if self.is_saved:
            plt.savefig(self.__get_image_path(name))
        if self.is_shown:
            plt.show()

    def plot_weights(self, weights, n_weights, **config):
        """
        Trained nn layer weights representation
        :param weights: Weights representation images
        :type weights: List[tensor[float]]
        :param n_weights: Number of weights representations
        :type n_weights: int
        """
        mpl.rcParams.update(mpl.rcParamsDefault)
        title = config.get("title", "Layer weights")
        name = config.get("name", "layer_weights")
        dims = config.get("dims", (10, 10))
        figsize = config.get("figsize", (20, 10))

        rows = int(np.sqrt(n_weights))
        cols = int(np.sqrt(n_weights))

        fig = plt.figure(figsize=figsize)
        fig.suptitle(title)
        for i in range(rows * cols):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.imshow(weights[i].view(dims).cpu().numpy(), cmap='bone')
            ax.axis('off')

        if self.is_saved:
            plt.savefig(self.__get_image_path(name))
        if self.is_shown:
            plt.show()

    def plot_many_filtered_images(self, images, filtered_images, **config):
        """
        Images with their respective lists of convolutionally filtered images visualization
        :param images: Sample of images
        :type images: List[tensor[float]]
        :param filtered_images: Respective lists of convolutionally filtered images
        :type filtered_images: List[tensor[tensor[float]]]
        """
        mpl.rcParams.update(mpl.rcParamsDefault)
        title = config.get("title", "Convolutionally filtered images by neural network")
        name = config.get("name", "many_filtered_images")
        figsize = config.get("figsize", (20, 10))

        n_images = images.shape[0]
        n_filters = filtered_images.shape[1]

        fig = plt.figure(figsize=figsize)
        fig.suptitle(title)
        for i in range(n_images):

            ax = fig.add_subplot(n_images, n_filters + 1, i + 1 + (i * n_filters))
            ax.imshow(images[i].permute(1, 2, 0), cmap='bone')
            ax.set_title('Original')
            ax.axis('off')

            for j in range(n_filters):
                image = filtered_images[i][j]
                ax = fig.add_subplot(n_images, n_filters + 1, i + 1 + (i * n_filters) + j + 1)
                ax.imshow(image.numpy(), cmap='bone')
                ax.set_title(f'Filtered {j + 1}')
                ax.axis('off')

        if self.is_saved:
            plt.savefig(self.__get_image_path(name))
        if self.is_shown:
            plt.show()

    def plot_filters(self, conv_filters, **config):
        """
        Trained by neural network filters visualization
        :param conv_filters: Trained convolutional filters
        :type conv_filters: List[tensor[float]]
        """
        mpl.rcParams.update(mpl.rcParamsDefault)
        title = config.get("title", "Convolutional filters trained by neural network")
        name = config.get("name", "trained_filters")
        figsize = config.get("figsize", (20, 10))

        conv_filters = conv_filters.cpu()
        n_filters = conv_filters.shape[0]

        fig = plt.figure(figsize=figsize)
        fig.suptitle(title)
        for i in range(n_filters):
            ax = fig.add_subplot(1, n_filters, i + 1)
            ax.imshow(conv_filters[i].permute(1, 2, 0), cmap='bone')
            ax.axis('off')

        if self.is_saved:
            plt.savefig(self.__get_image_path(name))
        if self.is_shown:
            plt.show()

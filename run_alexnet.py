import torch
import torch.utils.data as data
import torchvision

import numpy as np
import random

from config.main_config import MainConfig
from config.alexnet_config import AlexNetConfig
from utils.visualizations import Visualization
import utils.calculation
from models.alexnet import AlexNet
from models.model_processor import ModelProcessor
from utils.lr_finder import LRFinder

main_config = MainConfig.from_json()
print(f"Main config: {main_config.to_dict()}")

random.seed(main_config.random_seed)
np.random.seed(main_config.random_seed)
torch.manual_seed(main_config.random_seed)
torch.cuda.manual_seed(main_config.random_seed)
torch.backends.cudnn.deterministic = True


alexnet_nn_config = AlexNetConfig.from_json()
print(f"AlexNet NN config: {alexnet_nn_config.to_dict()}")


# Datasets
cifar10_dataset = alexnet_nn_config.utilized_dataset.value(main_config.path_data, main_config.tv_split_ratio)
train_data, valid_data, test_data = cifar10_dataset.get_datasets()
print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')


# Visualizer
visualization = Visualization(main_config.path_storage_visualization,
                              alexnet_nn_config.name,
                              is_saved=main_config.is_visualization_saved,
                              is_shown=main_config.is_visualization_shown)


# Image examples
N_EXAMPLES = 25
images = list(map(lambda i: train_data[i][0], range(N_EXAMPLES)))
labels = list(map(lambda i: test_data.classes[train_data[i][1]], range(N_EXAMPLES)))
visualization.plot_images(images, labels, title="Training images sample", name="train_images")
visualization.plot_images(utils.calculation.normalize_images(images), labels,
                          title="Training images sample - normalized", name="train_images_normalized")


# Convolutional filtering and pooling examples
N_EXAMPLES = 5
images = list(map(lambda i: train_data[i][0], range(N_EXAMPLES)))

conv_filter = [[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]]
f_images, f_filters = utils.calculation.get_filtered_images(images, conv_filter)
visualization.plot_filter(f_images, f_filters,
                          title="Convolutionally filtered images - horizontal upper", name="filtered_images_hor_up")
visualization.plot_filter(utils.calculation.normalize_images(f_images), utils.calculation.normalize_images(f_filters),
                          title="Convolutionally filtered images - horizontal upper - normalized",
                          name="filtered_images_hor_up_normalized")


conv_filter = [[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]
f_images, f_filters = utils.calculation.get_filtered_images(images, conv_filter)
visualization.plot_filter(f_images, f_filters,
                          title="Convolutionally filtered images - vertical left", name="filtered_images_ver_left")
visualization.plot_filter(utils.calculation.normalize_images(f_images), utils.calculation.normalize_images(f_filters),
                          title="Convolutionally filtered images - vertical left - normalized",
                          name="filtered_images_ver_left_normalized")

p_images, p_pooled = utils.calculation.get_pooled_images(images, "max", 2)
visualization.plot_pool(p_images, p_pooled,
                        title="Pooled images - max 2", name="pooled_images_max_2")
visualization.plot_pool(utils.calculation.normalize_images(p_images), utils.calculation.normalize_images(p_pooled),
                        title="Pooled images - max 2 - normalized", name="pooled_images_max_2_normalized")
p_images, p_pooled = utils.calculation.get_pooled_images(images, "mean", 2)
visualization.plot_pool(p_images, p_pooled,
                        title="Pooled images - mean 2", name="pooled_images_mean_2")
visualization.plot_pool(utils.calculation.normalize_images(p_images), utils.calculation.normalize_images(p_pooled),
                        title="Pooled images - mean 2 - normalized", name="pooled_images_mean_2_normalized")


# Data loaders
train_loader = data.DataLoader(train_data, shuffle=True, batch_size=alexnet_nn_config.hparam_batch_size)
valid_loader = data.DataLoader(valid_data, batch_size=alexnet_nn_config.hparam_batch_size)
test_loader = data.DataLoader(test_data, batch_size=alexnet_nn_config.hparam_batch_size)


# Model definition
model = AlexNet(alexnet_nn_config.param_ft_in_out_channels,
                alexnet_nn_config.param_ft_kernel_size,
                alexnet_nn_config.param_ft_pool_kernel_size,
                alexnet_nn_config.param_ft_stride,
                alexnet_nn_config.param_ft_padding,
                alexnet_nn_config.param_clf_dims,
                alexnet_nn_config.param_clf_dropout,
                main_config.path_storage_models,
                alexnet_nn_config.name)
print(f"The model has {model.count_params()} trainable parameters.")


# Optimal learning rate finding
optimizer = alexnet_nn_config.lrf_optimizer.value(model.parameters(), lr=alexnet_nn_config.lrf_start_lr)
device = torch.device(main_config.cuda_device if torch.cuda.is_available() else main_config.non_cuda_device)
criterion = alexnet_nn_config.lrf_criterion.value()

model = model.to(device)
criterion = criterion.to(device)

lr_finder = LRFinder(model, optimizer, criterion, device, main_config.path_storage_models, alexnet_nn_config.name)
lrs, losses = lr_finder.range_test(train_loader,
                                   end_lr=alexnet_nn_config.lrf_end_lr,
                                   num_iter=alexnet_nn_config.lrf_num_iter)

visualization.plot_lr_finder(lrs, losses)


# Model hyperparams
optimizer = alexnet_nn_config.hparam_optimizer.value(model.parameters(), lr=alexnet_nn_config.hparam_learning_rate)
criterion = alexnet_nn_config.hparam_criterion.value()
device = torch.device(main_config.cuda_device if torch.cuda.is_available() else main_config.non_cuda_device)
model = model.to(device)
criterion = criterion.to(device)

# Model processor
model_processor = ModelProcessor(model, criterion, optimizer, device, alexnet_nn_config.hparam_epochs,
                                 is_launched_in_notebook=main_config.is_launched_in_notebook)
# Training
model_processor.process(train_loader, valid_loader, test_loader)
# Prediction
images, labels, probs = model_processor.get_predictions(test_loader)
pred_labels = utils.calculation.get_predicted_labels(probs)
visualization.plot_confusion_matrix(labels, pred_labels,
                                    title="Confusion matrix", name="confusion_matrix")


# Most incorrect classifications
N_EXAMPLES = 25
incorrect_examples = utils.calculation.get_most_incorrect_examples(utils.calculation.normalize_images(images),
                                                                   labels, probs)
visualization.plot_most_incorrect(incorrect_examples, N_EXAMPLES, class_names=test_data.classes,
                                  title="Most incorrect classifications", name="most_incorrect")

# Layer representations
outputs, intermediates, labels = model_processor.get_representations(train_loader)

output_pca_data = utils.calculation.get_pca(outputs)
visualization.plot_representations(output_pca_data, labels,
                                   title="PCA output layer representation", name="pca_output")

intermediate_pca_data = utils.calculation.get_pca(intermediates)
visualization.plot_representations(intermediate_pca_data, labels,
                                   title="PCA convolutional layer representation", name="pca_hidden")

N_EXAMPLES = 5_000
output_tsne_data = utils.calculation.get_tsne(outputs, n_examples=N_EXAMPLES)
visualization.plot_representations(output_tsne_data, labels, n_examples=N_EXAMPLES,
                                   title="TSNE output layer representation", name="tsne_output")

intermediate_tsne_data = utils.calculation.get_tsne(intermediates, n_examples=N_EXAMPLES)
visualization.plot_representations(intermediate_tsne_data, labels, n_examples=N_EXAMPLES,
                                   title="TSNE convolutional layer representation", name="tsne_hidden")


# Image imagination
IMAGE_LABEL = "frog"
best_image, best_prob = model_processor.imagine_image(test_data.classes.index(IMAGE_LABEL),
                                                      shape=[256, 3, 32, 32], n_iterations=10_000)
print(f"Best image probability: {best_prob.item()*100:.2f}%")
best_image = utils.calculation.normalize_images(best_image.unsqueeze(0)).squeeze(0)
visualization.plot_image(best_image,
                         title=f"Best imagined image for label {IMAGE_LABEL}", name=f"best_imagined_{IMAGE_LABEL}")


# Image trained conv_filters
N_EXAMPLES = 5
N_FILTERS = 7
images = utils.calculation.normalize_images(list(map(lambda i: train_data[i][0], range(N_EXAMPLES))))
filters = model.features[0].weight.data[:N_FILTERS]

visualization.plot_many_filtered_images(*utils.calculation.get_many_filtered_images(images, filters),
                                        title="Convolutionally filtered images by neural network", name="filtered_images_by_nn")
visualization.plot_many_filtered_images(*utils.calculation.get_many_filtered_images(best_image.unsqueeze(0), filters),
                                        title="Convolutionally filtered best image by neural network", name="filtered_best_image_by_nn")
visualization.plot_filters(filters, title="Trained convolutional filters", name="trained_filters")


# Pretrained AlexNet model
filters = torchvision.models.alexnet(pretrained=True).features[0].weight.data[:N_FILTERS]
visualization.plot_filters(filters, title="Pretrained model convolutional filters", name="pretrained_filters")

import torch
import torch.utils.data as data

import numpy as np
import random

from config.main_config import MainConfig
from config.lenet_config import LeNetConfig
from utils.visualizations import Visualization
import utils.calculation
from models.lenet import LeNet
from models.model_processor import ModelProcessor

main_config = MainConfig.from_json()
print(f"Main config: {main_config.to_dict()}")

random.seed(main_config.random_seed)
np.random.seed(main_config.random_seed)
torch.manual_seed(main_config.random_seed)
torch.cuda.manual_seed(main_config.random_seed)
torch.backends.cudnn.deterministic = True


lenet_config = LeNetConfig.from_json()
print(f"LeNet NN config: {lenet_config.to_dict()}")


# Datasets
mnist_dataset = lenet_config.utilized_dataset.value(main_config.path_data, main_config.tv_split_ratio)
train_data, valid_data, test_data = mnist_dataset.get_datasets()
print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')


# Visualizer
visualization = Visualization(main_config.path_storage_visualization,
                              lenet_config.name,
                              is_saved=main_config.is_visualization_saved,
                              is_shown=main_config.is_visualization_shown)


# Convolutional filtering and pooling examples
N_EXAMPLES = 5
images = list(map(lambda i: train_data[i][0], range(N_EXAMPLES)))

conv_filter = [[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]]
visualization.plot_filter(*utils.calculation.get_filtered_images(images, conv_filter),
                          title="Convolutionally filtered images - horizontal upper", name="filtered_images_hor_up")

conv_filter = [[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]]
visualization.plot_filter(*utils.calculation.get_filtered_images(images, conv_filter),
                          title="Convolutionally filtered images - horizontal downer", name="filtered_images_hor_down")

conv_filter = [[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]
visualization.plot_filter(*utils.calculation.get_filtered_images(images, conv_filter),
                          title="Convolutionally filtered images - vertical left", name="filtered_images_ver_left")

conv_filter = [[1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]]
visualization.plot_filter(*utils.calculation.get_filtered_images(images, conv_filter),
                          title="Convolutionally filtered images - vertical right", name="filtered_images_ver_right")

conv_filter = [[-2, -1, 0],
               [-1, 0, 1],
               [0, 1, 2]]
visualization.plot_filter(*utils.calculation.get_filtered_images(images, conv_filter),
                          title="Convolutionally filtered images - diagonal upper left", name="filtered_images_diag_up_left")

conv_filter = [[2, 1, 0],
               [1, 0, -1],
               [0, -1, -2]]
visualization.plot_filter(*utils.calculation.get_filtered_images(images, conv_filter),
                          title="Convolutionally filtered images - diagonal downer right", name="filtered_images_diag_down_right")

visualization.plot_pool(*utils.calculation.get_pooled_images(images, "max", 2),
                        title="Pooled images - max 2", name="pooled_images_max_2")
visualization.plot_pool(*utils.calculation.get_pooled_images(images, "max", 3),
                        title="Pooled images - max 3", name="pooled_images_max_3")
visualization.plot_pool(*utils.calculation.get_pooled_images(images, "mean", 2),
                        title="Pooled images - mean 2", name="pooled_images_mean_2")
visualization.plot_pool(*utils.calculation.get_pooled_images(images, "mean", 3),
                        title="Pooled images - mean 3", name="pooled_images_mean_3")


# Data loaders
train_loader = data.DataLoader(train_data, shuffle=True, batch_size=lenet_config.hparam_batch_size)
valid_loader = data.DataLoader(valid_data, batch_size=lenet_config.hparam_batch_size)
test_loader = data.DataLoader(test_data, batch_size=lenet_config.hparam_batch_size)


# Model definition
model = LeNet(lenet_config.param_in_channels,
              lenet_config.param_mid_channels,
              lenet_config.param_out_channels,
              lenet_config.param_kernel_size,
              lenet_config.param_pool_kernel_size,
              lenet_config.param_input_dim,
              lenet_config.param_hidden_dims,
              lenet_config.param_output_dim,
              main_config.path_storage_models,
              lenet_config.name)
print(f"The model has {model.count_params()} trainable parameters.")

# Model hyperparams
optimizer = lenet_config.hparam_optimizer.value(model.parameters(), lr=lenet_config.hparam_learning_rate)
criterion = lenet_config.hparam_criterion.value()
device = torch.device(main_config.cuda_device if torch.cuda.is_available() else main_config.non_cuda_device)
model = model.to(device)
criterion = criterion.to(device)

# Model processor
model_processor = ModelProcessor(model, criterion, optimizer, device, lenet_config.hparam_epochs,
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
incorrect_examples = utils.calculation.get_most_incorrect_examples(images, labels, probs)
visualization.plot_most_incorrect(incorrect_examples, N_EXAMPLES,
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
IMAGE_LABEL = 3
best_image, best_prob = model_processor.imagine_image(IMAGE_LABEL, shape=[32, 1, 28, 28])
print(f"Best image probability: {best_prob.item()*100:.2f}%")
visualization.plot_image(best_image,
                         title=f"Best imagined image of digit {IMAGE_LABEL}", name=f"best_imagined_{IMAGE_LABEL}")


# Image trained conv_filters
N_EXAMPLES = 5
images = list(map(lambda i: train_data[i][0], range(N_EXAMPLES)))
filters = model.conv1.weight.data

visualization.plot_many_filtered_images(*utils.calculation.get_many_filtered_images(images, filters),
                                        title="Convolutionally filtered images by neural network", name="filtered_images_by_nn")
visualization.plot_many_filtered_images(*utils.calculation.get_many_filtered_images(best_image.unsqueeze(0), filters),
                                        title="Convolutionally filtered best image by neural network", name="filtered_best_image_by_nn")
visualization.plot_filters(filters, title="Trained convolutional filters", name="trained_filters")

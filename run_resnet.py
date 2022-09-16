import torch
import torch.utils.data as data

import numpy as np
import random
import copy

from config.main_config import MainConfig
from config.resnet_config import ResNetConfig
import factories.enums
from utils.visualizations import Visualization
import utils.calculation
from models.resnet import ResNet
from models.model_processor import ModelProcessor
from utils.lr_finder import LRFinder

main_config = MainConfig.from_json()
print(f"Main config: {main_config.to_dict()}")

random.seed(main_config.random_seed)
np.random.seed(main_config.random_seed)
torch.manual_seed(main_config.random_seed)
torch.cuda.manual_seed(main_config.random_seed)
torch.backends.cudnn.deterministic = True


resnet_nn_config = ResNetConfig.from_json()
print(f"ResNet NN config: {resnet_nn_config.to_dict()}")


# Datasets
cub200_dataset = factories.enums.get_dataset(resnet_nn_config.utilized_dataset)(
    main_config.paths.data, main_config.tt_split_ratio, main_config.tv_split_ratio,
    are_parameters_calculated=False,
    ds_param_random_rotation=resnet_nn_config.ds_param.random_rotation,
    ds_param_random_horizontal_flip=resnet_nn_config.ds_param.random_horizontal_flip,
    ds_param_crop_size=resnet_nn_config.ds_param.crop_size,
    ds_param_crop_padding=resnet_nn_config.ds_param.crop_padding,
    ds_param_means=resnet_nn_config.ds_param.means,
    ds_param_stds=resnet_nn_config.ds_param.stds
)
train_data, valid_data, test_data = cub200_dataset.get_datasets()
print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')


# Visualizer
visualization = Visualization(main_config.paths.visualizations,
                              resnet_nn_config.name,
                              is_saved=main_config.is_visualization_saved,
                              is_shown=main_config.is_visualization_shown)


# Image examples
N_EXAMPLES = 25
images = list(map(lambda i: train_data[i][0], range(N_EXAMPLES)))
labels = list(map(lambda i: test_data.classes[train_data[i][1]], range(N_EXAMPLES)))
visualization.plot_images(images, labels, title="Training images sample", name="train_images", figsize=(15, 15))
visualization.plot_images(utils.calculation.normalize_images(images), labels,
                          title="Training images sample - normalized", name="train_images_normalized", figsize=(15, 15))


# Data loaders
train_loader = data.DataLoader(train_data, shuffle=True, batch_size=resnet_nn_config.hparam.batch_size)
valid_loader = data.DataLoader(valid_data, batch_size=resnet_nn_config.hparam.batch_size)
test_loader = data.DataLoader(test_data, batch_size=resnet_nn_config.hparam.batch_size)


# Model definition
model = ResNet(resnet_nn_config.param.preset,
               resnet_nn_config.param.clf.dims[0][-1],
               main_config.paths.models,
               resnet_nn_config.name)
print(f"The model has {model.count_params()} trainable parameters.")


# Optimal learning rate finding
model_for_lrf = copy.deepcopy(model)

optimizer_for_lrf = factories.enums.get_optimizer(resnet_nn_config.lrf.optimizer)(model_for_lrf.parameters(),
                                                                                  lr=resnet_nn_config.lrf.start_lr)
criterion_for_lrf = factories.enums.get_criterion(resnet_nn_config.lrf.criterion)()
if resnet_nn_config.lrf.device is not None:
    device_for_lrf = resnet_nn_config.lrf.device
else:
    device_for_lrf = torch.device(main_config.cuda_device if torch.cuda.is_available() else main_config.non_cuda_device)
model_for_lrf = model_for_lrf.to(device_for_lrf)
criterion_for_lrf = criterion_for_lrf.to(device_for_lrf)

lr_finder = LRFinder(model_for_lrf, optimizer_for_lrf, criterion_for_lrf, device_for_lrf,
                     main_config.paths.models, resnet_nn_config.name)
lrs, losses = lr_finder.range_test(train_loader,
                                   end_lr=resnet_nn_config.lrf.end_lr,
                                   num_iter=resnet_nn_config.lrf.num_iter)

visualization.plot_lr_finder(lrs, losses, skip_start=30, skip_end=30)


# Model hyperparams
model_params = [
    {'params': model.conv1.parameters(), 'lr': resnet_nn_config.hparam.learning_rate / 10},
    {'params': model.bn1.parameters(), 'lr': resnet_nn_config.hparam.learning_rate / 10},
    {'params': model.layer1.parameters(), 'lr': resnet_nn_config.hparam.learning_rate / 8},
    {'params': model.layer2.parameters(), 'lr': resnet_nn_config.hparam.learning_rate / 6},
    {'params': model.layer3.parameters(), 'lr': resnet_nn_config.hparam.learning_rate / 4},
    {'params': model.layer4.parameters(), 'lr': resnet_nn_config.hparam.learning_rate / 2},
    {'params': model.fc.parameters()}
]
optimizer = factories.enums.get_optimizer(resnet_nn_config.hparam.optimizer)(model.parameters(),
                                                                             lr=resnet_nn_config.hparam.learning_rate)
scheduler = factories.enums.get_lr_scheduler(resnet_nn_config.hparam.lr_scheduler)(
    optimizer,
    max_lr=list(map(lambda p: p['lr'], optimizer.param_groups)),
    total_steps=resnet_nn_config.hparam.epochs * len(train_loader)
)
criterion = factories.enums.get_criterion(resnet_nn_config.hparam.criterion)()
if resnet_nn_config.hparam.device is not None:
    device = resnet_nn_config.hparam.device
else:
    device = torch.device(main_config.cuda_device if torch.cuda.is_available() else main_config.non_cuda_device)
model = model.to(device)
criterion = criterion.to(device)
model = model.to(device)
criterion = criterion.to(device)

# Model processor
model_processor = ModelProcessor(model, criterion, optimizer, device, resnet_nn_config.hparam.epochs,
                                 is_topk_calculated=True, top_k=5,
                                 is_launched_in_notebook=main_config.is_launched_in_notebook)
# Training
model_processor.process(train_loader, valid_loader, test_loader)
# Prediction
images, labels, probs = model_processor.get_predictions(test_loader)
pred_labels = utils.calculation.get_predicted_labels(probs)
visualization.plot_confusion_matrix(labels, pred_labels,
                                    title="Confusion matrix", name="confusion_matrix", figsize=(50, 50),
                                    labels_range=len(test_data.classes))


# Most incorrect classifications
N_EXAMPLES = 36
incorrect_examples = utils.calculation.get_most_incorrect_examples(utils.calculation.normalize_images(images),
                                                                   labels, probs)
visualization.plot_most_incorrect(incorrect_examples, N_EXAMPLES, class_names=test_data.classes,
                                  title="Most incorrect classifications", name="most_incorrect")

# Layer representations
outputs, intermediates, labels = model_processor.get_representations(train_loader)

output_pca_data = utils.calculation.get_pca(outputs)
visualization.plot_representations(output_pca_data, labels,
                                   title="PCA output layer representation", name="pca_output", figsize=(15, 15))

intermediate_pca_data = utils.calculation.get_pca(intermediates)
visualization.plot_representations(intermediate_pca_data, labels,
                                   title="PCA convolutional layer representation", name="pca_hidden", figsize=(15, 15))

N_EXAMPLES = 5_000
output_tsne_data = utils.calculation.get_tsne(outputs, n_examples=N_EXAMPLES)
visualization.plot_representations(output_tsne_data, labels, n_examples=N_EXAMPLES,
                                   title="TSNE output layer representation", name="tsne_output", figsize=(15, 15))

intermediate_tsne_data = utils.calculation.get_tsne(intermediates, n_examples=N_EXAMPLES)
visualization.plot_representations(intermediate_tsne_data, labels, n_examples=N_EXAMPLES,
                                   title="TSNE convolutional layer representation", name="tsne_hidden", figsize=(15, 15))


# Image imagination
IMAGE_LABEL = "Bobolink"
best_image, best_prob = model_processor.imagine_image(test_data.classes.index(IMAGE_LABEL),
                                                      shape=[256, 3, 32, 32], n_iterations=5_000)
print(f"Best image probability: {best_prob.item()*100:.2f}%")
best_image = utils.calculation.normalize_images(best_image.unsqueeze(0)).squeeze(0)
visualization.plot_image(best_image,
                         title=f"Best imagined image for label {IMAGE_LABEL}", name=f"best_imagined_{IMAGE_LABEL}")


# Image trained conv_filters
N_EXAMPLES = 5
N_FILTERS = 7
images = utils.calculation.normalize_images(list(map(lambda i: train_data[i][0], range(N_EXAMPLES))))
filters = model.conv1.weight.data[:N_FILTERS]

visualization.plot_many_filtered_images(*utils.calculation.get_many_filtered_images(images, filters),
                                        title="Convolutionally filtered images by neural network", name="filtered_images_by_nn",
                                        figsize=(30, 30))
visualization.plot_many_filtered_images(*utils.calculation.get_many_filtered_images(best_image.unsqueeze(0), filters),
                                        title="Convolutionally filtered best image by neural network", name="filtered_best_image_by_nn",
                                        figsize=(30, 30))
visualization.plot_filters(filters, title="Trained convolutional filters", name="trained_filters", figsize=(30, 15))

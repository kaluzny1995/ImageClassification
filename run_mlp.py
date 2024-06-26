import torch
import torch.utils.data as data

import numpy as np
import random

from config.main_config import MainConfig
from config.mlp_config import MLPConfig
import factories.enums
from utils.visualizations import Visualization
import utils.calculation
from models.mlp import MLP
from models.model_processor import ModelProcessor

main_config = MainConfig.from_json()
print(f"Main config: {main_config.to_dict()}")

random.seed(main_config.random_seed)
np.random.seed(main_config.random_seed)
torch.manual_seed(main_config.random_seed)
torch.cuda.manual_seed(main_config.random_seed)
torch.backends.cudnn.deterministic = True


mlp_config = MLPConfig.from_json()
print(f"MLP NN config: {mlp_config.to_dict()}")


# Datasets
mnist_dataset = factories.enums.get_dataset(mlp_config.utilized_dataset)(main_config.paths.data, main_config.tv_split_ratio)
train_data, valid_data, test_data = mnist_dataset.get_datasets()
print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

# Visualizer
visualization = Visualization(main_config.paths.visualizations,
                              mlp_config.name,
                              is_saved=main_config.is_visualization_saved,
                              is_shown=main_config.is_visualization_shown)

# Image examples
N_EXAMPLES = 25
images = list(map(lambda i: train_data[i][0], range(N_EXAMPLES)))
visualization.plot_images(images, title="Training images sample", name="train_images")

N_EXAMPLES = 25
images = list(map(lambda i: valid_data[i][0], range(N_EXAMPLES)))
visualization.plot_images(images, title="Validation images sample", name="valid_images")


# Data loaders
train_loader = data.DataLoader(train_data, shuffle=True, batch_size=mlp_config.hparam.batch_size)
valid_loader = data.DataLoader(valid_data, batch_size=mlp_config.hparam.batch_size)
test_loader = data.DataLoader(test_data, batch_size=mlp_config.hparam.batch_size)


# Model definition
model = MLP(mlp_config.param.clf.dims,
            main_config.paths.models,
            mlp_config.name)
print(f"The model has {model.count_params()} trainable parameters.")

# Model hyperparams
optimizer = factories.enums.get_optimizer(mlp_config.hparam.optimizer)(model.parameters(),
                                                                       lr=mlp_config.hparam.learning_rate)
criterion = factories.enums.get_criterion(mlp_config.hparam.criterion)()
if mlp_config.hparam.device is not None:
    device = mlp_config.hparam.device
else:
    device = torch.device(main_config.cuda_device if torch.cuda.is_available() else main_config.non_cuda_device)
model = model.to(device)
criterion = criterion.to(device)

# Model processor
model_processor = ModelProcessor(model, criterion, optimizer, device, mlp_config.hparam.epochs,
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
                                   title="PCA hidden layer representation", name="pca_hidden")

N_EXAMPLES = 5_000
output_tsne_data = utils.calculation.get_tsne(outputs, n_examples=N_EXAMPLES)
visualization.plot_representations(output_tsne_data, labels, n_examples=N_EXAMPLES,
                                   title="TSNE output layer representation", name="tsne_output")

intermediate_tsne_data = utils.calculation.get_tsne(intermediates, n_examples=N_EXAMPLES)
visualization.plot_representations(intermediate_tsne_data, labels, n_examples=N_EXAMPLES,
                                   title="TSNE hidden layer representation", name="tsne_hidden")


# Image imagination
IMAGE_LABEL = 3
best_image, best_prob = model_processor.imagine_image(IMAGE_LABEL, shape=[32, 28, 28])
print(f"Best image probability: {best_prob.item()*100:.2f}%")
visualization.plot_image(best_image.unsqueeze(0),
                         title=f"Best imagined image of digit {IMAGE_LABEL}", name=f"best_imagined_{IMAGE_LABEL}")


# Model weights
N_WEIGHTS = 25
weights = model.input_fc.weight.data
visualization.plot_weights(weights, N_WEIGHTS, dims=(28, 28), title="Input layer trained weights", name="input_weights")

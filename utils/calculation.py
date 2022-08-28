from sklearn import decomposition
from sklearn import manifold
import torch
import torch.nn.functional as F


def normalize_images(images):
    def normalize(image):
        image_min = image.min()
        image_max = image.max()
        image.clamp_(min=image_min, max=image_max)
        image.add_(-image_min).div_(image_max - image_min + 1e-5)
        return image

    return torch.stack(list(map(lambda i: normalize(i), images)))


def get_filtered_images(images, conv_filter):
    images = torch.cat([i.unsqueeze(0) for i in images], dim=0).cpu()
    conv_filter = torch.FloatTensor(conv_filter).unsqueeze(0).unsqueeze(0).cpu()
    # adjust filter to number of image channels (1 for gray, 3 for RGB, etc.)
    conv_filter = conv_filter.repeat(1, images.size()[-3], 1, 1)
    filtered_images = F.conv2d(images, conv_filter)
    return images, filtered_images


def get_pooled_images(images, pool_type, pool_size):
    images = torch.cat([i.unsqueeze(0) for i in images], dim=0).cpu()
    if pool_type.lower() == 'max':
        pool_function = F.max_pool2d
    elif pool_type.lower() in ['mean', 'avg']:
        pool_function = F.avg_pool2d
    else:
        raise ValueError(f'pool_type must be either max or mean, got: {pool_type}')
    pooled_images = pool_function(images, kernel_size=pool_size)
    return images, pooled_images


def get_many_filtered_images(images, conv_filters):
    images = torch.cat([i.unsqueeze(0) for i in images], dim=0).cpu()
    conv_filters = conv_filters.cpu()
    filtered_images = F.conv2d(images, conv_filters)
    return images, filtered_images


def get_predicted_labels(probs):
    return torch.argmax(probs, 1)


def get_most_incorrect_examples(images, labels, probs):
    pred_labels = get_predicted_labels(probs)
    corrects = torch.eq(labels, pred_labels)
    incorrect_examples = filter(lambda x: not x[-1], zip(images, labels, probs, corrects))
    incorrect_examples = list(map(lambda x: x[:-1], incorrect_examples))
    incorrect_examples.sort(reverse=True, key=lambda x: x[2].max(dim=0).values)
    return incorrect_examples


def get_pca(data, n_components=2):
    pca = decomposition.PCA()
    pca.n_components = n_components
    pca_data = pca.fit_transform(data)
    return pca_data


def get_tsne(data, n_components=2, n_examples=None):
    if n_examples is not None:
        data = data[:n_examples]
    tsne = manifold.TSNE(n_components=n_components, random_state=0)
    tsne_data = tsne.fit_transform(data)
    return tsne_data

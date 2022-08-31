from torchvision import models


pretrained_models_dict = dict(
    alexnet=models.alexnet,
    vgg11=models.vgg11,
    vgg11_bn=models.vgg11_bn,
    vgg13=models.vgg13,
    vgg13_bn=models.vgg13_bn
)

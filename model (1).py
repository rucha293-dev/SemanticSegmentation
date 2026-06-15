import torch.nn as nn
from torchvision import models
from dataset import NUM_CLASSES


def get_model(backbone='resnet101'):
    """
    Returns DeepLabV3 with ResNet50 or ResNet101 backbone.
    Replaces both classifier heads for NUM_CLASSES output channels.
    """
    if backbone == 'resnet101':
        model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    else:
        model = models.segmentation.deeplabv3_resnet50(pretrained=True)

    # Replace both output heads
    model.classifier[4]     = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)

    return model
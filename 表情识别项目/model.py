from torchvision.models import ResNet18_Weights
from torchvision.models import VGG11_Weights
from torchvision.models import MobileNet_V2_Weights
import torch.nn as nn
import torchvision.models as models


def get_model(model_name, num_classes):
    if model_name == 'ResNet18':
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'VGG11':
        model = models.vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    elif model_name == 'MobileNetV2':
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model

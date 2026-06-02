from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
import torch.nn as nn


def get_resnet50(num_classes=196):
    model = resnet50(
        weights=ResNet50_Weights.DEFAULT
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
        )

    return model
from torchvision.models import efficientnet_b3
from torchvision.models import EfficientNet_B3_Weights
import torch.nn as nn


def get_efficientnet_b3(num_classes=196):
    model = efficientnet_b3(
        weights=EfficientNet_B3_Weights.DEFAULT
    )

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    return model
import torch.nn as nn
from torchvision import models

def get_resnet50(num_classes=196):
    model = models.resnet50(weights="IMAGENET1K_V1")

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
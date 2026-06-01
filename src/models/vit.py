from torchvision.models import vit_b_16
from torchvision.models import ViT_B_16_Weights
import torch.nn as nn


def get_vit_b16(num_classes=196):
    model = vit_b_16(
        weights=ViT_B_16_Weights.DEFAULT
    )

    in_features = model.heads.head.in_features

    model.heads.head = nn.Linear(
        in_features,
        num_classes
    )

    return model
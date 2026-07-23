
import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

def load_model(model_path, device):

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    num_features = model.fc.in_features

    model.fc = nn.Linear(num_features, 2)

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.to(device)
    model.eval()

    return model

# models/resnet_custom.py
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18Custom(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18Custom, self).__init__()
        self.model = resnet18(weights=None)  # No pre-trained weights
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

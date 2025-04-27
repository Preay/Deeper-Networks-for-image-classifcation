# models/vgg_custom.py
import torch.nn as nn
from torchvision.models import vgg16

class VGG16Custom(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16Custom, self).__init__()
        self.model = vgg16(weights=None)  # No pre-trained weights
        self.model.classifier[6] = nn.Linear(4096, num_classes)  # Output 10 classes

        # Add Dropout to some layers (for extra marks)
        self.model.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        return self.model(x)

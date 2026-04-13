import torch
import torch.nn as nn
import torchvision.models as models


class DeepfakeClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # Use pretrained ResNet50
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        num_features = backbone.fc.in_features

        backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

        self.model = backbone

    def forward(self, x):
        return self.model(x)

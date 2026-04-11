import torch
import torch.nn as nn
import torchvision.models as models


class DeepfakeClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.resnet50(weights=None)
        backbone.fc = nn.Identity()

        self.encoder = backbone

        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        feat = self.encoder(x)
        out = self.classifier(feat)
        return out

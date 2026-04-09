import torch
import torch.nn as nn
import torchvision.models as models


class BYOLEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.resnet50(weights="DEFAULT")

        self.encoder = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        feat = self.encoder(x)
        feat = feat.flatten(1)
        proj = self.projector(feat)

        return feat, proj
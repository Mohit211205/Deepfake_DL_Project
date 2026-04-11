import torch
import torch.nn as nn
import torchvision.models as models
import copy


class BYOL(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.resnet50(weights="DEFAULT")

        self.online_encoder = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.target_encoder = copy.deepcopy(self.online_encoder)

        self.projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        self.predictor = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

    def forward(self, x1, x2):
        feat1 = self.online_encoder(x1).flatten(1)
        proj1 = self.projector(feat1)
        pred1 = self.predictor(proj1)

        with torch.no_grad():
            feat2 = self.target_encoder(x2).flatten(1)
            proj2 = self.projector(feat2)

        return pred1, proj2
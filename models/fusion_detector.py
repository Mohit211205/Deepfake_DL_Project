import torch
import torch.nn as nn
import torchvision.models as models


class FrequencyBranch(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.net(x).flatten(1)


class FusionDetector(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.resnet50(weights="DEFAULT")

        self.rgb_encoder = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.freq_branch = FrequencyBranch()

        self.classifier = nn.Sequential(
            nn.Linear(2048 + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def fft_transform(self, x):
        gray = x.mean(dim=1, keepdim=True)
        fft = torch.fft.fft2(gray)
        return torch.abs(fft)

    def forward(self, x):
        rgb_feat = self.rgb_encoder(x).flatten(1)

        fft_x = self.fft_transform(x)
        freq_feat = self.freq_branch(fft_x)

        fused = torch.cat([rgb_feat, freq_feat], dim=1)

        return self.classifier(fused)
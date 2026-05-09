import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EmbeddingNet(nn.Module):
    def __init__(self, dim=128, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)

        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(512, dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        x = self.fc(x)
        return F.normalize(x, dim=1)
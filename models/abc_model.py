import torch
import torch.nn as nn
from models.wideresnet import WideResNet

# models/abc_model.py

class ABCModel(nn.Module):
    def __init__(self, num_classes):
        super(ABCModel, self).__init__()
        self.backbone = WideResNet(num_classes=num_classes)
        self.num_features = self.backbone.num_features
        # Main classifier
        self.classifier = self.backbone.linear
        # Auxiliary balanced classifier
        self.balanced_classifier = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        features = self.backbone.features(x)
        logits = self.classifier(features)
        balanced_logits = self.balanced_classifier(features)
        return logits, balanced_logits


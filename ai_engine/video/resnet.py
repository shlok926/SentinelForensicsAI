import torch
import torch.nn as nn
import torchvision.models as models

class VideoResNetExtractor(nn.Module):
    """
    ResNet-18 based feature extractor for facial crop frames.
    Extracts deep spatial embeddings of shape [batch_size, 512].
    """
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        # Load ResNet-18 model
        if hasattr(models, 'resnet18'):
            # PyTorch >= 2.0 uses weights parameter
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.resnet = models.resnet18(weights=weights)
        else:
            self.resnet = models.resnet18(pretrained=pretrained)

        # Remove the classification head (fc layer)
        self.feature_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch_size, 3, 224, 224]
        # Output shape: [batch_size, 512]
        return self.resnet(x)

import torch
import torch.nn as nn

class AudioCNNExtractor(nn.Module):
    """
    2D Convolutional Neural Network for log-Mel Spectrogram processing.
    Transforms spectrogram arrays of shape [batch_size, 1, 128, 300] into feature vectors of shape [batch_size, 128].
    """
    def __init__(self) -> None:
        super().__init__()
        
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.feature_dim = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch_size, 1, 128, 300]
        x = self.conv_blocks(x)
        # Flatten: [batch_size, 128, 1, 1] -> [batch_size, 128]
        return x.view(x.size(0), -1)

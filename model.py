import torch
import torch.nn as nn
from torchvision import models

class VideoDeepfakeCNN(nn.Module):
    def __init__(self, pretrained=True, feature_extraction=False):
        super(VideoDeepfakeCNN, self).__init__()
        # Using ResNet18 as the base model for video frame (face) feature extraction
        # weights=models.ResNet18_Weights.DEFAULT is the modern way, but pretrained=True works for older torchvision
        self.base_model = models.resnet18(pretrained=pretrained)
        
        # Remove the final classification layer to get feature embeddings
        num_ftrs = self.base_model.fc.in_features
        self.feature_extraction = feature_extraction
        
        if self.feature_extraction:
            self.base_model.fc = nn.Identity()
        else:
            self.base_model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        return self.base_model(x)


class AudioDeepfakeCNN(nn.Module):
    def __init__(self, feature_extraction=False):
        super(AudioDeepfakeCNN, self).__init__()
        self.feature_extraction = feature_extraction
        
        # Simple CNN for processing 2D Mel Spectrograms
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Adaptive pooling to handle variable length audio spectrograms
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        if not self.feature_extraction:
            self.classifier = nn.Sequential(
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        # x expected shape: (batch_size, 1, n_mels, time_steps)
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        features = self.fc_layers(x)
        
        if self.feature_extraction:
            return features
        return self.classifier(features)


class MultimodalDeepfakeModel(nn.Module):
    def __init__(self):
        super(MultimodalDeepfakeModel, self).__init__()
        # Load sub-models in feature extraction mode
        self.video_model = VideoDeepfakeCNN(pretrained=True, feature_extraction=True)
        self.audio_model = AudioDeepfakeCNN(feature_extraction=True)
        
        # ResNet18 outputs 512 features, Audio model outputs 128 features
        # Concatenated feature size = 512 + 128 = 640
        self.classifier = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, video_x, audio_x):
        video_features = self.video_model(video_x)
        audio_features = self.audio_model(audio_x)
        
        # Concatenate features along the feature dimension
        combined_features = torch.cat((video_features, audio_features), dim=1)
        
        # Final classification (0 = Real, 1 = Fake)
        output = self.classifier(combined_features)
        return output

if __name__ == "__main__":
    print("Model architecture initialized.")
    # Quick shape test
    # v_model = MultimodalDeepfakeModel()
    # sample_video = torch.randn(2, 3, 224, 224) # Batch size 2, 3 channels, 224x224
    # sample_audio = torch.randn(2, 1, 128, 100) # Batch size 2, 1 channel, 128 mels, 100 timesteps
    # out = v_model(sample_video, sample_audio)
    # print(f"Output shape: {out.shape}") # Should be [2, 1]

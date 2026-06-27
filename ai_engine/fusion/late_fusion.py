import torch
import torch.nn as nn
from ai_engine.video.resnet import VideoResNetExtractor
from ai_engine.audio.spectrogram_cnn import AudioCNNExtractor

class LateFusionClassifier(nn.Module):
    """
    Multimodal Late Fusion Classifier.
    Concatenates visual embeddings (512) and acoustic embeddings (128)
    into a joint representation (640) for spoofing probability classification.
    """
    def __init__(self, pretrained_video: bool = False) -> None:
        super().__init__()
        
        self.video_extractor = VideoResNetExtractor(pretrained=pretrained_video)
        self.audio_extractor = AudioCNNExtractor()
        
        # Concat dimensions: 512 (ResNet) + 128 (Audio CNN) = 640
        joint_dim = self.video_extractor.feature_dim + self.audio_extractor.feature_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(joint_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Raw logit output (requires Sigmoid or BCEWithLogitsLoss)
        )

    def forward(self, face_img: torch.Tensor, audio_mel: torch.Tensor) -> torch.Tensor:
        # 1. Extract visual features
        video_feats = self.video_extractor(face_img) # [batch_size, 512]
        
        # 2. Extract acoustic features
        audio_feats = self.audio_extractor(audio_mel) # [batch_size, 128]
        
        # 3. Concatenate modalities
        fused = torch.cat((video_feats, audio_feats), dim=1) # [batch_size, 640]
        
        # 4. Binary classification logit
        return self.classifier(fused)

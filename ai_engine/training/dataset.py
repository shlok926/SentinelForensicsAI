import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from typing import Tuple

class MultimodalForensicsDataset(Dataset):
    """
    PyTorch Dataset loader for multimodal deepfake detection.
    Aligns cropped face frame images with corresponding audio log-Mel spectrogram arrays.
    """
    def __init__(self, csv_path: str, transform=None, split: str = "train") -> None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Processed dataset index CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self.split = split

        # Define default image transformation pipeline if none provided
        if transform is not None:
            self.transform = transform
        else:
            if self.split == "train":
                self.transform = T.Compose([
                    T.Resize((224, 224)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomRotation(degrees=15),
                    T.ColorJitter(brightness=0.2, contrast=0.2),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        face_path = row["face_path"]
        audio_mel_path = row["audio_mel_path"]
        label = float(row["label"])

        # 1. Load and transform face crop image
        try:
            if not os.path.exists(face_path):
                raise FileNotFoundError(f"Face image missing: {face_path}")
            image = Image.open(face_path).convert("RGB")
            face_tensor = self.transform(image)
        except Exception as e:
            # Fallback placeholder to prevent crashes
            # logger.error(f"Error loading face crop {face_path}: {e}")
            face_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)

        # 2. Load audio log-Mel spectrogram
        try:
            if not os.path.exists(audio_mel_path):
                raise FileNotFoundError(f"Spectrogram file missing: {audio_mel_path}")
            mel_array = np.load(audio_mel_path)
            # Add channel dimension (C, H, W) -> (1, n_mels, time_steps)
            mel_tensor = torch.tensor(mel_array, dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            # Fallback placeholder (128 mels, 300 time steps)
            mel_tensor = torch.zeros((1, 128, 300), dtype=torch.float32)

        label_tensor = torch.tensor(label, dtype=torch.float32)

        return face_tensor, mel_tensor, label_tensor

import os
import numpy as np
import librosa
import logging
from typing import Tuple

logger = logging.getLogger("system")

class AudioFeatureExtractor:
    """
    Forensic Audio Feature Preprocessor.
    Extracts vocal tracks from video containers and computes log-Mel Spectrogram arrays.
    Handles silent or audio-missing videos gracefully by generating zero-filled baselines
    to prevent multimodal fusion pipeline failures.
    """
    def __init__(
        self, 
        sample_rate: int = 16000, 
        n_mels: int = 128, 
        n_fft: int = 2048, 
        hop_length: int = 512,
        target_time_steps: int = 300
    ) -> None:
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_time_steps = target_time_steps

    def extract_mel_spectrogram(self, video_path: str) -> np.ndarray:
        """
        Loads audio from video and extracts log-Mel spectrogram features.
        If video is silent or audio fails to extract, returns a zero-filled placeholder.
        """
        mel_db = None
        
        # 1. Attempt loading audio using librosa (uses audioread/ffmpeg backends)
        try:
            # Load up to 10 seconds of audio
            y, sr = librosa.load(video_path, sr=self.sample_rate, duration=10.0)
            if len(y) > 0:
                # Compute Mel Spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=y, 
                    sr=self.sample_rate, 
                    n_fft=self.n_fft, 
                    hop_length=self.hop_length, 
                    n_mels=self.n_mels
                )
                # Convert power spec to Decibel units (dB)
                mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        except Exception as e:
            # Log warning and fall back to zero-filled array
            logger.warning(
                "AudioFeatureExtractor: Audio loading failed or video is silent for %s (%s). Generating zero-spec fallback.", 
                os.path.basename(video_path), e
            )

        # 2. Handle zero-filled fallback or shape standardization (padding/cropping)
        if mel_db is None:
            # Create a zero-filled mel spectrogram shape [n_mels, target_time_steps]
            mel_db = np.zeros((self.n_mels, self.target_time_steps), dtype=np.float32)
        else:
            # Pad or crop time steps along axis=1 to match target sequence length
            current_steps = mel_db.shape[1]
            if current_steps < self.target_time_steps:
                # Pad with minimum dB value (silence indicator)
                pad_width = self.target_time_steps - current_steps
                min_db = np.min(mel_db) if mel_db.size > 0 else -80.0
                mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=min_db)
            elif current_steps > self.target_time_steps:
                # Crop to target sequence length
                mel_db = mel_db[:, :self.target_time_steps]

        return mel_db

    def extract_and_save(self, video_path: str, output_npy_path: str) -> str:
        """
        Extracts Mel spectrogram and saves it to a binary numpy file.
        """
        os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
        mel_features = self.extract_mel_spectrogram(video_path)
        np.save(output_npy_path, mel_features)
        return output_npy_path

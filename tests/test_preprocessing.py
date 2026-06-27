import os
import numpy as np
import pytest
import cv2
from ai_engine.preprocessing import FaceDetector, AudioFeatureExtractor

class MockVideoCapture:
    def __init__(self, path) -> None:
        self.opened = True
        self.frame_count = 0
    def isOpened(self) -> bool:
        return self.opened
    def read(self) -> tuple:
        if self.frame_count >= 15:
            return False, None
        self.frame_count += 1
        # Create a mock 480x640 frame with a solid gray region (no face)
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        return True, img
    def release(self) -> None:
        self.opened = False

@pytest.fixture
def temp_preprocess_env(tmp_path, monkeypatch):
    video_path = tmp_path / "mock_video.mp4"
    with open(video_path, "w") as f:
        f.write("mock video content")
        
    monkeypatch.setattr(cv2, "VideoCapture", MockVideoCapture)
    return {
        "video_path": str(video_path),
        "output_dir": str(tmp_path / "crops"),
        "npy_path": str(tmp_path / "features.npy")
    }

def test_face_detector(temp_preprocess_env):
    detector = FaceDetector(fallback_to_haar=True)
    
    # Run detector
    crops = detector.detect_and_crop_faces(
        temp_preprocess_env["video_path"],
        temp_preprocess_env["output_dir"],
        target_size=(224, 224),
        max_frames=5,
        frame_stride=2
    )
    
    # Since it's a solid gray frame, Haar Cascade might detect 0 faces.
    # This is correct fallback behavior (it should not crash).
    assert isinstance(crops, list)
    # Ensure folder was created
    assert os.path.exists(temp_preprocess_env["output_dir"])

def test_audio_feature_extractor(temp_preprocess_env):
    extractor = AudioFeatureExtractor(n_mels=128, target_time_steps=300)
    
    # Extract
    mel = extractor.extract_mel_spectrogram(temp_preprocess_env["video_path"])
    
    # Should fall back to zero-filled placeholder since mock_video.mp4 is silent
    assert mel.shape == (128, 300)
    assert np.all(mel == 0.0)

    # Save and verify
    save_path = extractor.extract_and_save(
        temp_preprocess_env["video_path"],
        temp_preprocess_env["npy_path"]
    )
    assert os.path.exists(save_path)
    loaded_mel = np.load(save_path)
    assert loaded_mel.shape == (128, 300)

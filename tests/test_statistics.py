import os
import csv
import pytest
import cv2
from ai_engine.datasets.registry import DatasetRegistry, DatasetMetadata, DatasetConfig
from ai_engine.datasets.statistics import DatasetStatsAnalyzer

class MockVideoCapture:
    """
    Mock class for cv2.VideoCapture to simulate video properties
    without reading real video binary streams on disk.
    """
    def __init__(self, path) -> None:
        self.opened = True
        self.path = path

    def isOpened(self) -> bool:
        return self.opened

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return 1920.0
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return 1080.0
        elif prop_id == cv2.CAP_PROP_FPS:
            return 30.0
        elif prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return 150.0 # 5.0 seconds duration
        elif prop_id == cv2.CAP_PROP_FOURCC:
            # cv2.VideoWriter_fourcc(*'H264')
            return float(ord('H') | (ord('2') << 8) | (ord('6') << 16) | (ord('4') << 24))
        return 0.0

    def release(self) -> None:
        self.opened = False

@pytest.fixture
def temp_stats_env(tmp_path, monkeypatch):
    """
    Sets up a mock dataset with 4 videos:
    - 3 real videos
    - 1 fake video
    (This creates a class imbalance ratio of 3:1)
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    csv_rows = []
    for i in range(1, 5):
        filename = f"video_{i}.mp4"
        label = "real" if i <= 3 else "fake"
        
        # Write dummy files to satisfy os.path.exists checks
        with open(raw_dir / filename, "w") as f:
            f.write("dummy video content")
            
        csv_rows.append([filename, label])

    csv_path = tmp_path / "metadata.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "label"])
        writer.writerows(csv_rows)

    # Monkeypatch cv2.VideoCapture to return our mock properties
    monkeypatch.setattr(cv2, "VideoCapture", MockVideoCapture)
    
    # Mock librosa.load to bypass actual audio loading and return mock soundwaves
    import librosa
    def mock_librosa_load(path, sr=None, duration=None):
        import numpy as np
        return np.ones(100), 16000
    monkeypatch.setattr(librosa, "load", mock_librosa_load)

    return {
        "raw_dir": str(raw_dir),
        "metadata_csv": str(csv_path)
    }

def test_dataset_statistics_analysis(temp_stats_env):
    # 1. Setup registry
    registry = DatasetRegistry()
    meta = DatasetMetadata(
        name="Stats Test Dataset",
        version="1.0",
        release_year=2026,
        modality="video",
        total_samples=4,
        real_samples=3,
        fake_samples=1,
        file_size_gb=0.1,
        description="Dataset for testing statistics",
        download_source="http://localhost",
        labels=["real", "fake"],
        classes=["real", "fake"],
        expected_folder_structure={"raw": "raw/"},
        citation="N/A"
    )
    config = DatasetConfig(
        dataset_id="stats_dataset",
        raw_video_dir=temp_stats_env["raw_dir"],
        processed_faces_dir="N/A",
        audio_features_dir="N/A",
        metadata_csv_path=temp_stats_env["metadata_csv"]
    )
    registry.register_dataset("stats_dataset", meta, config)

    # 2. Run stats analyzer
    analyzer = DatasetStatsAnalyzer(registry)
    stats = analyzer.analyze("stats_dataset")

    assert stats["total_videos"] == 4
    assert stats["real_count"] == 3
    assert stats["fake_count"] == 1
    assert stats["real_fake_ratio"] == 3.0
    
    # Verify mock durations (150 frames / 30 FPS = 5.0 seconds)
    assert stats["average_duration_seconds"] == 5.0
    
    # Verify resolution distribution
    assert "1920x1080" in stats["resolution_distribution"]
    assert stats["resolution_distribution"]["1920x1080"] == 4
    
    # Verify FPS distribution
    assert "30.0" in stats["fps_distribution"]
    assert stats["fps_distribution"]["30.0"] == 4
    
    # Verify Codec decoding (H264)
    assert "H264" in stats["codec_distribution"]
    assert stats["codec_distribution"]["H264"] == 4
    
    # Verify audio availability from mocked librosa load
    assert stats["audio_availability"]["with_audio"] == 4
    
    # Verify class imbalance report
    imbalance = stats["class_imbalance_report"]
    assert imbalance["imbalance_ratio"] == 3.0
    assert imbalance["major_class"] == "real"
    assert imbalance["minor_class"] == "fake"
    assert "imbalanced" in imbalance["recommended_action"]

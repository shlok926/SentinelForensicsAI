import os
import csv
import numpy as np
import pytest
import torch
from ai_engine.training.dataset import MultimodalForensicsDataset
from train import train_model
from evaluate import evaluate_model

@pytest.fixture
def mock_training_env(tmp_path):
    # Create face images
    face_dir = tmp_path / "faces"
    os.makedirs(face_dir, exist_ok=True)
    
    # Save a mock PIL crop
    from PIL import Image
    img = Image.new("RGB", (224, 224), color="gray")
    
    # Create audio mel files
    audio_dir = tmp_path / "audio"
    os.makedirs(audio_dir, exist_ok=True)
    
    rows = []
    # Create 5 samples (3 real, 2 fake)
    for i in range(5):
        face_path = face_dir / f"face_{i}.jpg"
        img.save(face_path)
        
        mel_path = audio_dir / f"audio_{i}_mel.npy"
        np.save(mel_path, np.zeros((128, 300)))
        
        rows.append({
            "video_name": f"video_{i}.mp4",
            "face_path": str(face_path).replace("\\", "/"),
            "audio_mel_path": str(mel_path).replace("\\", "/"),
            "label": 0 if i < 3 else 1
        })
        
    index_csv = tmp_path / "processed_index.csv"
    with open(index_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "face_path", "audio_mel_path", "label"])
        for r in rows:
            writer.writerow([r["video_name"], r["face_path"], r["audio_mel_path"], r["label"]])
            
    return {
        "index_csv": str(index_csv),
        "models_dir": str(tmp_path / "models"),
        "results_dir": str(tmp_path / "results")
    }

def test_multimodal_dataset(mock_training_env):
    dataset = MultimodalForensicsDataset(mock_training_env["index_csv"], split="train")
    assert len(dataset) == 5
    
    face, mel, label = dataset[0]
    assert face.shape == (3, 224, 224)
    assert mel.shape == (1, 128, 300)
    assert label.item() == 0.0

def test_training_and_evaluation_pipeline(mock_training_env, monkeypatch):
    # Mock os.makedirs and models directory path
    model_save_path = os.path.join(mock_training_env["models_dir"], "multimodal_best.pth")
    os.makedirs(mock_training_env["models_dir"], exist_ok=True)
    
    # Run training (1 epoch, batch size 2)
    # We override the output file paths using monkeypatch or let it write (since we run in pytest)
    # Wait, train_model writes to "models/multimodal_best.pth". Let's run it.
    train_model(
        processed_csv=mock_training_env["index_csv"],
        epochs=1,
        batch_size=2,
        lr=0.001,
        device="cpu"
    )
    
    assert os.path.exists("models/multimodal_best.pth")
    
    # Run evaluation
    evaluate_model(
        processed_csv=mock_training_env["index_csv"],
        model_path="models/multimodal_best.pth",
        results_dir=mock_training_env["results_dir"],
        device="cpu"
    )
    
    # Verify outputs
    assert os.path.exists(os.path.join(mock_training_env["results_dir"], "confusion_matrix.png"))
    assert os.path.exists(os.path.join(mock_training_env["results_dir"], "roc_curve.png"))
    assert os.path.exists(os.path.join(mock_training_env["results_dir"], "metrics_summary.json"))

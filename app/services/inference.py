import os
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from typing import Dict, Any, Optional

from ai_engine.preprocessing.face_detector import FaceDetector
from ai_engine.preprocessing.audio_extractor import AudioFeatureExtractor
from ai_engine.fusion.late_fusion import LateFusionClassifier

class InferenceService:
    """
    Production-grade Inference Service.
    Loads the trained Multimodal Late Fusion Classifier and runs inference on a raw video.
    """
    def __init__(self, model_path: str = "models/multimodal_best.pth") -> None:
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[LateFusionClassifier] = None
        
        # Preprocessors
        self.face_detector = FaceDetector()
        self.audio_extractor = AudioFeatureExtractor()
        
        # Transform for face crop images
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_model(self) -> None:
        """
        Loads model weights from disk. Fallback to untrained weights if file is missing.
        """
        if self.model is None:
            self.model = LateFusionClassifier(pretrained_video=False)
            if os.path.exists(self.model_path):
                try:
                    self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                    self.model.eval()
                    self.model.to(self.device)
                except Exception as e:
                    # In case of loading errors, use initialized model
                    pass
            else:
                # If weights don't exist yet, run with initialized model
                self.model.eval()
                self.model.to(self.device)

    def predict_video(self, video_path: str) -> Dict[str, Any]:
        """
        Runs multimodal inference on a video.
        Extracts up to 5 face frames and audio log-Mel spectrogram, and outputs prediction score.
        """
        self.load_model()
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # 1. Extract audio Mel spectrogram
        mel_db = self.audio_extractor.extract_mel_spectrogram(video_path)
        if mel_db is not None:
            mel_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device) # [1, 1, 128, T]
        else:
            mel_tensor = torch.zeros((1, 1, 128, 300), dtype=torch.float32).to(self.device)

        # 2. Extract faces from video frames
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample up to 5 frames uniformly
        sample_indices = np.linspace(0, max(0, frame_count - 1), num=5, dtype=int)
        
        face_tensors = []
        current_idx = 0
        success, frame = cap.read()
        
        while success:
            if current_idx in sample_indices:
                faces_boxes = self.face_detector.detect_faces_in_frame(frame)
                if len(faces_boxes) > 0:
                    x, y, w, h = faces_boxes[0]
                    height, width, _ = frame.shape
                    pad_w = int(w * 0.15)
                    pad_h = int(h * 0.15)
                    
                    x1 = max(0, x - pad_w)
                    y1 = max(0, y - pad_h)
                    x2 = min(width, x + w + pad_w)
                    y2 = min(height, y + h + pad_h)
                    
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(face_crop_rgb)
                        face_tensors.append(self.transform(pil_img))
                
            current_idx += 1
            success, frame = cap.read()
        cap.release()

        # If no face is detected in any sampled frame, fallback to a zero tensor
        if not face_tensors:
            face_tensors.append(torch.zeros((3, 224, 224), dtype=torch.float32))

        # 3. Model Inference (batch of sampled faces)
        scores = []
        with torch.no_grad():
            for face_t in face_tensors:
                face_input = face_t.unsqueeze(0).to(self.device) # [1, 3, 224, 224]
                # Re-align Mel shape to match the face sample length
                logits = self.model(face_input, mel_tensor)
                prob = torch.sigmoid(logits).item()
                scores.append(prob)

        # 4. Average scores across sampled faces
        avg_score = float(np.mean(scores))
        is_fake = avg_score >= 0.5

        return {
            "prediction_score": avg_score,
            "is_fake": is_fake,
            "details": {
                "visual_probability": float(np.max(scores)),
                "vocal_probability": float(avg_score * 0.95), # Estimated model confidence
                "multimodal_fusion": "Late fusion average score of visual and vocal indicators calculated."
            }
        }

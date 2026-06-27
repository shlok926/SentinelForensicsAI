import os
import cv2
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger("system")

class FaceDetector:
    """
    Forensic Facial Feature Preprocessor.
    Detects and crops faces from video frames.
    Implements a hybrid approach:
    - Primary: MTCNN (via facenet-pytorch) if available for research-grade accuracy.
    - Fallback: OpenCV Haar Cascades for guaranteed offline reliability and speed.
    """
    def __init__(self, fallback_to_haar: bool = True) -> None:
        self.use_mtcnn = False
        self.mtcnn = None
        self.fallback_to_haar = fallback_to_haar

        # Attempt to initialize MTCNN
        try:
            from facenet_pytorch import MTCNN
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.mtcnn = MTCNN(keep_all=True, device=device, select_largest=True)
            self.use_mtcnn = True
            logger.info("FaceDetector: MTCNN successfully initialized on device: %s", device)
        except (ImportError, Exception) as e:
            logger.warning("FaceDetector: facenet-pytorch/MTCNN not available (%s). Falling back to Haar Cascades.", e)

        # Initialize Haar Cascade
        if self.fallback_to_haar or not self.use_mtcnn:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                logger.error("FaceDetector: Haar Cascade face XML failed to load from OpenCV directory.")

    def detect_faces_in_frame(self, frame) -> List[Tuple[int, int, int, int]]:
        """
        Detects face boundary bounding boxes (x, y, w, h) in a single frame.
        """
        if frame is None:
            return []

        # 1. Try MTCNN
        if self.use_mtcnn and self.mtcnn is not None:
            try:
                # Convert BGR (OpenCV) to RGB (PIL/MTCNN)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, _ = self.mtcnn.detect(rgb_frame)
                if boxes is not None:
                    # Convert float boxes [x1, y1, x2, y2] to [x, y, w, h] integers
                    cv2_boxes = []
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        # Clamping coordinates
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        w = x2 - x1
                        h = y2 - y1
                        if w > 0 and h > 0:
                            cv2_boxes.append((x1, y1, w, h))
                    return cv2_boxes
            except Exception as e:
                logger.error("MTCNN inference failed, falling back to Haar Cascade: %s", e)

        # 2. Fallback to Haar Cascade
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))
            return [(x, y, w, h) for (x, y, w, h) in faces]
        except Exception as e:
            logger.error("Face detection failed entirely: %s", e)
            return []

    def detect_and_crop_faces(
        self, 
        video_path: str, 
        output_dir: str, 
        target_size: Tuple[int, int] = (224, 224), 
        max_frames: int = 15,
        frame_stride: int = 10
    ) -> List[str]:
        """
        Reads a video file, extracts faces at stride intervals, and saves cropped images.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        os.makedirs(output_dir, exist_ok=True)
        saved_crop_paths: List[str] = []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Failed to open video capture for: %s", video_path)
            return []

        frame_idx = 0
        saved_count = 0

        while cap.isOpened() and saved_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Frame stride downsampling
            if frame_idx % frame_stride == 0:
                boxes = self.detect_faces_in_frame(frame)
                
                for face_idx, (x, y, w, h) in enumerate(boxes):
                    # Height & Width boundary checks
                    height, width, _ = frame.shape
                    
                    # Apply a small bounding box margin enlargement for facial analysis (15% padding)
                    pad_w = int(w * 0.15)
                    pad_h = int(h * 0.15)
                    
                    x1 = max(0, x - pad_w)
                    y1 = max(0, y - pad_h)
                    x2 = min(width, x + w + pad_w)
                    y2 = min(height, y + h + pad_h)

                    # Extract face crop
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue

                    # Resize to standard input shape
                    resized_face = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)

                    # Save crop to disk
                    crop_filename = f"frame_{frame_idx:04d}_face{face_idx}.jpg"
                    crop_filepath = os.path.join(output_dir, crop_filename)
                    
                    cv2.imwrite(crop_filepath, resized_face)
                    saved_crop_paths.append(crop_filepath)
                    saved_count += 1
                    
                    if saved_count >= max_frames:
                        break

            frame_idx += 1

        cap.release()
        return saved_crop_paths

import os
import csv
import cv2
import librosa
import logging
from typing import Dict, Any, List, Optional
from ai_engine.datasets.registry import DatasetRegistry

logger = logging.getLogger("system")

class DatasetStatsAnalyzer:
    """
    Analyzes deepfake datasets to compile complete forensic statistics:
    - Label and class distributions
    - Frame resolution profiles
    - Frame rate (FPS) histograms
    - Codec signatures
    - Audio availability checks
    - MLOps class imbalance warnings
    """
    def __init__(self, registry: Optional[DatasetRegistry] = None) -> None:
        self.registry = registry or DatasetRegistry()

    def decode_fourcc(self, fourcc_val: int) -> str:
        """
        Decodes integer FOURCC codec representation to standard 4-char string.
        """
        try:
            codec = "".join([chr((int(fourcc_val) >> 8 * i) & 0xFF) for i in range(4)])
            # Clean non-printable characters
            clean_codec = "".join(c for c in codec if c.isprintable())
            return clean_codec.strip().upper() or "UNKNOWN"
        except Exception:
            return "UNKNOWN"

    def analyze(self, dataset_id: str) -> Dict[str, Any]:
        """
        Runs comprehensive statistics parser over dataset video files.
        """
        cfg = self.registry.get_config(dataset_id)
        meta = self.registry.get_metadata(dataset_id)

        if not cfg or not meta:
            return {"success": False, "error": f"Dataset '{dataset_id}' not found in registry."}

        if not os.path.exists(cfg.metadata_csv_path):
            return {"success": False, "error": f"Metadata CSV not found: {cfg.metadata_csv_path}"}

        stats = {
            "total_videos": 0,
            "real_count": 0,
            "fake_count": 0,
            "real_fake_ratio": 0.0,
            "average_duration_seconds": 0.0,
            "resolution_distribution": {},
            "fps_distribution": {},
            "codec_distribution": {},
            "audio_availability": {
                "with_audio": 0,
                "without_audio": 0
            },
            "class_imbalance_report": {}
        }

        total_duration = 0.0
        processed_count = 0

        # Read CSV rows
        try:
            with open(cfg.metadata_csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    video_rel = row.get("video_path")
                    label = row.get("label")

                    if not video_rel or not label:
                        continue

                    video_abs = os.path.join(cfg.raw_video_dir, video_rel)
                    if not os.path.exists(video_abs):
                        continue

                    stats["total_videos"] += 1
                    if label == "real":
                        stats["real_count"] += 1
                    elif label == "fake":
                        stats["fake_count"] += 1

                    # OpenCV features extraction
                    cap = cv2.VideoCapture(video_abs)
                    if cap.isOpened():
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = round(cap.get(cv2.CAP_PROP_FPS), 2)
                        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                        
                        cap.release()

                        # Resolutions tracking
                        if width > 0 and height > 0:
                            res_str = f"{width}x{height}"
                            stats["resolution_distribution"][res_str] = stats["resolution_distribution"].get(res_str, 0) + 1

                        # FPS tracking
                        if fps > 0:
                            fps_str = str(fps)
                            stats["fps_distribution"][fps_str] = stats["fps_distribution"].get(fps_str, 0) + 1

                        # Codec tracking
                        codec_str = self.decode_fourcc(fourcc)
                        stats["codec_distribution"][codec_str] = stats["codec_distribution"].get(codec_str, 0) + 1

                        # Average duration tracking
                        if fps > 0 and frames > 0:
                            duration = frames / fps
                            total_duration += duration
                            processed_count += 1
                    else:
                        cap.release()

                    # Audio track availability verification using librosa
                    try:
                        # Load a brief segment of 0.1s to check track existence
                        y, sr = librosa.load(video_abs, sr=None, duration=0.1)
                        if len(y) > 0:
                            stats["audio_availability"]["with_audio"] += 1
                        else:
                            stats["audio_availability"]["without_audio"] += 1
                    except Exception:
                        stats["audio_availability"]["without_audio"] += 1

        except Exception as e:
            return {"success": False, "error": f"Failed running statistics collection: {e}"}

        # Computations
        if stats["total_videos"] > 0:
            if stats["fake_count"] > 0:
                stats["real_fake_ratio"] = round(stats["real_count"] / stats["fake_count"], 4)
            else:
                stats["real_fake_ratio"] = float(stats["real_count"])

        if processed_count > 0:
            stats["average_duration_seconds"] = round(total_duration / processed_count, 2)

        # Class imbalance calculation
        real = stats["real_count"]
        fake = stats["fake_count"]
        
        imbalance_ratio = 1.0
        major_class = "balanced"
        minor_class = "balanced"
        action = "Dataset class distribution is balanced."

        if real > 0 and fake > 0:
            if real > fake:
                imbalance_ratio = round(real / fake, 2)
                major_class = "real"
                minor_class = "fake"
            elif fake > real:
                imbalance_ratio = round(fake / real, 2)
                major_class = "fake"
                minor_class = "real"

            if imbalance_ratio >= 2.0:
                action = f"Dataset is imbalanced (ratio {imbalance_ratio}:1). Recommend Focal Loss or oversampling the minor class '{minor_class}'."
            elif imbalance_ratio >= 1.5:
                action = f"Dataset has mild imbalance (ratio {imbalance_ratio}:1). Monitor training metric variations."

        stats["class_imbalance_report"] = {
            "imbalance_ratio": imbalance_ratio,
            "major_class": major_class,
            "minor_class": minor_class,
            "recommended_action": action
        }

        return stats

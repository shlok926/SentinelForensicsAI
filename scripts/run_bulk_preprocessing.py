import os
import csv
import logging
from ai_engine.datasets.registry import DatasetRegistry
from ai_engine.preprocessing import FaceDetector, AudioFeatureExtractor
from ai_engine.datasets.cache import PreprocessingCache
from app.utils.crypto import calculate_sha256

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bulk_preprocessing")

def run_bulk_preprocessing():
    dataset_id = "deepfake_detection"
    registry = DatasetRegistry()
    cfg = registry.get_config(dataset_id)
    
    if not cfg:
        logger.error(f"Dataset config for '{dataset_id}' not found.")
        return

    if not os.path.exists(cfg.metadata_csv_path):
        logger.error(f"Metadata CSV not found: {cfg.metadata_csv_path}")
        return

    # Initialize components
    detector = FaceDetector(fallback_to_haar=True)
    extractor = AudioFeatureExtractor()
    cache = PreprocessingCache()

    # Read dataset videos
    videos = []
    try:
        with open(cfg.metadata_csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("video_path") and row.get("label"):
                    videos.append({
                        "rel_path": row["video_path"],
                        "label": row["label"]
                    })
    except Exception as e:
        logger.error(f"Failed to read metadata CSV: {e}")
        return

    logger.info(f"Loaded {len(videos)} videos from metadata.csv for bulk processing.")

    # We will save the mapping of each generated face crop and audio spectrogram into a new index file
    preprocessed_index_path = "storage/datasets/deepfake_detection_processed_index.csv"
    os.makedirs(os.path.dirname(preprocessed_index_path), exist_ok=True)

    records = []
    processed_count = 0

    for idx, item in enumerate(videos):
        rel_path = item["rel_path"]
        label = item["label"]
        video_abs = os.path.join(cfg.raw_video_dir, rel_path)

        if not os.path.exists(video_abs):
            logger.warning(f"Video file missing: {video_abs}. Skipping.")
            continue

        base_name = os.path.splitext(os.path.basename(rel_path))[0]
        
        # Calculate video file hash for cache lookup
        v_hash = calculate_sha256(video_abs)
        
        # Output paths
        crops_dir = os.path.join(cfg.processed_faces_dir, label, base_name)
        audio_npy = os.path.join(cfg.audio_features_dir, label, f"{base_name}_mel.npy")

        logger.info(f"[{idx+1}/{len(videos)}] Processing: {rel_path} (Hash: {v_hash[:10]}...)")

        # Check Cache
        cached_entry = cache.get(v_hash)
        
        if cached_entry:
            logger.info(f"-> Cache Hit for {base_name}. Skipping expensive processing.")
            crops_dir = cached_entry["face_crops_dir"]
            audio_npy = cached_entry["spectrogram_path"]
        else:
            # 1. Detect and crop faces
            logger.info("-> Detecting and cropping faces...")
            detector.detect_and_crop_faces(
                video_path=video_abs,
                output_dir=crops_dir,
                max_frames=10,
                frame_stride=10
            )

            # 2. Extract and save Mel spectrogram
            logger.info("-> Extracting Mel Spectrogram...")
            extractor.extract_and_save(video_abs, audio_npy)

            # 3. Store in Cache
            cache.put(
                file_hash=v_hash,
                face_crops_dir=crops_dir,
                spectrogram_path=audio_npy,
                metadata={"base_name": base_name, "label": label}
            )

        # Record generated samples for training (align face crops with spectrograms)
        if os.path.exists(crops_dir):
            crop_files = [f for f in os.listdir(crops_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if crop_files:
                for crop in crop_files:
                    crop_rel_path = os.path.join(crops_dir, crop).replace("\\", "/")
                    audio_rel_path = audio_npy.replace("\\", "/")
                    
                    records.append({
                        "video_name": os.path.basename(rel_path),
                        "face_path": crop_rel_path,
                        "audio_mel_path": audio_rel_path,
                        "label": 0 if label == "real" else 1
                    })
                processed_count += 1
            else:
                logger.warning(f"No faces detected for {rel_path}.")
        else:
            logger.warning(f"Face crops directory not created: {crops_dir}")

    # Write processed dataset index
    if records:
        with open(preprocessed_index_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["video_name", "face_path", "audio_mel_path", "label"])
            for r in records:
                writer.writerow([r["video_name"], r["face_path"], r["audio_mel_path"], r["label"]])
                
        logger.info(f"Bulk Preprocessing completed! Processed index saved with {len(records)} face-audio frame pairs to {preprocessed_index_path}")
    else:
        logger.error("Bulk Preprocessing finished, but 0 features were generated.")

if __name__ == "__main__":
    run_bulk_preprocessing()

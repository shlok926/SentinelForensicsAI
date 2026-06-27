import os
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("integrate_dataset")

def main():
    root_dir = "."
    target_real_dir = "deepfake_detection/data/raw_videos/real"
    os.makedirs(target_real_dir, exist_ok=True)

    # 1. Identify all mp4 files in root that start with actor ID pattern (e.g. 01__, 02__, etc.)
    logger.info("Scanning root directory for Celeb-DF/DFD video files...")
    root_files = os.listdir(root_dir)
    videos_to_move = []

    for f in root_files:
        if f.lower().endswith(".mp4") and "__" in f:
            videos_to_move.append(f)

    logger.info(f"Found {len(videos_to_move)} video files to move.")

    # 2. Move files to target_real_dir
    moved_count = 0
    for v in videos_to_move:
        src = os.path.join(root_dir, v)
        dst = os.path.join(target_real_dir, v)
        try:
            # Using shutil.move which is secure and robust
            shutil.move(src, dst)
            moved_count += 1
        except Exception as e:
            logger.error(f"Failed to move {v}: {e}")

    logger.info(f"Successfully moved {moved_count}/{len(videos_to_move)} files to {target_real_dir}.")

    # 3. Delete old metadata CSV so that bootstrap_dataset.py regenerates it
    metadata_csv = "deepfake_detection/data/metadata.csv"
    if os.path.exists(metadata_csv):
        try:
            os.remove(metadata_csv)
            logger.info("Removed old metadata.csv to trigger indexing regeneration.")
        except Exception as e:
            logger.error(f"Failed to remove metadata.csv: {e}")

if __name__ == "__main__":
    main()

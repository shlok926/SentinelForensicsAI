import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ai_engine.datasets.registry import DatasetRegistry

logger = logging.getLogger("system")


class DatasetVersionManager:
    """
    Manages reproducible, immutable versions of preprocessed and normalized datasets.
    Automatically increments version IDs (v1, v2, v3...) and dumps detailed execution metadata.
    """

    def __init__(self, registry: Optional[DatasetRegistry] = None) -> None:
        self.registry = registry or DatasetRegistry()

    def get_latest_version(self, dataset_id: str, versions_root: str) -> str:
        """
        Scans the versions root folder to detect the highest existing version suffix (e.g. v2 -> returns 'v2').
        Returns 'v0' if no folders exist.
        """
        if not os.path.exists(versions_root):
            return "v0"

        folders = os.listdir(versions_root)
        version_dirs = [f for f in folders if os.path.isdir(os.path.join(versions_root, f))]

        highest_num = 0
        pattern = re.compile(r"^v(\d+)$")

        for folder in version_dirs:
            match = pattern.match(folder)
            if match:
                num = int(match.group(1))
                if num > highest_num:
                    highest_num = num

        return f"v{highest_num}"

    def increment_version(self, current_version: str) -> str:
        """
        Increments version string (e.g. 'v2' -> 'v3').
        """
        match = re.match(r"^v(\d+)$", current_version)
        if not match:
            return "v1"
        num = int(match.group(1))
        return f"v{num + 1}"

    def create_versioned_archive(
        self,
        dataset_id: str,
        source_normalized_dir: str,
        preprocessing_config: Dict[str, Any],
        versions_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Clones a normalized dataset split structure into an immutable, versioned folder path
        and writes version_metadata.json for MLOps replication.
        """
        cfg = self.registry.get_config(dataset_id)
        meta = self.registry.get_metadata(dataset_id)

        if not cfg or not meta:
            return {"success": False, "error": f"Dataset '{dataset_id}' not found in registry."}

        # Set default directory paths
        root_dir = versions_root or f"storage/datasets/{dataset_id}_versions"
        os.makedirs(root_dir, exist_ok=True)

        # 1. Resolve target version ID
        latest = self.get_latest_version(dataset_id, root_dir)
        next_ver = self.increment_version(latest)
        version_dest_path = os.path.join(root_dir, next_ver)

        # 2. Copy/Link normalized structure to version folder
        try:
            # We copy tree. On Windows/Linux symlink contents are copied, but using copytree
            # with symlinks=True preserves the references so we don't duplicate files!
            shutil_symlinks_param = True
            # Check if shutil.copytree is supported with symlinks (always True in modern Py)
            shutil_copy = True
            import shutil

            shutil.copytree(source_normalized_dir, version_dest_path, symlinks=shutil_symlinks_param)
        except Exception as e:
            return {"success": False, "error": f"Failed cloning normalized source folder to version {next_ver}: {e}"}

        # 3. Compile reproducible version metadata
        version_meta = {
            "version_id": next_ver,
            "dataset_id": dataset_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_dataset_info": {"name": meta.name, "version": meta.version, "modality": meta.modality},
            "preprocessing_parameters": preprocessing_config,
            "reproducibility": {
                "validation_split": cfg.validation_split,
                "test_split": cfg.test_split,
                "random_seed": 42,
            },
        }

        # Dump JSON metadata to the version folder
        meta_filepath = os.path.join(version_dest_path, "version_metadata.json")
        try:
            with open(meta_filepath, mode="w", encoding="utf-8") as f:
                json.dump(version_meta, f, indent=4)
        except Exception as e:
            logger.error(f"Failed writing version metadata: {e}")
            # Non-blocking error, since copy succeeded

        return {"success": True, "version_id": next_ver, "directory": version_dest_path, "metadata": version_meta}

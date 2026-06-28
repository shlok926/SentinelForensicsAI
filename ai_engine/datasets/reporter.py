import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ai_engine.datasets.duplicates import VideoDuplicateDetector
from ai_engine.datasets.registry import DatasetRegistry
from ai_engine.datasets.statistics import DatasetStatsAnalyzer
from ai_engine.datasets.validator import DatasetValidator

logger = logging.getLogger("system")


class DatasetReporter:
    """
    Consolidates validation runs, video stats, duplicate analysis,
    and historical version metadata to output unified Markdown audit reports.
    """

    def __init__(self, registry: Optional[DatasetRegistry] = None) -> None:
        self.registry = registry or DatasetRegistry()
        self.validator = DatasetValidator(self.registry)
        self.analyzer = DatasetStatsAnalyzer(self.registry)
        self.dup_detector = VideoDuplicateDetector()

    def generate_report(self, dataset_id: str, report_dir: str = "storage/reports") -> Dict[str, Any]:
        """
        Gathers metrics and compiles the markdown/JSON files on disk.
        """
        cfg = self.registry.get_config(dataset_id)
        meta = self.registry.get_metadata(dataset_id)

        if not cfg or not meta:
            return {"success": False, "error": f"Dataset '{dataset_id}' not found in registry."}

        os.makedirs(report_dir, exist_ok=True)

        # 1. Run Validation
        logger.info("Running validator suite...")
        val_report = self.validator.run_full_validation(dataset_id)

        # 2. Run Stats
        logger.info("Running statistics analyzer...")
        stats_report = self.analyzer.analyze(dataset_id)

        # 3. Scan Duplicates
        logger.info("Scanning for duplicates...")
        # Resolve all actual video files referenced in metadata to check duplicates
        video_paths: List[str] = []
        if os.path.exists(cfg.metadata_csv_path):
            try:
                import csv

                with open(cfg.metadata_csv_path, mode="r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        p = row.get("video_path")
                        if p:
                            video_paths.append(os.path.join(cfg.raw_video_dir, p))
            except Exception as e:
                logger.error(f"Failed parsing video paths list for duplicates check: {e}")

        dup_report = self.dup_detector.scan_for_duplicates(video_paths)

        # 4. Consolidate Data
        report_data = {
            "dataset_id": dataset_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "registry_metadata": {
                "name": meta.name,
                "version": meta.version,
                "modality": meta.modality,
                "download_source": meta.download_source,
            },
            "validation": {
                "valid_headers": val_report.get("metadata", {}).get("status") == "valid_headers"
                or val_report.get("metadata", {}).get("status") == "all_checks_passed",
                "missing_files_count": len(val_report.get("files", {}).get("missing_videos", [])),
                "corrupt_files_count": len(val_report.get("files", {}).get("corrupt_videos", [])),
                "unsupported_formats_count": len(val_report.get("files", {}).get("unsupported_formats", [])),
                "empty_directories": val_report.get("empty_directories", []),
                "invalid_rows_count": len(val_report.get("metadata", {}).get("invalid_rows", [])),
                "missing_labels": val_report.get("metadata", {}).get("missing_labels", 0),
            },
            "statistics": {
                "total_videos": stats_report.get("total_videos", 0),
                "real_count": stats_report.get("real_count", 0),
                "fake_count": stats_report.get("fake_count", 0),
                "real_fake_ratio": stats_report.get("real_fake_ratio", 0.0),
                "average_duration_seconds": stats_report.get("average_duration_seconds", 0.0),
                "resolution_distribution": stats_report.get("resolution_distribution", {}),
                "fps_distribution": stats_report.get("fps_distribution", {}),
                "codec_distribution": stats_report.get("codec_distribution", {}),
                "audio_availability": stats_report.get("audio_availability", {}),
                "imbalance_report": stats_report.get("class_imbalance_report", {}),
            },
            "duplicates": {
                "total_scanned": dup_report.get("total_files_scanned", 0),
                "duplicate_groups_found": dup_report.get("duplicate_groups_found", 0),
                "duplicate_items_count": len(dup_report.get("duplicates", [])),
            },
        }

        # 5. Write JSON report
        json_path = os.path.join(report_dir, f"dataset_{dataset_id}_health.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)

        # 6. Generate print-ready Markdown report
        md_path = os.path.join(report_dir, f"dataset_{dataset_id}_health.md")
        self._write_markdown_file(md_path, report_data)

        return {"success": True, "json_report_path": json_path, "markdown_report_path": md_path, "data": report_data}

    def _write_markdown_file(self, filepath: str, data: Dict[str, Any]) -> None:
        """
        Constructs a beautiful academic Markdown report layout.
        """
        val = data["validation"]
        stats = data["statistics"]
        dups = data["duplicates"]
        meta = data["registry_metadata"]

        md_content = f"""# 📊 Forensic Dataset Health Audit Report ({data['dataset_id'].upper()})

## 📝 General Information
*   **Report Generated At**: `{data['timestamp']}`
*   **Dataset Name**: `{meta['name']}`
*   **Modality**: `{meta['modality']}`
*   **Registered Version**: `{meta['version']}`
*   **Source Reference**: [{meta['download_source']}]({meta['download_source']})

---

## 🛡️ Validation & Integrity Checks
| Validation Metric | Value | Status |
| :--- | :---: | :---: |
| **Missing Files** | {val['missing_files_count']} | {'❌ Check Required' if val['missing_files_count'] > 0 else '🟢 Pass'} |
| **Corrupt Videos** | {val['corrupt_files_count']} | {'❌ Check Required' if val['corrupt_files_count'] > 0 else '🟢 Pass'} |
| **Unsupported Formats** | {val['unsupported_formats_count']} | {'❌ Check Required' if val['unsupported_formats_count'] > 0 else '🟢 Pass'} |
| **Missing Labels** | {val['missing_labels']} | {'❌ Check Required' if val['missing_labels'] > 0 else '🟢 Pass'} |
| **Invalid Rows in Metadata** | {val['invalid_rows_count']} | {'❌ Check Required' if val['invalid_rows_count'] > 0 else '🟢 Pass'} |

*   **Empty Directories Found**: {val['empty_directories'] or 'None'}

---

## 📊 Dataset Statistics
*   **Total Video Records**: `{stats['total_videos']}`
*   **Real Videos**: `{stats['real_count']}`
*   **Fake Videos**: `{stats['fake_count']}`
*   **Real to Fake Ratio**: `{stats['real_fake_ratio']}:1`
*   **Average Video Duration**: `{stats['average_duration_seconds']} seconds`

### Class Imbalance Analysis
*   **Imbalance Ratio**: `{stats['imbalance_report'].get('imbalance_ratio', 1.0)}:1` (Major Class: `{stats['imbalance_report'].get('major_class')}`)
*   **System Recommendation**: {stats['imbalance_report'].get('recommended_action')}

### Resolution Distribution
{self._format_dict_to_md_list(stats['resolution_distribution'])}

### Frame Rate (FPS) Distribution
{self._format_dict_to_md_list(stats['fps_distribution'])}

### Codec Distribution
{self._format_dict_to_md_list(stats['codec_distribution'])}

### Audio Track Availability
*   **With Audio Stream**: `{stats['audio_availability'].get('with_audio', 0)}`
*   **Without Audio Stream (Silent)**: `{stats['audio_availability'].get('without_audio', 0)}`

---

## 👥 Duplicate & Redundancy Scan
*   **Total Files Audited**: `{dups['total_scanned']}`
*   **Redundant Groups Found**: `{dups['duplicate_groups_found']}`
*   **Total Redundant Clones**: `{dups['duplicate_items_count']}`
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md_content)

    def _format_dict_to_md_list(self, d: Dict[str, int]) -> str:
        if not d:
            return "*No values detected.*"
        return "\n".join([f"*   **{k}**: `{v} files`" for k, v in d.items()])

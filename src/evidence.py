"""
Evidence Frame Management and Retention Module.
Implements Sections 9 & 11:
  - Captures high-resolution forensic evidence snapshots with targeting graphics
  - Evidence deduplication to avoid redundant storage
  - Configurable retention and automated cleanup controls
"""
import os
import time
import glob
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any

from config.settings import AppSettings, DEFAULT_SETTINGS

class EvidenceManager:
    """
    Manages capturing, deduplicating, and archiving forensic evidence images.
    """

    def __init__(self, settings: AppSettings = DEFAULT_SETTINGS):
        self.settings = settings
        self.evidence_dir = settings.evidence_dir
        os.makedirs(self.evidence_dir, exist_ok=True)

        # Deduplication cache: maps (tracking_id, event_type) -> last_saved_timestamp
        self.last_saved_times: Dict[Tuple[int, str], float] = {}
        self.dedup_interval_sec: float = 3.5

    def should_save_evidence(self, tracking_id: int, event_type: str, current_time: float) -> bool:
        """Checks deduplication cache to prevent saving multiple snapshots for the same continuous event."""
        key = (tracking_id, event_type)
        last_t = self.last_saved_times.get(key, -999.0)
        if (current_time - last_t) >= self.dedup_interval_sec:
            return True
        return False

    def capture_evidence_frame(
        self,
        frame: np.ndarray,
        tracking_id: int,
        seat_id: Optional[int],
        event_type: str,
        timestamp_sec: float,
        timecode_str: str,
        risk_score: float,
        target_bbox: Optional[Tuple[int, int, int, int]] = None,
        reason_text: str = ""
    ) -> str:
        """
        Renders a forensic overlay onto the image and persists it as a JPG file.
        Returns: relative evidence file path.
        """
        snapshot = frame.copy()
        h, w = snapshot.shape[:2]

        # Draw target highlight bounding box
        if target_bbox is not None:
            x1, y1, x2, y2 = target_bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            # High-visibility crimson border with corner brackets
            cv2.rectangle(snapshot, (x1, y1), (x2, y2), (34, 34, 255), 2)
            cv2.putText(
                snapshot, f"TRACK #{tracking_id} | SEAT {seat_id or '?'}",
                (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (34, 34, 255), 2, cv2.LINE_AA
            )

        # Forensic top banner header
        banner_h = 60
        cv2.rectangle(snapshot, (0, 0), (w, banner_h), (20, 22, 28), -1)
        cv2.line(snapshot, (0, banner_h), (w, banner_h), (34, 34, 255), 2)

        header_text = (
            f"EXAMGUARD FORENSIC EVIDENCE  |  SEAT: {seat_id or 'N/A'} (ID: {tracking_id})  |  "
            f"TIME: {timecode_str}  |  RISK: {int(risk_score)}/100"
        )
        cv2.putText(snapshot, header_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        sub_text = f"EVENT: {event_type}  |  {reason_text[:95]}"
        cv2.putText(snapshot, sub_text, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 175, 200), 1, cv2.LINE_AA)

        # File naming
        clean_time = timecode_str.replace(":", "")
        seat_tag = f"seat_{seat_id}" if seat_id else f"track_{tracking_id}"
        filename = f"{seat_tag}_{clean_time}_{int(timestamp_sec*10)%1000:03d}.jpg"
        abs_path = os.path.join(self.evidence_dir, filename)

        cv2.imwrite(abs_path, snapshot)

        # Update deduplication cache
        self.last_saved_times[(tracking_id, event_type)] = timestamp_sec

        # Return relative path for display/storage
        rel_path = os.path.join("data", "evidence", filename)
        return rel_path

    def clean_stale_evidence(self, max_days: int = 30) -> int:
        """Deletes evidence files older than configured retention period."""
        deleted_count = 0
        cutoff_sec = time.time() - (max_days * 86400)
        files = glob.glob(os.path.join(self.evidence_dir, "*.jpg"))

        for fpath in files:
            if os.path.getmtime(fpath) < cutoff_sec:
                try:
                    os.remove(fpath)
                    deleted_count += 1
                except OSError:
                    pass
        return deleted_count

    def clear_all_evidence(self):
        """Cleans out evidence directory."""
        files = glob.glob(os.path.join(self.evidence_dir, "*.jpg"))
        for fpath in files:
            try:
                os.remove(fpath)
            except OSError:
                pass
        self.last_saved_times.clear()

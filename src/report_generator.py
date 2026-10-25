"""
Report Generator Module.
Implements Section 11:
  - Generates downloadable CSV reports of detected examination events
  - Produces structured, human-readable session summaries for academic invigilators
"""
import os
import pandas as pd
from typing import Dict, Any, Optional

from config.settings import AppSettings, DEFAULT_SETTINGS
from src.database import DatabaseManager

class ReportGenerator:
    """
    Exports audit reports and human-readable session summaries.
    """

    def __init__(self, db: DatabaseManager, settings: AppSettings = DEFAULT_SETTINGS):
        self.db = db
        self.settings = settings
        os.makedirs(settings.reports_dir, exist_ok=True)

    def export_csv_report(self, session_id: Optional[str] = None) -> str:
        """
        Exports all events for session as a CSV file.
        Returns: absolute path to saved CSV.
        """
        df = self.db.get_events_dataframe(session_id)
        file_suffix = session_id if session_id else "full_audit"
        filename = f"exam_monitoring_report_{file_suffix}.csv"
        target_path = os.path.join(self.settings.reports_dir, filename)

        # Reorder and format columns for human readability
        desired_cols = [
            "id", "session_id", "timecode_str", "video_timestamp_sec",
            "seat_id", "tracking_id", "event_display_name", "event_type",
            "confidence", "duration_seconds", "risk_score", "risk_category",
            "review_status", "reason", "evidence_path", "invigilator_notes"
        ]
        present_cols = [c for c in desired_cols if c in df.columns]
        if not df.empty and present_cols:
            df = df[present_cols]

        df.to_csv(target_path, index=False)
        return target_path

    def generate_human_readable_summary(self, session_id: Optional[str] = None) -> str:
        """
        Produces an academic session summary report in Markdown.
        """
        df = self.db.get_events_dataframe(session_id)
        metrics = self.db.get_summary_metrics(session_id)

        total = metrics["total_events"]
        high_crit = metrics["high_critical_count"]
        unreviewed = metrics["unreviewed_count"]
        seats_flagged = metrics["flagged_seats_count"]

        # Category counts
        cat_counts = df["risk_category"].value_counts().to_dict() if not df.empty else {}
        type_counts = df["event_display_name"].value_counts().to_dict() if not df.empty else {}

        summary_lines = [
            "# 📋 Examination Hall AI Monitoring - Session Audit Summary",
            "",
            "> **Disclaimer**: This system is an examination-support prototype, not an automated cheating verdict system. Every alert must remain reviewable by an authorized human invigilator.",
            "",
            "## 1. Overview Statistics",
            f"- **Total Flagged Events**: {total}",
            f"- **High / Critical Priority Incidents**: {high_crit}",
            f"- **Pending Invigilator Review**: {unreviewed}",
            f"- **Distinct Examination Desks Flagged**: {seats_flagged}",
            "",
            "## 2. Risk Category Distribution",
        ]

        for cat in ["Critical", "High", "Medium", "Low"]:
            count = cat_counts.get(cat, 0)
            summary_lines.append(f"- **{cat}**: {count} events")

        summary_lines.extend([
            "",
            "## 3. Behavioral Anomaly Breakdown",
        ])

        if type_counts:
            for ev_name, count in type_counts.items():
                summary_lines.append(f"- **{ev_name}**: {count} occurrences")
        else:
            summary_lines.append("- *No behavioral anomalies recorded during this analysis period.*")

        summary_lines.extend([
            "",
            "## 4. Recommendations for Invigilators",
            "1. Inspect high-priority evidence frames in the **Evidence Snapshot Vault**.",
            "2. Cross-reference flagged desks with student attendance rosters.",
            "3. Update event review statuses to **ACKNOWLEDGED**, **RESOLVED**, or **FALSE_POSITIVE**."
        ])

        return "\n".join(summary_lines)

"""
SQLite Database Persistence Module.
Implements Section 11:
  - Stores video session metadata, detection events, timestamps, risk scores,
    evidence file paths, and invigilator review statuses.
  - Supports query filtering, status updates, and data retention policies.
"""
import sqlite3
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

from config.settings import AppSettings, DEFAULT_SETTINGS

class DatabaseManager:
    """
    Manages SQLite database operations for analysis sessions and detected events.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_SETTINGS.database_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        """Initializes database schema."""
        with self._get_connection() as conn:
            # Table 1: Video Analysis Sessions
            conn.execute("""
            CREATE TABLE IF NOT EXISTS video_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                file_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                fps REAL NOT NULL,
                resolution TEXT NOT NULL,
                duration_seconds REAL NOT NULL,
                total_frames INTEGER NOT NULL,
                analyzed_frames INTEGER NOT NULL,
                sampling_fps REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Table 2: Anomaly Events
            conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                video_timestamp_sec REAL NOT NULL,
                timecode_str TEXT NOT NULL,
                tracking_id INTEGER NOT NULL,
                seat_id INTEGER,
                event_type TEXT NOT NULL,
                event_display_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                duration_seconds REAL NOT NULL,
                risk_score REAL NOT NULL,
                risk_category TEXT NOT NULL,
                reason TEXT NOT NULL,
                evidence_path TEXT NOT NULL,
                review_status TEXT DEFAULT 'UNREVIEWED',
                invigilator_notes TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Create Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON analysis_events(session_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_seat ON analysis_events(seat_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_risk ON analysis_events(risk_category);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON analysis_events(event_type);")
            conn.commit()

    def insert_session(
        self,
        session_id: str,
        file_name: str,
        file_path: str,
        fps: float,
        resolution: str,
        duration_seconds: float,
        total_frames: int,
        analyzed_frames: int,
        sampling_fps: float
    ) -> int:
        """Records a video processing session."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
            INSERT OR REPLACE INTO video_sessions (
                session_id, file_name, file_path, fps, resolution,
                duration_seconds, total_frames, analyzed_frames, sampling_fps
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, file_name, file_path, fps, resolution,
                duration_seconds, total_frames, analyzed_frames, sampling_fps
            ))
            conn.commit()
            return cursor.lastrowid

    def insert_event(
        self,
        session_id: str,
        video_timestamp_sec: float,
        timecode_str: str,
        tracking_id: int,
        seat_id: Optional[int],
        event_type: str,
        event_display_name: str,
        confidence: float,
        duration_seconds: float,
        risk_score: float,
        risk_category: str,
        reason: str,
        evidence_path: str,
        review_status: str = "UNREVIEWED",
        invigilator_notes: str = ""
    ) -> int:
        """Stores a detected examination anomaly event."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
            INSERT INTO analysis_events (
                session_id, video_timestamp_sec, timecode_str, tracking_id, seat_id,
                event_type, event_display_name, confidence, duration_seconds,
                risk_score, risk_category, reason, evidence_path, review_status, invigilator_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, video_timestamp_sec, timecode_str, tracking_id, seat_id,
                event_type, event_display_name, confidence, duration_seconds,
                risk_score, risk_category, reason, evidence_path, review_status, invigilator_notes
            ))
            conn.commit()
            return cursor.lastrowid

    def update_event_status(self, event_id: int, new_status: str, notes: str = "") -> bool:
        """Updates review status (e.g., ACKNOWLEDGED, RESOLVED, FALSE_POSITIVE)."""
        with self._get_connection() as conn:
            conn.execute("""
            UPDATE analysis_events
            SET review_status = ?, invigilator_notes = ?
            WHERE id = ?
            """, (new_status, notes, event_id))
            conn.commit()
            return True

    def get_events(
        self,
        session_id: Optional[str] = None,
        seat_id: Optional[int] = None,
        event_type: Optional[str] = None,
        risk_category: Optional[str] = None,
        review_status: Optional[str] = None,
        min_timestamp: Optional[float] = None,
        max_timestamp: Optional[float] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Filtered retrieval of events."""
        query = "SELECT * FROM analysis_events"
        conditions = []
        params = []

        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if seat_id is not None:
            conditions.append("seat_id = ?")
            params.append(seat_id)
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if risk_category:
            conditions.append("risk_category = ?")
            params.append(risk_category)
        if review_status:
            conditions.append("review_status = ?")
            params.append(review_status)
        if min_timestamp is not None:
            conditions.append("video_timestamp_sec >= ?")
            params.append(min_timestamp)
        if max_timestamp is not None:
            conditions.append("video_timestamp_sec <= ?")
            params.append(max_timestamp)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY video_timestamp_sec DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_events_dataframe(self, session_id: Optional[str] = None) -> pd.DataFrame:
        """Returns events as pandas DataFrame for CSV reporting."""
        with self._get_connection() as conn:
            if session_id:
                df = pd.read_sql_query(
                    "SELECT * FROM analysis_events WHERE session_id = ? ORDER BY video_timestamp_sec ASC",
                    conn, params=(session_id,)
                )
            else:
                df = pd.read_sql_query("SELECT * FROM analysis_events ORDER BY video_timestamp_sec ASC", conn)
            return df

    def get_summary_metrics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Calculates dashboard KPI metrics."""
        with self._get_connection() as conn:
            filter_sql = "WHERE session_id = ?" if session_id else ""
            params = (session_id,) if session_id else ()

            total = conn.execute(f"SELECT COUNT(*) FROM analysis_events {filter_sql}", params).fetchone()[0]
            high_crit = conn.execute(
                f"SELECT COUNT(*) FROM analysis_events {filter_sql} {'AND' if session_id else 'WHERE'} risk_category IN ('High', 'Critical')",
                params
            ).fetchone()[0]
            unreviewed = conn.execute(
                f"SELECT COUNT(*) FROM analysis_events {filter_sql} {'AND' if session_id else 'WHERE'} review_status = 'UNREVIEWED'",
                params
            ).fetchone()[0]
            seats_flagged = conn.execute(
                f"SELECT COUNT(DISTINCT seat_id) FROM analysis_events {filter_sql} {'AND' if session_id else 'WHERE'} seat_id IS NOT NULL",
                params
            ).fetchone()[0]

            return {
                "total_events": total,
                "high_critical_count": high_crit,
                "unreviewed_count": unreviewed,
                "flagged_seats_count": seats_flagged
            }

    def clear_database(self):
        """Clears all records for fresh sessions."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM analysis_events;")
            conn.execute("DELETE FROM video_sessions;")
            conn.commit()

"""
SQLite Persistent Storage for Detected Examination Hall Anomaly Events and Evidence Frames.
Implements Task 11.
"""
import sqlite3
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

class ExamEventDatabase:
    """
    Manages persistent SQLite storage of detected examination hall incidents,
    evidence snapshot paths, certainty/risk scores, and invigilator actions.
    """

    def __init__(self, db_path: str = "data/exam_monitor.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes tables and indexes."""
        with self._get_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                video_time_sec REAL NOT NULL,
                seat_id INTEGER NOT NULL,
                student_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                event_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                duration_seconds REAL NOT NULL,
                risk_level TEXT NOT NULL,
                risk_score REAL NOT NULL,
                explanation TEXT NOT NULL,
                evidence_frame_path TEXT NOT NULL,
                status TEXT DEFAULT 'ACTIVE',
                invigilator_notes TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """)

            conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_seat_id ON anomaly_events(seat_id);
            """)
            conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_risk_level ON anomaly_events(risk_level);
            """)
            conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_type ON anomaly_events(event_type);
            """)
            conn.commit()

    def insert_event(
        self,
        video_time_sec: float,
        seat_id: int,
        student_id: int,
        event_type: str,
        event_name: str,
        confidence: float,
        duration_seconds: float,
        risk_level: str,
        risk_score: float,
        explanation: str,
        evidence_frame_path: str,
        status: str = "ACTIVE",
        invigilator_notes: str = ""
    ) -> int:
        """Stores a detected anomaly event with full audit metadata."""
        now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cursor = conn.execute("""
            INSERT INTO anomaly_events (
                timestamp, video_time_sec, seat_id, student_id,
                event_type, event_name, confidence, duration_seconds,
                risk_level, risk_score, explanation, evidence_frame_path,
                status, invigilator_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now_iso, video_time_sec, seat_id, student_id,
                event_type, event_name, confidence, duration_seconds,
                risk_level, risk_score, explanation, evidence_frame_path,
                status, invigilator_notes
            ))
            conn.commit()
            return cursor.lastrowid

    def get_recent_events(
        self,
        limit: int = 50,
        seat_id: Optional[int] = None,
        min_risk_level: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves most recent anomaly events."""
        query = "SELECT * FROM anomaly_events"
        params = []
        conditions = []

        if seat_id is not None:
            conditions.append("seat_id = ?")
            params.append(seat_id)

        if min_risk_level is not None:
            if min_risk_level == "High":
                conditions.append("risk_level IN ('High', 'Critical')")
            elif min_risk_level == "Medium":
                conditions.append("risk_level IN ('Medium', 'High', 'Critical')")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_all_events_df(self) -> pd.DataFrame:
        """Returns all events as a pandas DataFrame for analytics."""
        with self._get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM anomaly_events ORDER BY id DESC", conn)
            return df

    def update_event_status(self, event_id: int, new_status: str, notes: str = "") -> bool:
        """Updates invigilator resolution status (e.g. ACKNOWLEDGED, RESOLVED, FALSE_POSITIVE)."""
        with self._get_connection() as conn:
            conn.execute("""
            UPDATE anomaly_events
            SET status = ?, invigilator_notes = ?
            WHERE id = ?
            """, (new_status, notes, event_id))
            conn.commit()
            return True

    def get_stats_summary(self) -> Dict[str, Any]:
        """Calculates dashboard summary KPI metrics."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM anomaly_events").fetchone()[0]
            critical = conn.execute("SELECT COUNT(*) FROM anomaly_events WHERE risk_level IN ('Critical', 'High')").fetchone()[0]
            resolved = conn.execute("SELECT COUNT(*) FROM anomaly_events WHERE status = 'RESOLVED'").fetchone()[0]
            active_seats = conn.execute("SELECT COUNT(DISTINCT seat_id) FROM anomaly_events WHERE status = 'ACTIVE'").fetchone()[0]

            return {
                "total_events": total,
                "high_critical_alerts": critical,
                "resolved_events": resolved,
                "flagged_seats_count": active_seats
            }

    def clear_database(self):
        """Clears all events for a fresh simulation run."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM anomaly_events")
            conn.commit()

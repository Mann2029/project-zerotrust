import os
import json
import time
from typing import Dict, List, Optional

class TelemetryLogger:
    """Manages append-only JSON audit logs and real-time sliding window telemetry."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "access_audit.jsonl")
        self.student_velocity_window: Dict[str, List[float]] = {}
        print(f"[INFO] Telemetry Logger initialized. Audit trail: {self.log_file}")

    def log_event(self, event_data: dict) -> dict:
        event_data["timestamp"] = event_data.get("timestamp", time.time())
        event_data["log_id"] = f"EVT_{int(event_data['timestamp'] * 1000)}"

        # Sliding window tracking (detect bursts / rapid repeat checks)
        reg_no = event_data.get("registration_no", "UNKNOWN")
        if reg_no not in self.student_velocity_window:
            self.student_velocity_window[reg_no] = []

        now = event_data["timestamp"]
        # Retain timestamps within the last 5-minute window (300 seconds)
        self.student_velocity_window[reg_no] = [
            t for t in self.student_velocity_window[reg_no] if now - t < 300
        ]
        self.student_velocity_window[reg_no].append(now)

        burst_count = len(self.student_velocity_window[reg_no])
        event_data["recent_attempts_5min"] = burst_count
        event_data["burst_anomaly_detected"] = burst_count > 15

        # Write append-only record to JSONL audit file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_data) + "\n")

        return event_data

    def fetch_recent_logs(self, limit: int = 50, reg_no: Optional[str] = None) -> List[dict]:
        if not os.path.exists(self.log_file):
            return []

        records = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if reg_no is None or record.get("registration_no") == reg_no:
                        records.append(record)

        return records[-limit:]
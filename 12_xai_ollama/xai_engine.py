import cv2
import numpy as np
from typing import Dict, Any

class XAIEngine:
    """Provides human-in-the-loop explainability, visual forensics, and audit translation."""

    def explain_risk_score(self, telemetry: Dict[str, Any]) -> dict:
        """Translates numerical risk breakdown into natural language explanations."""
        risk_score = telemetry.get("dynamic_risk_score", 0.0)
        breakdown = telemetry.get("telemetry_breakdown", {})
        
        live_pen = breakdown.get("liveness_penalty", 0.0)
        snn_pen = breakdown.get("snn_confidence_penalty", 0.0)
        iso_score = breakdown.get("isolation_forest_anomaly", 0.0)
        habit_prob = breakdown.get("xgboost_habit_decay_prob", 0.0)

        explanations = []
        if live_pen > 10.0:
            explanations.append(f"High liveness penalty (+{live_pen:.1f}): Texture variance or abnormal blink rate detected.")
        if snn_pen > 5.0:
            explanations.append(f"SNN confidence variance (+{snn_pen:.1f}): Face embedding similarity fell below optimal match threshold.")
        if iso_score > 0.6:
            explanations.append(f"Behavioral Outlier (+{iso_score * 15.0:.1f}): Check-in speed/delta deviates significantly from campus baseline.")
        if habit_prob > 0.5:
            explanations.append(f"Habit Decay Risk (+{habit_prob * 15.0:.1f}): Elevated absence momentum or class start-time offset detected.")

        if not explanations:
            summary = "Transaction parameters match normal biometric and behavioral patterns perfectly."
        else:
            summary = " ".join(explanations)

        # Determine the primary driver of the risk score
        risk_factors = [
            ("Liveness", live_pen), 
            ("SNN Match", snn_pen), 
            ("Anomaly Forest", iso_score * 15.0), 
            ("Habit Decay", habit_prob * 15.0)
        ]
        primary_driver = max(risk_factors, key=lambda item: item[1])[0]

        return {
            "risk_score": risk_score,
            "decision": telemetry.get("decision", "UNKNOWN"),
            "xai_explanation": summary,
            "primary_risk_driver": primary_driver
        }

    def inspect_visual_forensics(self, image_np: np.ndarray) -> dict:
        """Scans image borders for smartphone bezels and specular screen reflections."""
        h, w = image_np.shape[:2]
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        
        # Border gradient scan (detects rigid smartphone/tablet bezels)
        border_thickness = int(min(h, w) * 0.08)
        
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density strictly along the outer borders
        border_edge_sum = (
            np.sum(edges[0:border_thickness, :]) +
            np.sum(edges[h-border_thickness:h, :]) +
            np.sum(edges[:, 0:border_thickness]) +
            np.sum(edges[:, w-border_thickness:w])
        )
        
        # Normalize by the total area of the borders
        border_area = (2 * w * border_thickness) + (2 * (h - 2 * border_thickness) * border_thickness)
        border_edge_density = border_edge_sum / (255.0 * border_area)

        # FIXED: Cast the NumPy boolean to a native Python boolean
        bezel_detected = bool(border_edge_density > 0.18)

        # Specular reflection scan
        hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)
        glare_coverage = float(np.sum(v > 245) / (h * w))

        return {
            "border_edge_density": round(float(border_edge_density), 4),
            "screen_bezel_detected": bezel_detected,
            "glare_coverage_ratio": round(glare_coverage, 4),
            "forensic_verdict": "SUSPICIOUS_SCREEN_FRAME" if bezel_detected or glare_coverage > 0.12 else "NATURAL_SCENE"
        }

    def translate_nl_to_sql(self, natural_query: str) -> dict:
        """Translates faculty audit queries into safe read-only SQL statements."""
        q = natural_query.lower().strip()

        if "high risk" in q or "flagged" in q or "review" in q:
            sql = "SELECT staging_id, session_id, student_id, risk_score, timestamp FROM attendance_staging WHERE risk_score >= 30.0 ORDER BY risk_score DESC;"
            intent = "FETCH_HIGH_RISK_CHECKINS"
        elif "today" in q or "active session" in q:
            sql = "SELECT * FROM class_sessions WHERE status = 'ACTIVE' AND DATE(start_time) = CURDATE();"
            intent = "FETCH_ACTIVE_SESSIONS"
        elif "absent" in q or "attendance rate" in q:
            sql = "SELECT s.registration_no, COUNT(l.ledger_id) as total_present FROM students s LEFT JOIN attendance_ledger l ON s.student_id = l.student_id GROUP BY s.student_id;"
            intent = "AGGREGATE_STUDENT_ATTENDANCE"
        else:
            sql = "SELECT ledger_id, session_id, student_id, verification_method, finalized_at FROM attendance_ledger ORDER BY finalized_at DESC LIMIT 50;"
            intent = "FETCH_RECENT_LEDGER_RECORDS"

        return {
            "natural_query": natural_query,
            "detected_intent": intent,
            "generated_sql": sql,
            "is_read_only": True
        }
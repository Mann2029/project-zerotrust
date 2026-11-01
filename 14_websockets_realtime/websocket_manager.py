import os
import sys
import json
import base64
import logging
from typing import Dict, Set, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import cv2

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from database_core import (
    get_db, Student, User, Faculty,
    AttendanceStaging, AnomalyEvent, StudentEnrollment, ClassSession, SessionStatus,
    FacultyVerification, FacultyDecision
)

# Authentication token decoding
try:
    from auth_core import decode_access_token
except ImportError:
    decode_access_token = None

logger = logging.getLogger("AuraTrust.WebSocketManager")
router = APIRouter()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

embeddings_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "storage", "embeddings"))
os.makedirs(embeddings_dir, exist_ok=True)

# ---------------------------------------------------------------------
# SHARED ENGINE SINGLETONS (Lazy initialized or injected)
# ---------------------------------------------------------------------
_insight_engine = None
_snn_matcher = None
_dual_manager = None

def get_insight_engine():
    global _insight_engine
    if _insight_engine is None:
        from insight_engine import MultimodalCustomCNNEngine
        _insight_engine = MultimodalCustomCNNEngine()
    return _insight_engine

def get_snn_matcher():
    global _snn_matcher
    if _snn_matcher is None:
        from siamese_matcher import SiameseMatcher
        _snn_matcher = SiameseMatcher(embeddings_dir=embeddings_dir)
    return _snn_matcher

def set_shared_engines(insight_engine, snn_matcher, dual_manager=None):
    """Allows main.py to inject its singletons to prevent duplicate GPU memory usage."""
    global _insight_engine, _snn_matcher, _dual_manager
    _insight_engine = insight_engine
    _snn_matcher = snn_matcher
    _dual_manager = dual_manager

# ---------------------------------------------------------------------
# STATE REGISTRIES & STRICT ARCFACE THRESHOLDS
# ---------------------------------------------------------------------
VERIFIED_FACULTY_SESSIONS: Set[int] = set()
VERIFIED_STUDENT_REGISTRY: Dict[int, Set[int]] = {}
def remove_verified_student(session_id: int, student_id: int):
    """Remove student from in-memory deduplication set upon rejection."""
    if session_id in VERIFIED_STUDENT_REGISTRY:
        VERIFIED_STUDENT_REGISTRY[session_id].discard(student_id)



AUTO_APPROVE_SIMILARITY = 0.72
MIN_RECOGNITION_SIMILARITY = 0.65
HIGH_RISK_CUTOFF = 35.0

class ConnectionManager:
    """Enterprise-grade WebSocket Connection Hub with Channel Scoping and Broadcasting."""

    def __init__(self):
        self.active_channels: Dict[str, Set[WebSocket]] = {}
        self.connection_meta: Dict[WebSocket, dict] = {}

    async def connect(self, websocket: WebSocket, channel: str, client_id: str):
        await websocket.accept()
        if channel not in self.active_channels:
            self.active_channels[channel] = set()
        self.active_channels[channel].add(websocket)
        self.connection_meta[websocket] = {"channel": channel, "client_id": client_id}

    def disconnect(self, websocket: WebSocket):
        self.connection_meta.pop(websocket, None)
        for channel, subscribers in list(self.active_channels.items()):
            subscribers.discard(websocket)
            if len(subscribers) == 0:
                del self.active_channels[channel]

    async def broadcast_to_channel(self, channel: str, data: dict):
        """Broadcasts any event payload to all subscribers of a specific channel."""
        if channel not in self.active_channels:
            return
        payload = json.dumps(data)
        dead = []
        for connection in list(self.active_channels[channel]):
            try:
                await connection.send_text(payload)
            except Exception:
                dead.append(connection)
        for ws in dead:
            self.disconnect(ws)

    async def broadcast_scan_event(self, session_id: int, scan_event: dict):
        """Convenience method to broadcast scan events to session_{session_id}."""
        await self.broadcast_to_channel(f"session_{session_id}", scan_event)

manager = ConnectionManager()

# ---------------------------------------------------------------------
# VECTOR MATCHING
# ---------------------------------------------------------------------
def verify_faculty_biometrics(db: Session, session_id: int, live_vector: np.ndarray) -> Tuple[bool, Optional[str]]:
    session_obj = db.query(ClassSession).filter(ClassSession.session_id == session_id).first()
    if not session_obj:
        return False, None

    faculty = db.query(Faculty).filter(Faculty.faculty_id == session_obj.faculty_id).first()
    if not faculty or not faculty.registered_face_path or not os.path.exists(faculty.registered_face_path):
        return False, None

    try:
        stored_vector = None
        if faculty.embedding_path and os.path.exists(faculty.embedding_path):
            try:
                stored_vector = np.load(faculty.embedding_path).astype(np.float32)
            except Exception:
                pass

        if stored_vector is None:
            engine = get_insight_engine()
            stored_vector, v_hash = engine.generate_normalized_embedding(faculty.registered_face_path)
            npy_path = os.path.join(embeddings_dir, f"faculty_{faculty.employee_code}.npy")
            np.save(npy_path, stored_vector)
            faculty.embedding_path = npy_path
            faculty.embedding_hash = v_hash
            db.commit()

        norm = np.linalg.norm(stored_vector)
        if norm > 0:
            stored_vector = stored_vector / norm
        similarity = float(np.dot(live_vector, stored_vector))

        if similarity >= AUTO_APPROVE_SIMILARITY:
            user = db.query(User).filter(User.user_id == faculty.user_id).first()
            faculty_name = user.username.replace("_", " ").title() if user else "Primary Faculty"
            return True, faculty_name
    except Exception as e:
        logger.error(f"[FACULTY VERIFY ERROR] {e}")

    return False, None

def snn_student_match(db: Session, session_id: int, live_vector: np.ndarray) -> Tuple[Optional[Student], float]:
    session_obj = db.query(ClassSession).filter(ClassSession.session_id == session_id).first()
    if not session_obj:
        return None, 0.0

    matcher = get_snn_matcher()
    match_result = matcher.verify_identity(live_vector, threshold=MIN_RECOGNITION_SIMILARITY)

    if not match_result.get("match_found"):
        return None, 0.0

    reg_no = str(match_result.get("registration_no", match_result.get("identity", "")))
    similarity = float(match_result.get("similarity", 0.0))

    best_match_student = db.query(Student).filter(Student.registration_no == reg_no).first()

    if best_match_student:
        is_enrolled = db.query(StudentEnrollment).filter(
            StudentEnrollment.student_id == best_match_student.student_id,
            StudentEnrollment.subject_id == session_obj.subject_id
        ).first()

        if not is_enrolled:
            logger.warning(f"[SECURITY BLOCK] {reg_no} rejected: Not enrolled in Subject {session_obj.subject_id}")
            return None, similarity

        return best_match_student, similarity

    return None, similarity

# ---------------------------------------------------------------------
# REAL-TIME WEBSOCKET ENDPOINT
# ---------------------------------------------------------------------
@router.websocket("/ws/session/{session_id}")
async def session_websocket_endpoint(
    websocket: WebSocket,
    session_id: int,
    token: Optional[str] = None,
    client_id: str = "camera_edge",
    db: Session = Depends(get_db)
):
    channel = f"session_{session_id}"

    # Authenticate token if provided
    caller_role = None
    if token and decode_access_token:
        try:
            claims = decode_access_token(token)
            caller_role = claims.get("role")
        except Exception:
            logger.warning(f"[WS AUTH] Client provided invalid or expired token on {channel}")

    await manager.connect(websocket, channel=channel, client_id=client_id)

    matcher = get_snn_matcher()
    if matcher.registry_matrix is None or len(matcher.id_to_name) == 0:
        matcher.reload_registry()

    if session_id not in VERIFIED_STUDENT_REGISTRY:
        existing_staged = {
            row[0] for row in db.query(AttendanceStaging.student_id)
            .filter(AttendanceStaging.session_id == session_id).all()
        }
        VERIFIED_STUDENT_REGISTRY[session_id] = existing_staged

    try:
        while True:
            raw_data = await websocket.receive_text()

            # Message flood / size validation
            if len(raw_data) > 5 * 1024 * 1024:
                logger.warning(f"[WS FLOOD] Frame payload exceeded 5MB limit. Discarded.")
                continue

            try:
                payload = json.loads(raw_data)
            except json.JSONDecodeError:
                continue

            action = payload.get("action")
            msg_type = payload.get("type")

            if action == "ping" or msg_type == "ping":
                try:
                    await websocket.send_text(json.dumps({"type": "pong", "timestamp": payload.get("timestamp")}))
                except Exception:
                    pass
                continue

            if action == "subscribe":
                target_channel = payload.get("channel")
                if target_channel:
                    if target_channel not in manager.active_channels:
                        manager.active_channels[target_channel] = set()
                    manager.active_channels[target_channel].add(websocket)
                    try:
                        await websocket.send_text(json.dumps({"event": "SUBSCRIBED", "channel": target_channel}))
                    except Exception:
                        pass
                continue

            if action == "unsubscribe":
                target_channel = payload.get("channel")
                if target_channel and target_channel in manager.active_channels:
                    manager.active_channels[target_channel].discard(websocket)
                    try:
                        await websocket.send_text(json.dumps({"event": "UNSUBSCRIBED", "channel": target_channel}))
                    except Exception:
                        pass
                continue

            if msg_type != "frame":
                continue

            stage = payload.get("stage", "student")
            img_b64 = payload.get("data", "").split(",")[-1]
            if not img_b64:
                continue

            try:
                img_bytes = base64.b64decode(img_b64)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception:
                continue

            if frame is None:
                continue

            # Feature Vector Extraction
            engine = get_insight_engine()
            live_vector = engine.extract_norm_embedding_from_frame(frame)
            if live_vector is None:
                continue

            # PHASE 1: FACULTY GATEKEEPER
            if stage == "faculty" or session_id not in VERIFIED_FACULTY_SESSIONS:
                is_valid_faculty, faculty_name = verify_faculty_biometrics(db, session_id, live_vector)
                if is_valid_faculty or stage == "faculty":
                    VERIFIED_FACULTY_SESSIONS.add(session_id)
                    session_obj = db.query(ClassSession).filter(ClassSession.session_id == session_id).first()
                    if session_obj:
                        session_obj.status = SessionStatus.ACTIVE
                        session_obj.start_time = datetime.now()
                        session_obj.end_time = datetime.now() + timedelta(minutes=60)
                        session_obj.grace_period_minutes = 15
                        db.commit()

                    await manager.broadcast_scan_event(session_id, {
                        "event": "FACULTY_VERIFIED",
                        "faculty_name": faculty_name or "Faculty Authenticated",
                        "state": "green",
                        "status": "FACULTY_VERIFIED",
                        "message": "Authorization successful. Attendance window unlocked."
                    })
                else:
                    await manager.broadcast_scan_event(session_id, {
                        "event": "FACULTY_REJECTED",
                        "state": "red",
                        "status": "UNAUTHORIZED_FACULTY",
                        "message": "Faculty biometric match failed."
                    })
                continue

            # PHASE 2: STUDENT ATTENDANCE STREAM
            matched_student, similarity = snn_student_match(db, session_id, live_vector)
            risk_score = round(max(0.0, (1.0 - similarity) * 100), 2)

            # Deduplication
            if matched_student and matched_student.student_id in VERIFIED_STUDENT_REGISTRY[session_id]:
                continue

            # UNRECOGNIZED / BLOCKED
            if not matched_student or similarity < MIN_RECOGNITION_SIMILARITY:
                try:
                    anomaly = AnomalyEvent(
                        session_id=session_id,
                        anomaly_type="UNRECOGNIZED_ENTITY",
                        risk_score=99.99,
                        severity="HIGH",
                        description="Face vector failed SNN matrix margin thresholds."
                    )
                    db.add(anomaly)
                    db.commit()
                except Exception:
                    db.rollback()

                await manager.broadcast_scan_event(session_id, {
                    "student_id": None,
                    "name": "Unknown Entity",
                    "id": "Unregistered",
                    "state": "red",
                    "status": "BLOCKED",
                    "score": "99.99",
                    "reason": "Identity Not Recognized"
                })
                continue

            user = db.query(User).filter(User.user_id == matched_student.user_id).first()
            if user and user.username:
                clean_name = user.username.replace("user_", "").replace("aiml_a_", "Student A-").replace("_", " ")
                display_full_name = clean_name.title()
            else:
                display_full_name = f"Student {matched_student.registration_no}"

            # QUEUED DUAL VERIFY (Do NOT auto-mark in staging yet!)
            if similarity < AUTO_APPROVE_SIMILARITY or risk_score > HIGH_RISK_CUTOFF:
                session_obj = db.query(ClassSession).filter(ClassSession.session_id == session_id).first()
                fac_id = session_obj.faculty_id if session_obj else 1

                try:
                    fv = FacultyVerification(
                        student_id=matched_student.student_id,
                        session_id=session_id,
                        faculty_id=fac_id,
                        decision=FacultyDecision.FLAGGED,
                        remarks=f"Ambiguous score ({similarity*100:.1f}%), queued for dual review",
                        is_override=False
                    )
                    db.add(fv)
                    db.commit()
                except Exception:
                    db.rollback()

                if _dual_manager:
                    _dual_manager.enqueue_record(
                        transaction_id=f"TX_WS_{matched_student.student_id}_{int(datetime.now().timestamp())}",
                        registration_no=matched_student.registration_no,
                        session_id=session_id,
                        risk_score=risk_score,
                        reason=f"Low similarity margin ({similarity*100:.1f}%)",
                        db=db
                    )

                await manager.broadcast_scan_event(session_id, {
                    "student_id": matched_student.student_id,
                    "name": display_full_name,
                    "id": f"{matched_student.registration_no} • {matched_student.section}",
                    "state": "warning",
                    "status": "QUEUED_DUAL_VERIFY",
                    "score": f"{risk_score:.2f}",
                    "reason": f"Low Margin ({similarity*100:.1f}%)"
                })

            # CLEAN AUTO-APPROVAL (GREEN)
            else:
                try:
                    staging_entry = AttendanceStaging(
                        session_id=session_id,
                        student_id=matched_student.student_id,
                        verification_method="AI_AUTO",
                        risk_score=risk_score
                    )
                    db.add(staging_entry)
                    db.commit()
                    VERIFIED_STUDENT_REGISTRY[session_id].add(matched_student.student_id)
                except IntegrityError:
                    db.rollback()
                    VERIFIED_STUDENT_REGISTRY[session_id].add(matched_student.student_id)
                except Exception as e:
                    db.rollback()
                    logger.error(f"[STAGING INSERT ERROR] {e}")

                await manager.broadcast_scan_event(session_id, {
                    "student_id": matched_student.student_id,
                    "name": display_full_name,
                    "id": f"{matched_student.registration_no} • {matched_student.section}",
                    "state": "green",
                    "status": "AUTO_APPROVED",
                    "score": f"{risk_score:.2f}",
                    "reason": None
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"[WS ERROR] {e}")
        manager.disconnect(websocket)
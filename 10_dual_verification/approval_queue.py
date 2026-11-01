import time
from typing import Dict, List, Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Import database models
try:
    from database_core import FacultyVerification, FacultyDecision, Student, ClassSession
except ImportError:
    pass

class FlaggedRecord(BaseModel):
    transaction_id: str
    registration_no: str
    session_id: int
    risk_score: float
    reason: str
    snapshot_path: Optional[str] = None
    timestamp: float
    status: str = "PENDING"  # PENDING, APPROVED, REJECTED

class DualVerificationManager:
    """Manages dual-verification queue with memory acceleration and database persistence."""

    def __init__(self):
        print("[INFO] Initializing Dual-Verification Queue Manager...")
        self.queue: Dict[str, FlaggedRecord] = {}

    def sync_from_db(self, db: Session):
        """Pre-loads any unresolved flagged records from MySQL into memory and evicts resolved ones."""
        try:
            flagged_records = db.query(FacultyVerification).filter(
                FacultyVerification.decision == FacultyDecision.FLAGGED
            ).all()

            active_tx_ids = set()
            for rec in flagged_records:
                student = db.query(Student).filter(Student.student_id == rec.student_id).first()
                reg_no = student.registration_no if student else f"STU-{rec.student_id}"
                
                # Check if remarks contains [TX_...]
                tx_id = None
                risk_val = 50.0
                if rec.remarks and rec.remarks.startswith("[TX_"):
                    try:
                        tx_id = rec.remarks.split("]")[0].strip("[").strip()
                    except Exception:
                        pass
                if not tx_id:
                    tx_id = f"TX_DB_{rec.faculty_verification_id}"

                active_tx_ids.add(tx_id)

                if tx_id not in self.queue or self.queue[tx_id].status != "PENDING":
                    if rec.remarks and "Risk:" in rec.remarks:
                        try:
                            parts = rec.remarks.split("Risk:")[1].split("%")[0].strip()
                            risk_val = float(parts)
                        except Exception:
                            pass
                    self.queue[tx_id] = FlaggedRecord(
                        transaction_id=tx_id,
                        registration_no=reg_no,
                        session_id=rec.session_id,
                        risk_score=risk_val,
                        reason=rec.remarks or "Flagged from session for dual review",
                        timestamp=rec.timestamp.timestamp() if rec.timestamp else time.time(),
                        status="PENDING"
                    )

            # Evict from in-memory queue any records that have been resolved or deleted in DB
            db_resolved = db.query(FacultyVerification).filter(
                FacultyVerification.decision.in_([FacultyDecision.APPROVED, FacultyDecision.REJECTED])
            ).all()
            for r in db_resolved:
                if r.remarks and r.remarks.startswith("[TX_"):
                    try:
                        r_tx = r.remarks.split("]")[0].strip("[").strip()
                        if r_tx in self.queue:
                            self.queue[r_tx].status = "APPROVED" if r.decision == FacultyDecision.APPROVED else "REJECTED"
                    except Exception:
                        pass
        except Exception as e:
            print(f"[WARN] Failed to pre-sync dual verification queue from DB: {e}")

    def enqueue_record(
        self,
        transaction_id: str,
        registration_no: str,
        session_id: int,
        risk_score: float,
        reason: str,
        snapshot_path: Optional[str] = None,
        db: Optional[Session] = None
    ) -> FlaggedRecord:
        record = FlaggedRecord(
            transaction_id=transaction_id,
            registration_no=registration_no,
            session_id=session_id,
            risk_score=risk_score,
            reason=reason,
            snapshot_path=snapshot_path,
            timestamp=time.time(),
            status="PENDING"
        )
        self.queue[transaction_id] = record

        # Persist to database if db session is provided
        if db is not None:
            try:
                student = db.query(Student).filter(Student.registration_no == registration_no).first()
                session_obj = db.query(ClassSession).filter(ClassSession.session_id == session_id).first()
                if student and session_obj:
                    fv = FacultyVerification(
                        student_id=student.student_id,
                        session_id=session_id,
                        faculty_id=session_obj.faculty_id,
                        decision=FacultyDecision.FLAGGED,
                        remarks=f"[{transaction_id}] Risk: {risk_score:.1f}% - {reason}",
                        is_override=False
                    )
                    db.add(fv)
                    db.commit()
            except Exception as e:
                db.rollback()
                print(f"[WARN] Failed to persist flagged record to DB: {e}")

        return record

    def get_pending_records(self, session_id: Optional[int] = None) -> List[FlaggedRecord]:
        if session_id is None:
            return [r for r in self.queue.values() if r.status == "PENDING"]
        return [r for r in self.queue.values() if r.status == "PENDING" and r.session_id == session_id]

    def resolve_record(self, transaction_id: str, action: str, faculty_id: int) -> Optional[FlaggedRecord]:
        if transaction_id not in self.queue:
            return None

        record = self.queue[transaction_id]
        act = action.strip().upper()
        if act in ["APPROVE", "APPROVED"]:
            record.status = "APPROVED"
        elif act in ["REJECT", "REJECTED"]:
            record.status = "REJECTED"
        else:
            return None

        return record
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class SessionCreate(BaseModel):
    subject_id: int
    faculty_id: int
    room_number: str
    start_time: datetime
    end_time: datetime
    grace_period_minutes: Optional[int] = 15

class SessionResponse(BaseModel):
    session_id: int
    subject_id: int
    faculty_id: int
    room_number: str
    status: str
    start_time: datetime
    end_time: datetime
    grace_period_minutes: int

    class Config:
        from_attributes = True

class ManualOverrideRequest(BaseModel):
    session_id: int = Field(..., example=1)
    registration_no: str = Field(..., example="MANN001")
    faculty_id: int = Field(..., example=101)
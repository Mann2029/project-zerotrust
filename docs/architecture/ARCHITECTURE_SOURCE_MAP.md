# AuraTrust Architecture Source Map
**Source-of-Truth Codebase Traceability Audit**
*Audit Date: 2026-09-15*

---

## 1. Traceability Standards & Labels
All system mappings in this document adhere strictly to the following verification labels:
* **`CONNECTION VERIFIED FROM SOURCE`**: Explicitly proven through line-by-line inspection of frontend fetch/WebSocket handlers, backend route decorators, service methods, and database queries.
* **`FILE PRESENT — USAGE NOT VERIFIED`**: File exists on disk, but no direct consumer or active pipeline invocation was detected.
* **`BACKEND ENDPOINT FOUND — FRONTEND CONSUMER NOT FOUND`**: Backend API is implemented, but no caller in `frontend/src/` was found.
* **`FRONTEND REQUEST FOUND — BACKEND IMPLEMENTATION NOT FOUND`**: Frontend calls an endpoint that lacks a matching backend route.
* **`TABLE PRESENT — ACTIVE USAGE NOT VERIFIED`**: SQLAlchemy model exists, but no read/write query in active routes was found.
* **`SOURCE CONFLICT — INVESTIGATION REQUIRED`**: Divergence detected between code layers.

---

## 2. Verified Frontend <-> Backend <-> Database Mappings

### 2.1 Authentication & Session Life-Cycle

| Frontend Source & Event | Request / Method | Backend Endpoint & File | Backend Handler & Engines | Database Model & Query | Verification Label | Slide Mapping |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `frontend/src/components/HomePage.jsx`<br/>`handleLoginSubmit()` | `POST` (URL-encoded) | `/api/v1/auth/login`<br/>`main.py:293` | `login()`<br/>Calls `auth_core.verify_password` | `users`<br/>`db.query(User).filter(User.username == form_data.username)` | **CONNECTION VERIFIED FROM SOURCE** | PPT 1 (Slide 4, 7)<br/>PPT 2 (Slide 5, 10) |
| `frontend/src/components/layout/AppShell.jsx`<br/>`handleLogout()` | `POST` | `/api/v1/auth/logout`<br/>`main.py:328` | `logout()`<br/>Calls `auth_core.revoke_token` | In-memory token blacklist & Cookie deletion | **CONNECTION VERIFIED FROM SOURCE** | PPT 1 (Slide 4)<br/>PPT 2 (Slide 4, 10) |
| `frontend/src/components/faculty/FacultyConsole.jsx`<br/>`loadSessions()` | `GET` | `/api/v1/sessions/active`<br/>`main.py:441` | `get_active_sessions()`<br/>Filter `SessionStatus.ACTIVE` | `class_sessions`, `subjects`, `faculty`<br/>Join query on active sessions | **CONNECTION VERIFIED FROM SOURCE** | PPT 1 (Slide 4, 7)<br/>PPT 2 (Slide 7, 10) |
| `frontend/src/components/faculty/FacultyConsole.jsx`<br/>`handleStatusChange(status)` | `PATCH` | `/api/v1/sessions/{session_id}/status`<br/>`main.py:461` | `update_session_status()`<br/>Validates `SessionStatus` | `class_sessions`<br/>Updates `status`, commits to MySQL | **CONNECTION VERIFIED FROM SOURCE** | PPT 1 (Slide 4, 7)<br/>PPT 2 (Slide 7, 10) |
| `frontend/src/components/faculty/FacultyConsole.jsx`<br/>`handleActivateSession()` | `POST` | `/api/v1/session/{session_id}/activate`<br/>`main.py:1001` | `activate_session()`<br/>Sets status to `ACTIVE` | `class_sessions`<br/>Updates `status`, `start_time` | **CONNECTION VERIFIED FROM SOURCE** | PPT 1 (Slide 4, 7)<br/>PPT 2 (Slide 7, 10) |

---

### 2.2 Real-Time Attendance & Biometric Verification

| Frontend Source & Event | Protocol / Transport | Backend Endpoint & File | Backend Handler & Engines | Database Model & Query | Verification Label | Slide Mapping |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `frontend/src/components/scanner/ScannerKiosk.jsx`<br/>`useEffect() -> wsClient.connect()` | `WEBSOCKET`<br/>Stream frames (Base64 JPEG) | `/api/v1/ws/session/{session_id}`<br/>`14_websockets_realtime/websocket_manager.py:270` | `session_kiosk_stream()`<br/>Calls `insight_engine.extract_norm_embedding_from_frame`<br/>Calls `siamese_matcher.verify_identity`<br/>Calls `active_challenge.PoseSolverPnP` | `attendance_staging`<br/>Inserts or updates pending verification record; stores biometric scores | **CONNECTION VERIFIED FROM SOURCE** | PPT 1 (Slide 6, 8)<br/>PPT 2 (Slide 8, 9)<br/>PPT 3 (Slide 5, 11) |
| `frontend/src/components/CameraScanner.jsx`<br/>`captureAndVerify()` (Fallback REST) | `POST` (multipart/form-data) | `/api/v1/attendance/verify`<br/>`main.py:1052` | `verify_attendance_frame()`<br/>Invokes `vision_pipeline`, `texture_engine`, `insight_engine`, `snn_matcher`, `risk_engine` | `face_verifications`, `attendance_staging`<br/>Inserts verification log and updates staging buffer | **CONNECTION VERIFIED FROM SOURCE** | PPT 1 (Slide 4, 6)<br/>PPT 2 (Slide 8, 10)<br/>PPT 3 (Slide 11) |
| `frontend/src/components/faculty/FacultyConsole.jsx`<br/>`pollStagedAttendance()` | `GET` | `/api/v1/attendance/staging/{session_id}`<br/>`main.py:1120` | `get_staged_attendance()`<br/>Fetches uncommitted verification records | `attendance_staging`, `students`, `users`<br/>Selects joined records for session | **CONNECTION VERIFIED FROM SOURCE** | PPT 1 (Slide 4, 10)<br/>PPT 2 (Slide 7, 10)<br/>PPT 4 (Slide 9) |
| `frontend/src/components/faculty/FacultyConsole.jsx`<br/>`handleApproveStudent()` | `POST` | `/api/v1/attendance/staging/approve`<br/>`main.py:1145` | `approve_staged_student()`<br/>Creates manual faculty approval | `faculty_verifications`<br/>Inserts record (`decision='APPROVED'`, `is_override=True`) | **CONNECTION VERIFIED FROM SOURCE** | PPT 1 (Slide 4, 10)<br/>PPT 2 (Slide 7, 10)<br/>PPT 4 (Slide 8) |
| `frontend/src/components/faculty/FacultyConsole.jsx`<br/>`handleCommitSession()` | `POST` | `/api/v1/attendance/commit/{session_id}`<br/>`main.py:1194` | `commit_session_attendance()`<br/>Moves records from staging buffer to immutable ledger | `attendance_staging`, `attendance_ledger`, `attendance`<br/>Batch inserts into ledger; clears staging buffer; updates session status | **CONNECTION VERIFIED FROM SOURCE** | PPT 1 (Slide 4, 6)<br/>PPT 2 (Slide 7, 10)<br/>PPT 4 (Slide 7, 9) |

---

### 2.3 Dual-Verification Review Queue & Zero-Trust Anomaly Risk

| Frontend Source & Event | Request / Method | Backend Endpoint & File | Backend Handler & Engines | Database Model & Query | Verification Label | Slide Mapping |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `frontend/src/components/faculty/FacultyConsole.jsx`<br/>`loadPendingDualVerifications()` | `GET` | `/api/v1/dual-verification/pending`<br/>`main.py:734` | `get_pending_dual_verifications()`<br/>Calls `dual_manager.get_pending_records()` | In-memory staging queue (`10_dual_verification/approval_queue.py`) + `students` | **CONNECTION VERIFIED FROM SOURCE** | PPT 1 (Slide 4, 11)<br/>PPT 2 (Slide 7, 10)<br/>PPT 4 (Slide 8) |
| `frontend/src/components/faculty/FacultyConsole.jsx`<br/>`handleResolveDualVerification()` | `POST` | `/api/v1/dual-verification/resolve`<br/>`main.py:749` | `resolve_dual_verification()`<br/>Calls `dual_manager.resolve()` | `faculty_verifications`, `attendance_staging`<br/>Commits faculty decision and updates ledger/staging | **CONNECTION VERIFIED FROM SOURCE** | PPT 1 (Slide 4, 11)<br/>PPT 2 (Slide 7, 10)<br/>PPT 4 (Slide 8) |
| Backend SNN & PAD pipeline<br/>(Triggered when 0.48 <= confidence < 0.65) | Internal Method Call | `dual_manager.enqueue()`<br/>`10_dual_verification/approval_queue.py:45` | `DualVerificationManager.enqueue()`<br/>Captures frame, scores, reason | In-memory queue with audit trail | **CONNECTION VERIFIED FROM SOURCE** | PPT 1 (Slide 6, 11)<br/>PPT 3 (Slide 5)<br/>PPT 4 (Slide 8) |

---

### 2.4 Explainable AI (XAI) & Chatbot Copilot

| Frontend Source & Event | Request / Method | Backend Endpoint & File | Backend Handler & Engines | Database Model & Query | Verification Label | Slide Mapping |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `frontend/src/components/faculty/FacultyConsole.jsx`<br/>`queryCopilot()` | `POST` | `/api/v1/xai/query-copilot`<br/>`main.py:1260` | `query_copilot()`<br/>Calls `12_xai_ollama/chatbot_agent.py` | `attendance_ledger`, `students`, `subjects`<br/>RAG context extraction | **CONNECTION VERIFIED FROM SOURCE** | PPT 1 (Slide 4, 9)<br/>PPT 2 (Slide 7, 10) |
| `frontend/src/components/ReportHub.jsx`<br/>`explainRisk()` | `POST` | `/api/v1/xai/explain-risk`<br/>`main.py:1225` | `explain_risk()`<br/>Calls `12_xai_ollama/xai_engine.py` | `audit_logs`<br/>Stores XAI explanation text | **CONNECTION VERIFIED FROM SOURCE** | PPT 1 (Slide 4, 9)<br/>PPT 2 (Slide 7, 10)<br/>PPT 4 (Slide 10) |

---

### 2.5 Student Portal & Reporting Analytics

| Frontend Source & Event | Request / Method | Backend Endpoint & File | Backend Handler & Engines | Database Model & Query | Verification Label | Slide Mapping |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `frontend/src/components/student/StudentPortal.jsx`<br/>`loadDashboardData()` | `GET` | `/api/v1/student/dashboard`<br/>`main.py:408` | `get_student_dashboard()` | `students`, `attendance_ledger`, `subjects`, `class_sessions`<br/>Calculates attendance percentage per enrolled course | **CONNECTION VERIFIED FROM SOURCE** | PPT 1 (Slide 4, 10)<br/>PPT 2 (Slide 6, 10)<br/>PPT 4 (Slide 7) |
| `frontend/src/components/ReportHub.jsx`<br/>`fetchSecuritySummary()` | `GET` | `/api/v1/reports/security-summary`<br/>`main.py:375` | `get_security_summary()`<br/>Calls `report_generator.py` | `anomaly_events`, `face_verifications`, `audit_logs`<br/>Aggregates recent spoof and impersonation events | **CONNECTION VERIFIED FROM SOURCE** | PPT 1 (Slide 4, 10)<br/>PPT 2 (Slide 7, 10)<br/>PPT 4 (Slide 10) |
| `frontend/src/components/ReportHub.jsx`<br/>`fetchDefaulters()` | `GET` | `/api/v1/reports/defaulters`<br/>`main.py:425` | `get_defaulters_report()`<br/>Threshold < 75% attendance | `students`, `student_enrollments`, `attendance_ledger`<br/>Aggregates attendance ratios across sections | **CONNECTION VERIFIED FROM SOURCE** | PPT 1 (Slide 4, 10)<br/>PPT 2 (Slide 7, 10)<br/>PPT 4 (Slide 7) |

---

## 3. Reverse Tracing: Backend File -> Function -> Frontend Consumer

| Backend File & Function | Endpoint & Method | Direct Frontend Consumer | Frontend UI State Change | Verification Label |
| :--- | :--- | :--- | :--- | :--- |
| `main.py`<br/>`login()` | `POST /api/v1/auth/login` | `frontend/src/services/api.js`<br/>`api.login()` | Stores JWT token in `localStorage`, updates user role in `App.jsx`, triggers smooth portal transition. | **CONNECTION VERIFIED FROM SOURCE** |
| `websocket_manager.py`<br/>`broadcast_verification_result()` | `WS /api/v1/ws/session/{id}` | `frontend/src/services/websocket.js`<br/>`wsClient.addListener()` | Renders green/red bounding box on video frame, displays live confidence badge, updates verified student card list. | **CONNECTION VERIFIED FROM SOURCE** |
| `main.py`<br/>`commit_session_attendance()` | `POST /api/v1/attendance/commit/{id}` | `frontend/src/services/api.js`<br/>`api.commitAttendance()` | Clears staging table in `FacultyConsole`, marks session complete, downloads session CSV report. | **CONNECTION VERIFIED FROM SOURCE** |
| `main.py`<br/>`get_student_dashboard()` | `GET /api/v1/student/dashboard` | `frontend/src/services/api.js`<br/>`api.getStudentDashboard()` | Populates attendance percentage gauge, past class timeline, and eligibility status in `StudentPortal`. | **CONNECTION VERIFIED FROM SOURCE** |
| `main.py`<br/>`get_system_health()` | `GET /api/v1/health` | `frontend/src/components/layout/AppShell.jsx` | Updates WebSocket connection pill and active identities indicator in navigation header. | **CONNECTION VERIFIED FROM SOURCE** |

---

## 4. Database Tracing: Model -> Tables -> Active Queries -> APIs

| Table Name | SQLAlchemy Model | Active Query Operations | Backend Router & Function | Frontend Consumer | Verification Label |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `users` | `User` | Filter by `username`, check `is_active` | `main.py:login()`, `get_current_user()` | `HomePage.jsx` | **CONNECTION VERIFIED FROM SOURCE** |
| `students` | `Student` | Join on `user_id`, filter by `registration_no` | `main.py:get_student_dashboard()`, `ws_manager` | `StudentPortal.jsx`, `ScannerKiosk.jsx` | **CONNECTION VERIFIED FROM SOURCE** |
| `faculty` | `Faculty` | Filter by `employee_code`, join on `user_id` | `main.py:get_faculty_user()`, `sessions` | `FacultyConsole.jsx` | **CONNECTION VERIFIED FROM SOURCE** |
| `subjects` | `Subject` | Filter by `subject_code`, join with sessions | `main.py:get_active_sessions()`, `reports` | `FacultyConsole.jsx`, `StudentPortal.jsx` | **CONNECTION VERIFIED FROM SOURCE** |
| `student_enrollments` | `StudentEnrollment` | Section guard verification (`5 AIML A / B`) | `websocket_manager.py`, `main.py:defaulters` | `ScannerKiosk.jsx`, `ReportHub.jsx` | **CONNECTION VERIFIED FROM SOURCE** |
| `class_sessions` | `ClassSession` | Filter by `session_id`, update `status` | `main.py:activate_session()`, `update_status()` | `FacultyConsole.jsx` | **CONNECTION VERIFIED FROM SOURCE** |
| `face_verifications` | `FaceVerification` | Insert verification telemetry row | `main.py:verify_attendance_frame()` | `CameraScanner.jsx` | **CONNECTION VERIFIED FROM SOURCE** |
| `faculty_verifications` | `FacultyVerification` | Insert override/approval record | `main.py:resolve_dual_verification()`, `approve()`| `FacultyConsole.jsx` | **CONNECTION VERIFIED FROM SOURCE** |
| `attendance_staging` | `AttendanceStaging` | Upsert pending student, delete on reject | `websocket_manager.py`, `main.py:staging` | `ScannerKiosk.jsx`, `FacultyConsole.jsx` | **CONNECTION VERIFIED FROM SOURCE** |
| `attendance_ledger` | `AttendanceLedger` | Batch insert upon commit, query history | `main.py:commit_session_attendance()`, `student` | `FacultyConsole.jsx`, `StudentPortal.jsx` | **CONNECTION VERIFIED FROM SOURCE** |
| `attendance` | `Attendance` | Record final present/absent summary | `main.py:commit_session_attendance()` | `ReportHub.jsx` | **CONNECTION VERIFIED FROM SOURCE** |
| `anomaly_events` | `AnomalyEvent` | Insert spoof or presentation attack log | `main.py:log_anomaly()`, `websocket_manager.py` | `ReportHub.jsx` | **CONNECTION VERIFIED FROM SOURCE** |
| `audit_logs` | `AuditLog` | Append cryptographic audit entry | `main.py:telemetry`, `11_anomaly_detection` | `ReportHub.jsx` | **CONNECTION VERIFIED FROM SOURCE** |

---

## 5. Status of Endpoints & Unverified Assets

* **`BACKEND ENDPOINT FOUND — FRONTEND CONSUMER NOT FOUND`**:
  - `POST /api/v1/face/register/{student_id}`: Backend implementation present for single image face registration; frontend relies on batch script `seed_full_roster.py`.
  - `POST /api/v1/embeddings/batch-sync`: Backend endpoint present for forced hard-drive sync; handled automatically at engine initialization by `SiameseMatcher.reload_registry()`.
  - `POST /api/v1/chatbot/chat`: Legacy text chat endpoint; modern frontend routes all AI copilot requests through `POST /api/v1/xai/query-copilot`.
* **`FILE PRESENT — USAGE NOT VERIFIED`**:
  - `06_face_embedding/embedding_engine.py`: Predecessor prototype embedding stub; completely superseded by `06_face_embedding/insight_engine.py`.
  - `08_verification_engine/enroll_test_face.py`: Standalone CLI testing script for enrolled test faces.
  - `classroom.py`: Utility module superseded by `database_core.py` and `main.py`.

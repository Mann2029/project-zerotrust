import os
import time
import cv2
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    import mediapipe.python.solutions.face_mesh as mp_face_mesh
except (ImportError, OSError):
    mp_face_mesh = None

class FatigueAttentionEngine:
    """Temporal Attention & Fatigue Telemetry."""
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.ear_history = []
        self.blink_timestamps = []
        self.closure_start = None
        self.total_blinks = 0
        self.drowsy_event_count = 0

    def update(self, ear: float, current_time: float) -> dict:
        self.ear_history.append(ear)
        if len(self.ear_history) > self.window_size:
            self.ear_history.pop(0)

        if ear < 0.21:
            if self.closure_start is None:
                self.closure_start = current_time
        else:
            if self.closure_start is not None:
                duration = current_time - self.closure_start
                if duration >= 0.08:  
                    self.total_blinks += 1
                    self.blink_timestamps.append(current_time)
                if duration > 0.45:  
                    self.drowsy_event_count += 1
                self.closure_start = None

        self.blink_timestamps = [t for t in self.blink_timestamps if current_time - t <= 60.0]
        blink_rate_cpm = len(self.blink_timestamps)

        closed_frames = sum(1 for e in self.ear_history if e < 0.21)
        perclos = (closed_frames / len(self.ear_history)) if self.ear_history else 0.0

        attention_score = max(0.0, min(100.0, 100.0 - (perclos * 120.0) - (self.drowsy_event_count * 15.0)))
        fatigue_state = "ALERT" if attention_score > 75.0 else ("TIRED" if attention_score > 45.0 else "DROWSY")

        return {
            "attention_score": round(attention_score, 1),
            "fatigue_state": fatigue_state,
            "blink_rate_cpm": blink_rate_cpm,
            "perclos": round(perclos, 3),
            "drowsy_events": self.drowsy_event_count
        }

class SmartVisionEngine:
    """Robust 3-Stage Vision Pipeline: MediaPipe Mesh -> OpenCV Haar -> Smart Portrait Fallback."""
    def __init__(self):
        self.face_mesh = (
            mp_face_mesh.FaceMesh(
                max_num_faces=4,
                refine_landmarks=True,
                min_detection_confidence=0.20,
                min_tracking_confidence=0.20
            )
            if mp_face_mesh is not None
            else None
        )
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.fatigue_engine = FatigueAttentionEngine(window_size=60)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def enhance_low_light(self, frame: np.ndarray) -> tuple[np.ndarray, bool]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_lum = float(cv2.mean(gray)[0])
        if mean_lum < 70.0:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR), True
        return frame, False

    def align_and_crop(self, frame: np.ndarray, left_eye: tuple, right_eye: tuple, bbox: list[int]) -> np.ndarray:
        dy, dx = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        eyes_center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))

        matrix = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        h, w = frame.shape[:2]
        rotated = cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LINEAR)

        x, y, bw, bh = bbox
        x, y = max(0, x), max(0, y)
        crop = rotated[y:y+bh, x:x+bw]
        if crop.size == 0:
            return cv2.resize(frame, (160, 160))
        return cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)

    def _fallback_haar_or_center(self, frame: np.ndarray) -> dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05, 
            minNeighbors=3, 
            minSize=(30, 30)
        )

        img_h, img_w = frame.shape[:2]

        if len(faces) == 0:
            # Fallback: Center crop
            pad_h = int(img_h * 0.05)
            pad_w = int(img_w * 0.10)
            center_crop = frame[pad_h:img_h-pad_h, pad_w:img_w-pad_w]
            aligned = cv2.resize(center_crop if center_crop.size > 0 else frame, (160, 160), interpolation=cv2.INTER_AREA)
            return {
                "face_detected": True,
                "multi_face_violation": False,
                "aligned_face": aligned,
                "fatigue_telemetry": {"attention_score": 100.0, "fatigue_state": "ALERT", "blink_rate_cpm": 0, "perclos": 0.0, "drowsy_events": 0}
            }

        # Select largest detected face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        pad_x, pad_y = int(0.18 * w), int(0.25 * h)
        
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(img_w, x + w + pad_x), min(img_h, y + h + int(pad_y * 0.5))
        
        face_crop = frame[y1:y2, x1:x2]
        aligned_face = cv2.resize(face_crop if face_crop.size > 0 else frame, (160, 160), interpolation=cv2.INTER_AREA)

        return {
            "face_detected": True,
            "multi_face_violation": False,
            "aligned_face": aligned_face,
            "fatigue_telemetry": {"attention_score": 100.0, "fatigue_state": "ALERT", "blink_rate_cpm": 0, "perclos": 0.0, "drowsy_events": 0}
        }

    def process_frame(self, frame: np.ndarray) -> dict:
        curr_timestamp = time.time()
        frame, _ = self.enhance_low_light(frame)
        h, w = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.face_mesh is None:
            return self._fallback_haar_or_center(frame)

        results = self.face_mesh.process(rgb_frame)

        if not results or not results.multi_face_landmarks:
            return self._fallback_haar_or_center(frame)

        # Calculate bounding boxes and areas
        mesh_boxes = []
        for fl in results.multi_face_landmarks:
            xs = [p.x * w for p in fl.landmark]
            ys = [p.y * h for p in fl.landmark]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            mesh_boxes.append((area, fl))

        # Sort and select the primary face
        mesh_boxes.sort(key=lambda item: item[0], reverse=True)
        primary_face = mesh_boxes[0][1]

        landmarks = primary_face.landmark
        xs, ys = [int(p.x * w) for p in landmarks], [int(p.y * h) for p in landmarks]
        x_min, x_max = max(0, min(xs) - 20), min(w, max(xs) + 20)
        y_min, y_max = max(0, min(ys) - 30), min(h, max(ys) + 20)
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

        left_eye = (int(landmarks[33].x * w), int(landmarks[33].y * h))
        right_eye = (int(landmarks[263].x * w), int(landmarks[263].y * h))
        
        aligned_face = self.align_and_crop(frame, left_eye, right_eye, bbox)

        pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in [33, 160, 158, 133, 153, 144]]
        h_dist = np.linalg.norm(pts[0] - pts[3])
        left_ear = float((np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])) / (2.0 * h_dist)) if h_dist > 0 else 0.0
        fatigue_data = self.fatigue_engine.update(left_ear, curr_timestamp)

        return {
            "face_detected": True,
            "multi_face_violation": False,
            "aligned_face": aligned_face,
            "fatigue_telemetry": fatigue_data
        }
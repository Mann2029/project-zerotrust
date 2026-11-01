import cv2
import numpy as np

class PoseSolverPnP:
    """Solves 3D head rotation using Perspective-n-Point geometry."""
    
    def __init__(self):
        # 3D model points of a generic human face
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

    def estimate_pose(self, image: np.ndarray, landmarks: list) -> dict:
        h, w = image.shape[:2]
        
        # Extract corresponding 2D points from the face landmarks
        image_points = np.array([
            (landmarks[1].x * w, landmarks[1].y * h),     # Nose tip
            (landmarks[152].x * w, landmarks[152].y * h), # Chin
            (landmarks[33].x * w, landmarks[33].y * h),   # Left eye corner
            (landmarks[263].x * w, landmarks[263].y * h), # Right eye corner
            (landmarks[61].x * w, landmarks[61].y * h),   # Left mouth corner
            (landmarks[291].x * w, landmarks[291].y * h)  # Right mouth corner
        ], dtype="double")

        # Camera internals setup
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4, 1))

        # Solve PnP
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        # Convert rotation vector to rotation matrix, then to Euler angles
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        proj_mat = np.hstack((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_mat)

        pitch = euler_angles[0][0]
        yaw = euler_angles[1][0]
        roll = euler_angles[2][0]

        return {
            "pitch_deg": round(pitch, 2), # Nodding up/down
            "yaw_deg": round(yaw, 2),     # Turning left/right
            "roll_deg": round(roll, 2)    # Tilting side to side
        }
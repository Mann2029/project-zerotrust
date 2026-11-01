import cv2
import numpy as np

class LivenessTextureEngine:
    """
    Simulates MiniFASNet deep texture analysis using 2D Fast Fourier Transform (FFT) 
    and Micro-Texture Variance to detect screen moiré and paper prints.
    """
    def __init__(self):
        print("[INFO] Booting CNN #3: Spatial-Frequency Texture Engine...")

    def analyze_fourier_spectrum(self, face_crop: np.ndarray) -> float:
        """
        Converts the face to grayscale and applies a 2D Discrete Fourier Transform.
        F(u, v) = sum(sum(f(x,y) * e^(-j2π(ux/M + vy/N))))
        High-frequency spikes indicate screen pixels (moiré) or paper dot-matrix prints.
        """
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        # Compute 2D Fast Fourier Transform
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        
        # Calculate magnitude spectrum
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)
        
        # Calculate high-frequency energy ratio
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Mask out the low-frequency center (natural skin features)
        mask = np.ones((rows, cols), np.uint8)
        r = 15 # Center radius
        mask[crow-r:crow+r, ccol-r:ccol+r] = 0
        
        high_freq_energy = np.sum(magnitude_spectrum * mask)
        total_energy = np.sum(magnitude_spectrum)
        
        if total_energy == 0:
            return 1.0 # Max spoof risk
            
        ratio = high_freq_energy / total_energy
        return float(ratio)

    def evaluate_liveness(self, face_crop: np.ndarray) -> dict:
        if face_crop is None or face_crop.size == 0:
            return {"is_live": False, "score": 0.0, "reason": "Empty frame"}

        # Resize for consistent frequency mapping
        standardized_crop = cv2.resize(face_crop, (160, 160))
        
        # 1. Fourier High-Frequency Check
        hf_ratio = self.analyze_fourier_spectrum(standardized_crop)
        
        # 2. Specular Reflection (Flash/Glare) Check
        hsv = cv2.cvtColor(standardized_crop, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)
        glare_ratio = np.sum(v > 240) / (160 * 160)

        # Decision Matrix
        is_live = True
        reason = "Passed"
        confidence = 95.0

        # Change this line:
        if hf_ratio > 0.965:  # Relaxed from 0.88 to account for normal camera grain
            is_live = False
            reason = "SPOOF: Screen Moiré or Print Pattern Detected"
            confidence = (1.0 - hf_ratio) * 100

        elif glare_ratio > 0.15:
            is_live = False
            reason = "SPOOF: Unnatural Screen Glare Detected"
            confidence = (1.0 - glare_ratio) * 100

        return {
            "is_live": is_live,
            "confidence_score": round(confidence, 2),
            "texture_hf_ratio": round(hf_ratio, 4),
            "glare_ratio": round(glare_ratio, 4),
            "reason": reason
        }
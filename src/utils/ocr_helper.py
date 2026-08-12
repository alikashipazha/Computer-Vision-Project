import cv2
import numpy as np
try:
    import pytesseract
except ImportError:
    pytesseract = None

class OCRHelper:
    """
    Wrapper class for Tesseract OCR to evaluate document legibility.
    """
    def __init__(self):
        self.available = pytesseract is not None

    def get_ocr_confidence(self, img_bgr: np.ndarray) -> float:
        """
        Runs Tesseract OCR on a BGR image and computes the average word confidence.
        
        Args:
            img_bgr: Input OpenCV image in BGR format.
        Returns:
            Average confidence score of detected words (0.0 to 100.0).
        """
        if not self.available:
            return 0.0
        
        try:
            # Convert BGR to RGB as expected by Tesseract
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Fetch detailed OCR data including word-level confidence
            ocr_data = pytesseract.image_to_data(img_rgb, output_type=pytesseract.Output.DICT)
            
            # Filter valid word confidences (Tesseract returns -1 for empty regions/non-words)
            confidences = [float(conf) for conf in ocr_data['conf'] if conf != -1]
            
            if len(confidences) == 0:
                return 0.0
            
            return float(np.mean(confidences))
        except Exception as e:
            print(f"[OCR Warning] Failed to run Tesseract: {e}")
            return 0.0

    def extract_text(self, img_bgr: np.ndarray) -> str:
        """
        Extracts raw text from a BGR image.
        """
        if not self.available:
            return ""
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return pytesseract.image_to_string(img_rgb)
        except Exception:
            return ""
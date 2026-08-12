import cv2
import numpy as np
import random
from pathlib import Path
from typing import Tuple, List, Dict, Any

class SyntheticDatasetGenerator:
    """
    Synthetic dataset generator for document corner detection and enhancement networks.
    """
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        """
        Args:
            target_size: Output resolution of the rectified document (width, height).
        """
        self.target_size = target_size

    def _generate_random_quadrilateral(self, bg_w: int, bg_h: int) -> np.ndarray:
        """
        Generates 4 random points on the background to simulate a natural perspective.
        Points are ordered consistently: [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
        """
        # Define a safety margin to prevent points from escaping the background borders
        margin_w = int(bg_w * 0.15)
        margin_h = int(bg_h * 0.15)
        
        # Approximate document dimensions on the background
        doc_w = int(bg_w * 0.6)
        doc_h = int(bg_h * 0.6)
        
        # Center of the background canvas
        center_x = bg_w // 2
        center_y = bg_h // 2
        
        # Base rectangle vertices
        tl = [center_x - doc_w // 2, center_y - doc_h // 2]
        tr = [center_x + doc_w // 2, center_y - doc_h // 2]
        br = [center_x + doc_w // 2, center_y + doc_h // 2]
        bl = [center_x - doc_w // 2, center_y + doc_h // 2]
        
        # Add random jitter to simulate diverse camera angles
        jitter_range_x = int(bg_w * 0.08)
        jitter_range_y = int(bg_h * 0.08)
        
        points = [
            [tl[0] + random.randint(-jitter_range_x, jitter_range_x), tl[1] + random.randint(-jitter_range_y, jitter_range_y)],
            [tr[0] + random.randint(-jitter_range_x, jitter_range_x), tr[1] + random.randint(-jitter_range_y, jitter_range_y)],
            [br[0] + random.randint(-jitter_range_x, jitter_range_x), br[1] + random.randint(-jitter_range_y, jitter_range_y)],
            [bl[0] + random.randint(-jitter_range_x, jitter_range_x), bl[1] + random.randint(-jitter_range_y, jitter_range_y)]
        ]
        
        # Clip points to stay strictly within background boundaries
        for pt in points:
            pt[0] = max(0, min(pt[0], bg_w - 1))
            pt[1] = max(0, min(pt[1], bg_h - 1))
            
        return np.array(points, dtype=np.float32)

    def degrade_image(self, warped_img: np.ndarray) -> np.ndarray:
        """
        Placeholder for image degradation (shadows, blur, color casts).
        To be implemented with the actual degradation pipeline in Section 4.
        """
        return warped_img.copy()

    def generate_pair(self, clean_scan: np.ndarray, background: np.ndarray) -> Dict[str, Any]:
        """
        Generates a synthetic pair (degraded raw photo, annotated corners, rectified inputs/targets).
        
        Args:
            clean_scan: High-resolution clean scan (Ground Truth target).
            background: Background image.
            
        Returns:
            Dictionary containing:
                - 'raw_photo': Degraded composite image on background.
                - 'corners': 4 corner coordinates on the background.
                - 'rectified_input': Flat, degraded document crop.
                - 'rectified_target': Flat, clean document target.
        """
        bg_h, bg_w = background.shape[:2]
        
        # 1. Resize clean scan to standardize homography processing
        clean_scan_resized = cv2.resize(clean_scan, self.target_size)
        scan_h, scan_w = clean_scan_resized.shape[:2]
        
        # 2. Source points (corners of the clean flat scan)
        src_pts = np.array([
            [0, 0],                  # Top-Left
            [scan_w - 1, 0],         # Top-Right
            [scan_w - 1, scan_h - 1],# Bottom-Right
            [0, scan_h - 1]          # Bottom-Left
        ], dtype=np.float32)
        
        # 3. Destination points (random quadrilateral on the background)
        dst_pts = self._generate_random_quadrilateral(bg_w, bg_h)
        
        # 4. Perspective transform: warp scan onto the background
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_scan = cv2.warpPerspective(clean_scan_resized, M, (bg_w, bg_h))
        
        # Create mask to blend document smoothly onto background
        mask = np.zeros((scan_h, scan_w), dtype=np.uint8) + 255
        warped_mask = cv2.warpPerspective(mask, M, (bg_w, bg_h))
        
        # Composite scan and background
        raw_photo = background.copy()
        raw_photo[warped_mask > 0] = warped_scan[warped_mask > 0]
        
        # 5. Apply degradation pipeline
        degraded_raw_photo = self.degrade_image(raw_photo)
        
        # 6. Rectify back using inverse perspective transform
        M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
        rectified_input = cv2.warpPerspective(degraded_raw_photo, M_inv, self.target_size)
        
        return {
            "raw_photo": degraded_raw_photo,
            "corners": dst_pts,
            "rectified_input": rectified_input,
            "rectified_target": clean_scan_resized
        }
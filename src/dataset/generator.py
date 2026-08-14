import cv2
import numpy as np
import random
from typing import Tuple, Dict, Any

class SyntheticDatasetGenerator:
    """
    Synthetic dataset generator for document corner detection and enhancement networks.
    Features a physically-grounded 6-step photometric degradation pipeline built strictly on OpenCV and NumPy.
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
        margin_w = int(bg_w * 0.15)
        margin_h = int(bg_h * 0.15)
        
        doc_w = int(bg_w * 0.6)
        doc_h = int(bg_h * 0.6)
        
        center_x = bg_w // 2
        center_y = bg_h // 2
        
        tl = [center_x - doc_w // 2, center_y - doc_h // 2]
        tr = [center_x + doc_w // 2, center_y - doc_h // 2]
        br = [center_x + doc_w // 2, center_y + doc_h // 2]
        bl = [center_x - doc_w // 2, center_y + doc_h // 2]
        
        jitter_range_x = int(bg_w * 0.08)
        jitter_range_y = int(bg_h * 0.08)
        
        # REVERTED TO STRICT CCW SEQUENCE: [TL, BL, BR, TR]
        points = [
            [tl[0] + random.randint(-jitter_range_x, jitter_range_x), tl[1] + random.randint(-jitter_range_y, jitter_range_y)], # TL
            [bl[0] + random.randint(-jitter_range_x, jitter_range_x), bl[1] + random.randint(-jitter_range_y, jitter_range_y)], # BL
            [br[0] + random.randint(-jitter_range_x, jitter_range_x), br[1] + random.randint(-jitter_range_y, jitter_range_y)], # BR
            [tr[0] + random.randint(-jitter_range_x, jitter_range_x), tr[1] + random.randint(-jitter_range_y, jitter_range_y)]  # TR
        ]
        
        for pt in points:
            pt[0] = max(0, min(pt[0], bg_w - 1))
            pt[1] = max(0, min(pt[1], bg_h - 1))
            
        return np.array(points, dtype=np.float32)

    def _apply_resolution_loss(self, img: np.ndarray) -> np.ndarray:
        """
        Step 2: Simulate physical distance by downscaling and upscaling.
        Calibrated: Reduced max scale factor to 2.2x to prevent text from melting.
        """
        h, w = img.shape[:2]
        scale_factor = random.uniform(1.2, 2.2) # Gentle downscaling
        
        new_w = max(16, int(w / scale_factor))
        new_h = max(16, int(h / scale_factor))
        
        downscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_LINEAR)
        return upscaled

    def _apply_brightness_contrast_color(self, img: np.ndarray) -> np.ndarray:
        """
        Step 3: Apply random contrast, brightness, and color casting.
        Calibrated: Tightened ranges for a more natural indoor lighting look.
        """
        img_float = img.astype(np.float32)
        
        alpha = random.uniform(0.85, 1.15) # Milder contrast range
        beta = random.uniform(-20.0, 20.0) # Milder brightness shift
        img_adjusted = img_float * alpha + beta
        
        # Milder color casting (warm/cool)
        cast_b = random.uniform(0.92, 1.08)
        cast_r = random.uniform(0.92, 1.08)
        
        img_adjusted[:, :, 0] *= cast_b
        img_adjusted[:, :, 2] *= cast_r
        
        return np.clip(img_adjusted, 0, 255).astype(np.uint8)

    def _apply_illumination_and_shadows(self, img: np.ndarray) -> np.ndarray:
        """
        Step 4: Composite dynamic 2D linear illumination gradients and soft random shadow polygons.
        """
        h, w, c = img.shape
        img_float = img.astype(np.float32) / 255.0
        
        # 1. Generate 2D Linear Illumination Gradient
        x = np.linspace(0, 1, w, dtype=np.float32)
        y = np.linspace(0, 1, h, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        
        # Random linear gradient angle equation: ax + by + c
        angle_weight_x = random.uniform(-1.0, 1.0)
        angle_weight_y = random.uniform(-1.0, 1.0)
        gradient = (xv * angle_weight_x + yv * angle_weight_y)
        
        # Normalize gradient map to [min_intensity, 1.0]
        min_intensity = random.uniform(0.45, 0.75)
        grad_min, grad_max = gradient.min(), gradient.max()
        if grad_max > grad_min:
            gradient = (gradient - grad_min) / (grad_max - grad_min)
        gradient = gradient * (1.0 - min_intensity) + min_intensity
        gradient = np.expand_dims(gradient, axis=2)  # Expand to match channels (H, W, 1)
        
        img_gradient = img_float * gradient
        
        # 2. Composite Soft Shadow Polygons
        shadow_mask = np.zeros((h, w), dtype=np.float32)
        num_shadows = random.randint(1, 3)
        
        for _ in range(num_shadows):
            # Generate random polygon coordinates for the shadow shape
            num_vertices = random.randint(3, 5)
            pts = []
            for _ in range(num_vertices):
                pts.append([random.randint(0, w - 1), random.randint(0, h - 1)])
            pts = np.array(pts, dtype=np.int32)
            
            # Fill polygon on a temp mask
            temp_mask = np.zeros((h, w), dtype=np.float32)
            cv2.fillPoly(temp_mask, [pts], 1.0)
            
            # Blur the mask extensively with a massive kernel to create soft shadow edges
            blur_kernel = random.choice([71, 101, 131, 151])
            temp_mask_blurred = cv2.GaussianBlur(temp_mask, (blur_kernel, blur_kernel), 0)
            shadow_mask = np.maximum(shadow_mask, temp_mask_blurred)
            
        # Blend the soft shadow onto the image (dimming by 15% to 45% based on mask intensity)
        shadow_strength = random.uniform(0.15, 0.45)
        shadow_map = 1.0 - (shadow_mask * shadow_strength)
        shadow_map = np.expand_dims(shadow_map, axis=2)
        
        img_final = img_gradient * shadow_map
        return (img_final * 255.0).astype(np.uint8)

    def _apply_blur_and_noise(self, img: np.ndarray) -> np.ndarray:
        """
        Step 5: Apply Gaussian Blur and introduce additive Gaussian Noise.
        Calibrated: Limited kernel size and reduced noise standard deviation.
        """
        # Calibrated: Standard lens blur (avoiding 7x7 heavy blur)
        kernel_size = random.choice([3, 5])
        sigma = random.uniform(0.3, 1.0)
        blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        
        # Calibrated: Milder realistic sensor noise (std dev reduced from 12 to 6)
        noise_std = random.uniform(2.0, 6.0)
        noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
        
        img_noisy = blurred.astype(np.float32) + noise
        return np.clip(img_noisy, 0, 255).astype(np.uint8)

    def _apply_jpeg_compression(self, img: np.ndarray) -> np.ndarray:
        """
        Step 6: Re-encode image under a random JPEG quality parameter.
        Calibrated: Increased minimum quality floor to 50 to avoid blocky distortion.
        """
        quality = random.randint(50, 85) # High-quality mobile JPEG simulation
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        
        success, encoded_img = cv2.imencode('.jpg', img, encode_param)
        if success:
            return cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        return img.copy()

    def degrade_image(self, warped_img: np.ndarray) -> np.ndarray:
        """
        Sequential execution of the 6-step physical degradation pipeline.
        
        Pipeline order:
          1. Perspective Warp (applied outside in generate_pair)
          2. Resolution Loss (scaling down-up)
          3. Brightness, Contrast & Color Cast adjustments
          4. Illumination Gradients & Soft Shadows blending
          5. Gaussian Blur & Additive Gaussian Noise
          6. JPEG Re-encoding Compression
        """
        img = warped_img.copy()
        img = self._apply_resolution_loss(img)
        img = self._apply_brightness_contrast_color(img)
        img = self._apply_illumination_and_shadows(img)
        img = self._apply_blur_and_noise(img)
        img = self._apply_jpeg_compression(img)
        return img

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
        
        # 2. Source points in strict Counter-Clockwise [TL, BL, BR, TR] sequence
        src_pts = np.array([
            [0, 0],                  # TL
            [0, scan_h - 1],         # BL
            [scan_w - 1, scan_h - 1],# BR
            [scan_w - 1, 0]          # TR
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
        
        # 5. Apply complete sequential degradation pipeline to composite raw image
        degraded_raw_photo = self.degrade_image(raw_photo)
        
        # 6. Rectify degraded image back using inverse perspective transform
        M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
        rectified_input = cv2.warpPerspective(degraded_raw_photo, M_inv, self.target_size)
        
        return {
            "raw_photo": degraded_raw_photo,
            "corners": dst_pts,
            "rectified_input": rectified_input,
            "rectified_target": clean_scan_resized
        }
import os
import cv2
import numpy as np
import torch
from src.inference.corner_pipeline import CornerDetectionPipeline
from src.inference.pipeline import EnhancementPipeline
from src.dataset.loader import order_points

class EndToEndScannerPipeline:
    """
    Automatic end-to-end Document Scanner.
    Chains: Corner Detection Network (Approach B) -> Perspective Warp -> Document Enhancement Network.
    Requires no human intervention.
    """
    def __init__(
        self, 
        corner_checkpoint: str = 'checkpoints/corner_heat_best.pth',
        enhancement_checkpoint: str = 'checkpoints/enhancement_best.pth'
    ):
        # Initialize individual pipelines
        self.corner_pipeline = CornerDetectionPipeline(model_checkpoint_path=corner_checkpoint)
        self.enhancement_pipeline = EnhancementPipeline(model_checkpoint_path=enhancement_checkpoint)
        self.target_size = (512, 512)
        print("[Pipeline] End-to-end document scanner initialized successfully.")

    def scan_document(self, raw_img_bgr: np.ndarray) -> np.ndarray:
        """
        Executes the full automated scanning chain on a raw smartphone photo.
        
        Args:
            raw_img_bgr: Raw unprocessed smartphone image of a document.
        Returns:
            Cleaned, flat, scan-quality document crop.
        """
        orig_h, orig_w = raw_img_bgr.shape[:2]
        
        # Step 1: Predict corner landmarks using the trained Heatmap U-Net
        predicted_corners = self.corner_pipeline.predict_corners(raw_img_bgr) # Shape (4, 2)
        
        # Step 2: Order points mathematically to prevent perspective flipping/rotation
        ordered_corners = order_points(predicted_corners)
        
        # Step 3: Perform Perspective Warp (Rectification) on original high-res image
        src_pts = ordered_corners
        # Rectification target in strict Counter-Clockwise [TL, BL, BR, TR] sequence
        dst_pts = np.array([
            [0, 0],
            [0, self.target_size[1] - 1],
            [self.target_size[0] - 1, self.target_size[1] - 1],
            [self.target_size[0] - 1, 0]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        rectified_crop = cv2.warpPerspective(raw_img_bgr, M, self.target_size)
        
        # Step 4: Enhance the rectified crop (whiten background, remove shadows/noise)
        final_scan = self.enhancement_pipeline.process_image(rectified_crop)
        return final_scan
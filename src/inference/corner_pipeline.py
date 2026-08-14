import os
import cv2
import numpy as np
import torch
from src.dataset.loader import IMAGE_TRANSFORM
from src.models.corner_models import HeatmapUNet
from src.evaluation.evaluate_corners import extract_coords_from_heatmaps

class CornerDetectionPipeline:
    """
    Inference Pipeline for predicting and mapping document corners from raw smartphone photos.
    Uses Approach B (Heatmap Regression) as the standard high-accuracy detector.
    """
    def __init__(self, model_checkpoint_path: str = 'checkpoints/corner_heat_best.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_size = (512, 512)
        
        self.model = HeatmapUNet(in_channels=3, out_channels=4).to(self.device)
        
        if not os.path.exists(model_checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at: {model_checkpoint_path}")
            
        # FIX: Safely load checkpoint with weights_only=True and handle wrapped dictionary structure
        checkpoint = torch.load(model_checkpoint_path, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        print("[Pipeline] Corner detection network loaded successfully.")

    def predict_corners(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Predicts 4 corner coordinates scaled back to the original image dimensions.
        
        Args:
            img_bgr: Raw smartphone photo with arbitrary dimensions.
        Returns:
            Normalized coordinates scaled back to original resolution: shape (4, 2).
        """
        orig_h, orig_w = img_bgr.shape[:2]
        
        # 1. Preprocess: Resize and Normalize
        resized = cv2.resize(img_bgr, self.target_size)
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor_input = IMAGE_TRANSFORM(resized_rgb).to(self.device).unsqueeze(0)
        
        # 2. Predict Heatmaps
        with torch.no_grad():
            heatmaps = self.model(tensor_input)
            
        # 3. Extract normalized coordinates via spatial argmax
        normalized_coords = extract_coords_from_heatmaps(heatmaps).squeeze(0)  # Shape (4, 2)
        
        # 4. Scale corners back to original pixel dimensions
        scaled_coords = normalized_coords * np.array([orig_w, orig_h], dtype=np.float32)
        return scaled_coords

    def draw_predicted_corners(self, img_bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Overlays circles and index numbers on the original raw BGR photo.
        """
        drawn_img = img_bgr.copy()
        corner_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)] # Distinct colors per corner
        
        for i, pt in enumerate(corners.astype(np.int32)):
            cv2.circle(drawn_img, tuple(pt), 15, corner_colors[i], -1)
            cv2.putText(
                drawn_img, 
                str(i + 1), 
                (pt[0] + 20, pt[1] + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, 
                (255, 255, 255), 
                3, 
                cv2.LINE_AA
            )
        return drawn_img
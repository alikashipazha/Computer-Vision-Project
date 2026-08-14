import os
import cv2
import numpy as np
import torch
from src.dataset.loader import IMAGE_TRANSFORM, INV_IMAGE_TRANSFORM
from src.models.enhancement_model import CustomUNet

class EnhancementPipeline:
    """
    Inference Pipeline for enhancing rectified document crops.
    """
    def __init__(self, model_checkpoint_path: str = 'checkpoints/enhancement_best.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CustomUNet(in_channels=3, out_channels=3).to(self.device)
        
        if not os.path.exists(model_checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at: {model_checkpoint_path}")
            
        # FIX: Safely load checkpoint with weights_only=True and handle wrapped dictionary structure
        checkpoint = torch.load(model_checkpoint_path, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        print("[Pipeline] Enhancement network loaded successfully.")

    def process_image(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Enhances a single flat BGR document crop.
        """
        orig_h, orig_w = img_bgr.shape[:2]
        
        # 1. Preprocess: Resize and Normalize
        resized = cv2.resize(img_bgr, (512, 512))
        # Convert BGR to RGB
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor_input = IMAGE_TRANSFORM(resized_rgb).to(self.device).unsqueeze(0)
        
        # 2. Predict
        with torch.no_grad():
            tensor_output = self.model(tensor_input).squeeze(0).cpu()
            
        # 3. Post-process: Convert to numpy and restore scale
        pred_np = (tensor_output.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)
        
        # Resize back to original input dimensions
        final_output = cv2.resize(pred_bgr, (orig_w, orig_h))
        return final_output
import os
import cv2
import numpy as np
from src.dataset.loader import RealTestDataset, INV_IMAGE_TRANSFORM
from src.inference.pipeline import EnhancementPipeline

def main():
    os.makedirs('docs', exist_ok=True)
    
    try:
        # Load real test dataset to get a rectified real input
        real_ds = RealTestDataset(real_test_dir="data/real_test", target_size=(512, 512))
        sample = real_ds[0]
        
        # Revert PyTorch normalization to get the standard BGR rectified input
        input_unnorm = INV_IMAGE_TRANSFORM(sample['rectified_input'])
        rectified_input_np = (input_unnorm.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        rectified_input_bgr = cv2.cvtColor(rectified_input_np, cv2.COLOR_RGB2BGR)
        
        print("Initializing the enhancement pipeline...")
        pipeline = EnhancementPipeline()
        
        print("Enhancing the rectified input image...")
        enhanced_output = pipeline.process_image(rectified_input_bgr)
        
        # Save side-by-side visualization
        # Left: Rectified input (yellowish/shadowed), Right: Enhanced output (clean/white)
        comparison = np.hstack([rectified_input_bgr, enhanced_output])
        output_path = "docs/enhancement_visualization.jpg"
        cv2.imwrite(output_path, comparison)
        print(f"Visualization successful! Check output image at: {output_path}")
        
    except Exception as e:
        print(f"Error executing enhancement visualization pipeline: {e}")

if __name__ == "__main__":
    main()
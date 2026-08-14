import os
import cv2
import numpy as np
import glob
from pathlib import Path
from PIL import Image, ImageOps
from src.inference.corner_pipeline import CornerDetectionPipeline

def main():
    # Setup directories
    os.makedirs('docs', exist_ok=True)
    real_test_dir = Path("data/real_test/images")
    
    # Locate first available real test photo
    real_photos = sorted(glob.glob(str(real_test_dir / "*.*")))
    if len(real_photos) == 0:
        print(f"Error: No real test photos found in {real_test_dir}")
        return
        
    sample_photo_path = real_photos[0]
    print(f"Loading raw photo for visualization: {sample_photo_path}")
    
    try:
        # Load raw photo and correct smartphone EXIF orientation metadata
        pil_img = Image.open(sample_photo_path)
        pil_img = ImageOps.exif_transpose(pil_img)
        raw_photo = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Initialize the corner detection pipeline (loads best Heatmap checkpoint)
        pipeline = CornerDetectionPipeline()
        
        # Predict corners scaled back to original image resolution
        predicted_corners = pipeline.predict_corners(raw_photo)
        
        # Draw colored circles and indices on original high-res raw photo
        drawn_img = pipeline.draw_predicted_corners(raw_photo, predicted_corners)
        
        # Save output image
        output_path = "docs/corner_detection_visualization.jpg"
        cv2.imwrite(output_path, drawn_img)
        print(f"Visualization successful! Check output image at: {output_path}")
        
    except Exception as e:
        print(f"Error executing corner detection visualization pipeline: {e}")

if __name__ == "__main__":
    main()
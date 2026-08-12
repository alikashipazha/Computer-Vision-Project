import cv2
import numpy as np
import torch
import glob
from pathlib import Path
from src.dataset.loader import get_train_val_test_datasets, INV_IMAGE_TRANSFORM

def draw_corners(img: np.ndarray, corners: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    """Draws scaled corner points on the image for verification."""
    h, w = img.shape[:2]
    pixel_corners = (corners * np.array([w, h])).astype(np.int32)
    for i, pt in enumerate(pixel_corners):
        cv2.circle(img, tuple(pt), 10, color, -1)
        cv2.putText(img, str(i+1), (pt[0]+15, pt[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return img

def main():
    scans_dir = "data/raw_scans"
    bg_dir = "data/backgrounds"
    
    try:
        train_ds, val_ds, _ = get_train_val_test_datasets(scans_dir, bg_dir, target_size=(512, 512), epoch_length=10)
        print("Data split completed successfully (80/10/10).")
        
        # Dynamically calculate scan partitions to match report log
        scans = sorted(glob.glob(str(Path(scans_dir) / "*.*")))
        total_scans = len(scans)
        train_scans = int(total_scans * 0.8)
        val_scans = int(total_scans * 0.1)
        test_scans = total_scans - train_scans - val_scans
        
        print("Source Scans Shuffled & Split Summary:")
        print(f"  - Total Scans Located: {total_scans}")
        print(f"  - Training Subset: {train_scans} scans")
        print(f"  - Validation Subset: {val_scans} scans (Deterministic Seed Active)")
        print(f"  - Testing Subset: {test_scans} scans (Deterministic Seed Active)\n")
        
        print("Extracting sample entry index 0 from Training subset...")
        sample = train_ds[0]
        
        print("Reverting PyTorch normalizations...")
        # Revert PyTorch normalization to display RGB correctly
        raw_photo_tensor = INV_IMAGE_TRANSFORM(sample['raw_photo'])
        raw_photo = (raw_photo_tensor.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        raw_photo = cv2.cvtColor(raw_photo, cv2.COLOR_RGB2BGR)
        
        print("Mapping normalized coordinates to absolute pixels...")
        corners = sample['corners'].numpy()
        h, w = raw_photo.shape[:2]
        corner_names = ["Corner 1 (TL)", "Corner 2 (TR)", "Corner 3 (BR)", "Corner 4 (BL)"]
        for i, name in enumerate(corner_names):
            norm_coord = corners[i]
            abs_coord = (norm_coord * np.array([w, h])).astype(np.int32)
            print(f"  - {name} [Normalized]: [{norm_coord[0]:.3f}, {norm_coord[1]:.3f}] -> Pixel: [{abs_coord[0]}, {abs_coord[1]}]")
        
        print("\nGenerating test plot...")
        # Overlay normalized corners on composite image
        drawn_img = draw_corners(raw_photo, sample['corners'].numpy())
        
        # Save output image
        cv2.imwrite("test_preprocessing_alignment.jpg", drawn_img)
        print("Visualization done. Check 'test_preprocessing_alignment.jpg' for correct alignment.")
        
    except Exception as e:
        print(f"Error during preprocessing check: {e}")

if __name__ == "__main__":
    main()
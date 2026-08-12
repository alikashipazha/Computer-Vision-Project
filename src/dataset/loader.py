import cv2
import numpy as np
import json
import glob
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, List, Dict, Any

# Standard image normalization for PyTorch (ImageNet Mean & Std)
IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Inverse transform for display and visualization phases
INV_IMAGE_TRANSFORM = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
])


class SyntheticDocumentDataset(Dataset):
    """
    Synthetic training, validation, and testing dataset.
    Supports freezing for Val and Test splits using a fixed Seed.
    """
    def __init__(
        self, 
        scan_files: List[str], 
        background_files: List[str], 
        target_size: Tuple[int, int] = (512, 512), 
        epoch_length: int = 1000,
        frozen: bool = False
    ):
        self.scan_files = scan_files
        self.bg_files = background_files
        self.target_size = target_size
        self.epoch_length = epoch_length
        self.frozen = frozen
        
        # Use the geometric generator designed in Phase 1
        from src.dataset.generator import SyntheticDatasetGenerator
        self.generator = SyntheticDatasetGenerator(target_size=target_size)

    def __len__(self) -> int:
        return self.epoch_length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # If the dataset is frozen (Val/Test), we use a fixed seed based on the index
        # to ensure the output for this index remains identical across all epochs.
        if self.frozen:
            random.seed(idx)
            np.random.seed(idx)
        else:
            # Revert to fully random state for dynamic training
            random.seed(None)
            np.random.seed(None)

        scan_path = random.choice(self.scan_files)
        bg_path = random.choice(self.bg_files)

        clean_scan = cv2.imread(scan_path)
        background = cv2.imread(bg_path)

        data_pair = self.generator.generate_pair(clean_scan, background)

        # Background image dimensions for corner normalization
        bg_h, bg_w = data_pair['raw_photo'].shape[:2]

        # Normalize corners to [0, 1] range based on background image dimensions
        normalized_corners = data_pair['corners'].copy()
        normalized_corners[:, 0] /= bg_w
        normalized_corners[:, 1] /= bg_h

        # FIX: Resize the raw_photo to self.target_size so all batch entries have the same resolution
        resized_raw_photo = cv2.resize(data_pair['raw_photo'], self.target_size)

        # Preprocessing and normalizing images to send to the model
        raw_photo_tensor = IMAGE_TRANSFORM(resized_raw_photo)
        rectified_input_tensor = IMAGE_TRANSFORM(data_pair['rectified_input'])
        
        # Target image does not need complex image preprocessing, it is just transformed to Tensor
        rectified_target_tensor = transforms.ToTensor()(data_pair['rectified_target'])

        return {
            "raw_photo": raw_photo_tensor,                         # Corner detection network input (C, H, W)
            "corners": torch.tensor(normalized_corners, dtype=torch.float32), # Corner detection network target (4, 2)
            "rectified_input": rectified_input_tensor,             # Image enhancement network input (C, H, W)
            "rectified_target": rectified_target_tensor            # Image enhancement network target (C, H, W)
        }


class RealTestDataset(Dataset):
    """
    Custom loader for real test images.
    Parses the Roboflow annotations.json and prepares real images for final testing.
    """
    def __init__(self, real_test_dir: str, target_size: Tuple[int, int] = (512, 512)):
        self.real_test_dir = Path(real_test_dir)
        self.target_size = target_size
        self.annotations_path = self.real_test_dir / "annotations.json"
        
        # Load and parse Roboflow annotation file (standard COCO Keypoints or simple JSON)
        with open(self.annotations_path, 'r') as f:
            self.coco_data = json.load(f)
            
        # Map image IDs to label details
        self.images_info = {img['id']: img for img in self.coco_data['images']}
        self.annotations = self.coco_data['annotations']

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ann = self.annotations[idx]
        image_id = ann['image_id']
        image_info = self.images_info[image_id]
        
        # Read the real smartphone image
        img_path = self.real_test_dir / "images" / image_info['file_name']
        raw_photo = cv2.imread(str(img_path))
        if raw_photo is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
            
        orig_h, orig_w = raw_photo.shape[:2]
        
        # Extract 4 corners (absolute pixel coordinates) from Roboflow output
        # In COCO Keypoints format, the structure is [x1, y1, v1, x2, y2, v2, ...]
        keypoints = ann['keypoints']
        corners = []
        for i in range(0, 12, 3):
            corners.append([keypoints[i], keypoints[i+1]])
        corners = np.array(corners, dtype=np.float32) # Dimensions (4, 2)
        
        # 1. Resize real test image to standard model dimensions
        resized_photo = cv2.resize(raw_photo, self.target_size)
        
        # 2. Normalize corners to [0, 1] based on original labeled image dimensions
        normalized_corners = corners.copy()
        normalized_corners[:, 0] /= orig_w
        normalized_corners[:, 1] /= orig_h
        
        # 3. Generate real flat rectified crop based on real corners for image enhancement network
        # Convert original corners to image coordinates using the homography generator class
        from src.dataset.generator import SyntheticDatasetGenerator
        gen = SyntheticDatasetGenerator(target_size=self.target_size)
        
        # warpPerspective inputs require the pixel dimensions of the current image
        pixel_corners = normalized_corners * np.array([orig_w, orig_h], dtype=np.float32)
        
        src_pts = pixel_corners
        dst_pts = np.array([
            [0, 0],
            [self.target_size[0] - 1, 0],
            [self.target_size[0] - 1, self.target_size[1] - 1],
            [0, self.target_size[1] - 1]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        rectified_real_input = cv2.warpPerspective(raw_photo, M, self.target_size)
        
        # 4. Load corresponding reference scan (CamScanner) and resize to target dimensions for final evaluation
        ref_path = self.real_test_dir / "reference_scans" / image_info['file_name']
        ref_scan = cv2.imread(str(ref_path))
        if ref_scan is not None:
            resized_ref_scan = cv2.resize(ref_scan, self.target_size)
            rectified_target_tensor = transforms.ToTensor()(resized_ref_scan)
        else:
            # In case the reference scan file does not exist (e.g. naming mismatch)
            rectified_target_tensor = torch.zeros((3, self.target_size[1], self.target_size[0]))

        return {
            "raw_photo": IMAGE_TRANSFORM(resized_photo),                               # Corner detection input (C, H, W)
            "corners": torch.tensor(normalized_corners, dtype=torch.float32),         # Normalized corners (4, 2)
            "rectified_input": IMAGE_TRANSFORM(rectified_real_input),                 # Enhancement network input (C, H, W)
            "rectified_target": rectified_target_tensor                               # CamScanner reference scan (C, H, W)
        }


def get_train_val_test_datasets(
    raw_scans_dir: str, 
    backgrounds_dir: str, 
    target_size: Tuple[int, int] = (512, 512),
    epoch_length: int = 1000
) -> Tuple[SyntheticDocumentDataset, SyntheticDocumentDataset, SyntheticDocumentDataset]:
    """
    Split clean scans into 80% Train, 10% Val, 10% Test
    and return 3 distinct datasets with stable behavior.
    """
    scans = sorted(glob.glob(str(Path(raw_scans_dir) / "*.*")))
    backgrounds = sorted(glob.glob(str(Path(backgrounds_dir) / "*.*")))
    
    if len(scans) < 3:
        raise ValueError("Insufficient clean source scans for splitting. At least 3 scans are required.")
        
    random.seed(42) # To guarantee consistent split alignment across different runs
    random.shuffle(scans)
    
    n_scans = len(scans)
    train_end = int(n_scans * 0.8)
    val_end = int(n_scans * 0.9)
    
    train_scans = scans[:train_end]
    val_scans = scans[train_end:val_end]
    test_scans = scans[val_end:]
    
    # Create datasets
    train_dataset = SyntheticDocumentDataset(train_scans, backgrounds, target_size, epoch_length=epoch_length, frozen=False)
    val_dataset = SyntheticDocumentDataset(val_scans, backgrounds, target_size, epoch_length=int(epoch_length * 0.1), frozen=True)
    test_dataset = SyntheticDocumentDataset(test_scans, backgrounds, target_size, epoch_length=int(epoch_length * 0.1), frozen=True)
    
    return train_dataset, val_dataset, test_dataset
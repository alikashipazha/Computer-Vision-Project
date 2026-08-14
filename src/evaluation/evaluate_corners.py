import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.dataset.loader import get_train_val_test_datasets, RealTestDataset
from src.models.corner_models import DirectRegressionNet, HeatmapUNet
import torch.nn as nn

def extract_coords_from_heatmaps(heatmaps: torch.Tensor) -> np.ndarray:
    """
    Extracts (x, y) coordinates from 4-channel heatmaps using 2D Spatial Argmax.
    Returns normalized coordinates of shape (B, 4, 2).
    """
    B, C, H, W = heatmaps.shape
    coords = []
    for b in range(B):
        batch_coords = []
        for c in range(C):
            heatmap = heatmaps[b, c]
            flat_idx = torch.argmax(heatmap)
            y = float(flat_idx // W)
            x = float(flat_idx % W)
            # Normalize to [0, 1] range
            batch_coords.append([x / (W - 1), y / (H - 1)])
        coords.append(batch_coords)
    return np.array(coords, dtype=np.float32)


def calculate_metrics(preds: np.ndarray, targets: np.ndarray, threshold_px: float = 10.0) -> tuple:
    """
    Computes Mean Localization Error (L2 pixel distance on 512x512 canvas)
    and Success Rate (percentage of images where all 4 corners are <= threshold_px).
    
    Args:
        preds: Predicted corners in shape (N, 4, 2) normalized.
        targets: Ground-truth corners in shape (N, 4, 2) normalized.
        threshold_px: Target pixel threshold for a successful detection.
    """
    N = preds.shape[0]
    errors_px = []
    successes = 0
    
    for i in range(N):
        pred_scaled = preds[i] * 512.0
        target_scaled = targets[i] * 512.0
        
        # Calculate Euclidean distances for all 4 corners
        distances = np.linalg.norm(pred_scaled - target_scaled, axis=1) # Shape (4,)
        errors_px.extend(distances)
        
        # A sample is a complete success if all 4 corners fall within the threshold
        if np.all(distances <= threshold_px):
            successes += 1
            
    mean_err = float(np.mean(errors_px))
    success_rate = float(successes / N) * 100.0
    return mean_err, success_rate


@torch.no_grad()
def evaluate_model(model, dataset, approach_type, device) -> tuple:
    model.eval()
    all_preds = []
    all_targets = []
    
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    for batch in loader:
        inputs = batch['raw_photo'].to(device)
        targets = batch['corners'].numpy() # Shape (B, 4, 2)
        
        outputs = model(inputs)
        
        if approach_type == 'A': # Direct Coordinate Regression
            preds = outputs.cpu().numpy().reshape(-1, 4, 2)
        else: # Heatmap Regression
            preds = extract_coords_from_heatmaps(outputs)
            
        all_preds.append(preds)
        all_targets.append(targets)
        
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return calculate_metrics(all_preds, all_targets)


def main():
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # # 1. Initialize and load Approach A
    # model_a = DirectRegressionNet(in_channels=3).to(device)
    # path_a = 'checkpoints/corner_reg_best.pth'
    # if os.path.exists(path_a):
    #     # Load state dict (handle dict formats)
    #     state_dict = torch.load(path_a, map_location=device, weights_only=False)
    #     weights = state_dict['model_state_dict'] if 'model_state_dict' in state_dict else state_dict
        
    #     # FIX: Remap old layer 3 keys to new layer 4 keys to bypass sequential dropout shift
    #     if 'fc.3.weight' in weights:
    #         weights['fc.4.weight'] = weights.pop('fc.3.weight')
    #         weights['fc.4.bias'] = weights.pop('fc.3.bias')
            
    #     model_a.load_state_dict(weights)
    #     print("Approach A (Direct Regression) model loaded.")
    # else:
    #     print(f"Warning: Checkpoint not found at {path_a}")
        
    # 1. Initialize and load Approach A
    model_a = DirectRegressionNet(in_channels=3).to(device)
    path_a = 'checkpoints/corner_reg_best.pth'
    if os.path.exists(path_a):
        state_dict = torch.load(path_a, map_location=device, weights_only=False)
        weights = state_dict['model_state_dict'] if 'model_state_dict' in state_dict else state_dict
        
        # FIX: Bulletproof type-check for nn.Dropout presence in the model
        has_dropout_in_model = any(isinstance(layer, nn.Dropout) for layer in model_a.fc)
        
        # Bidirectional Key Remapper:
        if has_dropout_in_model and 'fc.3.weight' in weights:
            # Model HAS dropout, but checkpoint has NO dropout -> Map 3 to 4
            weights['fc.4.weight'] = weights.pop('fc.3.weight')
            weights['fc.4.bias'] = weights.pop('fc.3.bias')
            print("[Mapper] Remapped old fc.3 keys to new fc.4 keys (with dropout).")
            
        elif not has_dropout_in_model and 'fc.4.weight' in weights:
            # Model has NO dropout, but checkpoint HAS dropout -> Map 4 to 3
            weights['fc.3.weight'] = weights.pop('fc.4.weight')
            weights['fc.3.bias'] = weights.pop('fc.4.bias')
            print("[Mapper] Remapped checkpoint fc.4 keys back to fc.3 keys (no dropout).")
            
        model_a.load_state_dict(weights, strict=False)
        print("Approach A (Direct Regression) model loaded.")
    else:
        print(f"Warning: Checkpoint not found at {path_a}")

    # 2. Initialize and load Approach B
    model_b = HeatmapUNet(in_channels=3, out_channels=4).to(device)
    path_b = 'checkpoints/corner_heat_best.pth'
    if os.path.exists(path_b):
        # FIX: Safely load checkpoint with weights_only=True and handle wrapped dictionary structure
        checkpoint_b = torch.load(path_b, map_location=device, weights_only=True)
        if isinstance(checkpoint_b, dict) and 'model_state_dict' in checkpoint_b:
            model_b.load_state_dict(checkpoint_b['model_state_dict'])
            print(f"Approach B (Heatmap Regression) model loaded from checkpoint (Epoch {checkpoint_b.get('epoch', 'N/A')}).")
        else:
            model_b.load_state_dict(checkpoint_b)
            print("Approach B (Heatmap Regression) model loaded from raw state dict.")
    else:
        print(f"Warning: Checkpoint not found at {path_b}")

    # Load test sets
    _, _, test_ds_synthetic = get_train_val_test_datasets(
        raw_scans_dir="data/raw_scans",
        backgrounds_dir="data/backgrounds",
        target_size=(512, 512),
        epoch_length=100
    )
    test_ds_real = RealTestDataset(real_test_dir="data/real_test", target_size=(512, 512))

    print("\nRunning comparative evaluations...")
    
    # Evaluate Approach A (Direct Regression)
    err_a_syn, succ_a_syn = evaluate_model(model_a, test_ds_synthetic, 'A', device)
    err_a_real, succ_a_real = evaluate_model(model_a, test_ds_real, 'A', device)
    
    # Evaluate Approach B (Heatmap Regression)
    err_b_syn, succ_b_syn = evaluate_model(model_b, test_ds_synthetic, 'B', device)
    err_b_real, succ_b_real = evaluate_model(model_b, test_ds_real, 'B', device)
    
    # Print Comparative Performance Table
    print("\n### Corner Detection Performance Comparison Table")
    print("| Metric / Dataset Split | Approach A (Regression) | Approach B (Heatmaps) |")
    print("| :--- | :---: | :---: |")
    print(f"| **Synthetic Test Error (Mean px)** | {err_a_syn:.2f} px | {err_b_syn:.2f} px |")
    print(f"| **Synthetic Success Rate (<=10px)**| {succ_a_syn:.1f}% | {succ_b_syn:.1f}% |")
    print(f"| **Real Test Error (Mean px)**      | {err_a_real:.2f} px | {err_b_real:.2f} px |")
    print(f"| **Real Success Rate (<=10px)**     | {succ_a_real:.1f}% | {succ_b_real:.1f}% |")


if __name__ == "__main__":
    main()
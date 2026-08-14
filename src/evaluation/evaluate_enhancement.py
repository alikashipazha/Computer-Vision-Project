import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.dataset.loader import get_train_val_test_datasets, RealTestDataset, INV_IMAGE_TRANSFORM
from src.models.enhancement_model import CustomUNet
from src.utils.ocr_helper import OCRHelper

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) between [0, 255] images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Wrapper to compute SSIM using OpenCV/Manual formulation."""
    # Convert images to grayscale
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Calculate SSIM using OpenCV matchTemplate as variance approximation
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    g1 = g1.astype(np.float64)
    g2 = g2.astype(np.float64)
    
    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(g1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


@torch.no_grad()
def evaluate_synthetic_split(model, dataset, device) -> tuple:
    model.eval()
    psnr_scores = []
    ssim_scores = []
    
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    for batch in loader:
        inputs = batch['rectified_input'].to(device)
        targets = batch['rectified_target']
        
        preds = model(inputs).cpu()
        
        # Un-normalize inputs to original scale to compute metrics on [0, 255] RGB images
        for i in range(inputs.size(0)):
            # Convert tensors to OpenCV format
            pred_np = (preds[i].numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
            target_np = (targets[i].numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
            
            # Switch channels for metric correctness
            pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)
            target_bgr = cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR)
            
            psnr_scores.append(calculate_psnr(pred_bgr, target_bgr))
            ssim_scores.append(calculate_ssim(pred_bgr, target_bgr))
            
    return float(np.mean(psnr_scores)), float(np.mean(ssim_scores))


def compute_baseline(dataset) -> tuple:
    """Computes metrics on the raw degraded input itself relative to target."""
    psnr_scores = []
    ssim_scores = []
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    for batch in loader:
        # Revert standard normalization to match metric expectations
        inputs_normalized = batch['rectified_input']
        targets = batch['rectified_target']
        
        for i in range(inputs_normalized.size(0)):
            raw_unnorm = INV_IMAGE_TRANSFORM(inputs_normalized[i])
            input_np = (raw_unnorm.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
            target_np = (targets[i].numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
            
            input_bgr = cv2.cvtColor(input_np, cv2.COLOR_RGB2BGR)
            target_bgr = cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR)
            
            psnr_scores.append(calculate_psnr(input_bgr, target_bgr))
            ssim_scores.append(calculate_ssim(input_bgr, target_bgr))
            
    return float(np.mean(psnr_scores)), float(np.mean(ssim_scores))


def main():
    os.makedirs('docs/real_test_results', exist_ok=True)
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load trained model
    model = CustomUNet(in_channels=3, out_channels=3).to(device)
    checkpoint_path = 'checkpoints/enhancement_best.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}. Train the model first.")
        return
        
    # FIX: Safely load checkpoint with weights_only=True and handle wrapped dictionary structure
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully from checkpoint (Epoch {checkpoint.get('epoch', 'N/A')}).")
    else:
        model.load_state_dict(checkpoint)
        print("Model loaded successfully from raw state dict.")
    
    # 1. EVALUATE SYNTHETIC DATASETS
    train_ds, val_ds, test_ds = get_train_val_test_datasets(
        raw_scans_dir="data/raw_scans",
        backgrounds_dir="data/backgrounds",
        target_size=(512, 512),
        epoch_length=200
    )
    
    print("Evaluating synthetic metrics...")
    base_p, base_s = compute_baseline(test_ds)
    train_p, train_s = evaluate_synthetic_split(model, train_ds, device)
    val_p, val_s = evaluate_synthetic_split(model, val_ds, device)
    test_p, test_s = evaluate_synthetic_split(model, test_ds, device)
    
    # Print Markdown Table
    print("\n### Synthetic Splits Performance Table")
    print("| Split | PSNR (dB) | SSIM |")
    print("| :--- | :---: | :---: |")
    print(f"| No-Model Baseline (Test Split) | {base_p:.2f} | {base_s:.4f} |")
    print(f"| Training | {train_p:.2f} | {train_s:.4f} |")
    print(f"| Validation | {val_p:.2f} | {val_s:.4f} |")
    print(f"| Test | {test_p:.2f} | {test_s:.4f} |")
    
    # 2. EVALUATE REAL PHOTOS & OCR
    print("\nEvaluating real photos and OCR Readability...")
    real_ds = RealTestDataset(real_test_dir="data/real_test", target_size=(512, 512))
    ocr = OCRHelper()
    
    ocr_inputs, ocr_enhanced, ocr_refs = [], [], []
    
    model.eval()
    for idx in range(len(real_ds)):
        sample = real_ds[idx]
        raw_input_tensor = sample['rectified_input'].to(device).unsqueeze(0)
        
        with torch.no_grad():
            pred_tensor = model(raw_input_tensor).cpu().squeeze(0)
            
        # Revert normalization for input image
        input_unnorm = INV_IMAGE_TRANSFORM(sample['rectified_input'])
        input_np = (input_unnorm.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        input_bgr = cv2.cvtColor(input_np, cv2.COLOR_RGB2BGR)
        
        # Convert prediction to OpenCV image
        pred_np = (pred_tensor.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)
        
        # Convert reference scan to OpenCV image
        ref_np = (sample['rectified_target'].numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        ref_bgr = cv2.cvtColor(ref_np, cv2.COLOR_RGB2BGR)
        
        # Save qualitative side-by-side triplets
        triplet = np.hstack([input_bgr, pred_bgr, ref_bgr])
        cv2.imwrite(f"docs/real_test_results/triplet_{idx:02d}.jpg", triplet)
        
        # Measure OCR readability
        if ocr.available:
            ocr_inputs.append(ocr.get_ocr_confidence(input_bgr))
            ocr_enhanced.append(ocr.get_ocr_confidence(pred_bgr))
            ocr_refs.append(ocr.get_ocr_confidence(ref_bgr))
            
    if ocr.available:
        print("\n### OCR Word Confidence Scores (Average across real photos)")
        print(f"- Rectified Raw Photo Input: {np.mean(ocr_inputs):.2f}%")
        print(f"- Our Model Enhanced Output: {np.mean(ocr_enhanced):.2f}%")
        print(f"- Commercial CamScanner Reference: {np.mean(ocr_refs):.2f}%")
    else:
        print("\n[OCR Info] pytesseract is not installed or configured. Skipping OCR Readability calculation.")


if __name__ == "__main__":
    main()
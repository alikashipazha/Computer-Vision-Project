import os
import cv2
import numpy as np
import torch
from src.dataset.loader import RealTestDataset, INV_IMAGE_TRANSFORM, order_points
from src.inference.end_to_end_pipeline import EndToEndScannerPipeline
from src.inference.pipeline import EnhancementPipeline
from src.utils.ocr_helper import OCRHelper

def main():
    os.makedirs('docs/real_test_results', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize pipelines
    scanner = EndToEndScannerPipeline()
    enhancer_pipeline = EnhancementPipeline()
    ocr = OCRHelper()
    
    # Load real test dataset
    real_ds = RealTestDataset(real_test_dir="data/real_test", target_size=(512, 512))
    
    ocr_manual_rect = []
    ocr_auto_rect = []
    ocr_camscanner = []
    
    print("\nRunning End-to-End comparative evaluations on real photos...")
    for idx in range(len(real_ds)):
        sample = real_ds[idx]
        image_info = real_ds.images_info[real_ds.annotations[idx]['image_id']]
        file_name = image_info['file_name']
        
        # Load the original raw unresized photo
        raw_path = real_ds.real_test_dir / "images" / file_name
        from PIL import Image, ImageOps
        pil_img = Image.open(raw_path)
        pil_img = ImageOps.exif_transpose(pil_img)
        raw_photo = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Pipeline 1: Rectification using manual annotated corners
        annotated_corners = sample['corners'].numpy() * np.array([raw_photo.shape[1], raw_photo.shape[0]])
        ordered_annotated = order_points(annotated_corners)
        
        # Consistent Counter-Clockwise [TL, BL, BR, TR] target coordinates
        dst_pts = np.array([
            [0, 0],
            [0, 511],
            [511, 511],
            [511, 0]
        ], dtype=np.float32)
        
        # FIX: Cast inputs explicitly using .astype(np.float32) to prevent OpenCV assertion crashes
        M_manual = cv2.getPerspectiveTransform(ordered_annotated.astype(np.float32), dst_pts)
        rectified_manual = cv2.warpPerspective(raw_photo, M_manual, (512, 512))
        enhanced_manual = enhancer_pipeline.process_image(rectified_manual)
        
        # Pipeline 2: Fully automatic rectification using predicted corners
        # This chained measurement represents the full end-to-end scanner
        enhanced_auto = scanner.scan_document(raw_photo)

        # # Pipeline 1: Rectification using manual annotated corners (Strictly CCW: [TL, BL, BR, TR])
        # annotated_corners = sample['corners'].numpy() * np.array([raw_photo.shape[1], raw_photo.shape[0]])
        # ordered_annotated = order_points(annotated_corners)
        
        # # Consistent Counter-Clockwise [TL, BL, BR, TR] target coordinates for manual baseline
        # dst_pts_ccw = np.array([
        #     [0, 0],
        #     [0, 511],
        #     [511, 511],
        #     [511, 0]
        # ], dtype=np.float32)
        
        # M_manual = cv2.getPerspectiveTransform(ordered_annotated.astype(np.float32), dst_pts_ccw)
        # rectified_manual = cv2.warpPerspective(raw_photo, M_manual, (512, 512))
        # enhanced_manual = enhancer_pipeline.process_image(rectified_manual)
        
        # # Pipeline 2: Fully automatic rectification using predicted corners (Model Clockwise mapped inside scanner)
        # enhanced_auto = scanner.scan_document(raw_photo)
        
        # Pipeline 3: Commercial CamScanner Reference
        ref_path = real_ds.real_test_dir / "reference_scans" / file_name
        ref_scan = cv2.imread(str(ref_path))
        if ref_scan is not None:
            ref_scan_resized = cv2.resize(ref_scan, (512, 512))
        else:
            ref_scan_resized = np.zeros((512, 512, 3), dtype=np.uint8)
            
        # Save side-by-side comparative triplets
        # Columns: [Manual Rectified Output, Auto Predicted Scanner Output, CamScanner Scan]
        triplet = np.hstack([enhanced_manual, enhanced_auto, ref_scan_resized])
        cv2.imwrite(f"docs/real_test_results/end_to_end_{idx:02d}.jpg", triplet)
        
        # Run OCR Readability evaluations
        if ocr.available:
            ocr_manual_rect.append(ocr.get_ocr_confidence(enhanced_manual))
            ocr_auto_rect.append(ocr.get_ocr_confidence(enhanced_auto))
            ocr_camscanner.append(ocr.get_ocr_confidence(ref_scan_resized))
            
    # Output final comparative results
    print("\n### Automated Scanner Chain Performance Summary")
    print(f"- Total Real Test Documents Processed: {len(real_ds)}")
    
    if ocr.available:
        mean_manual = np.mean(ocr_manual_rect)
        mean_auto = np.mean(ocr_auto_rect)
        mean_cam = np.mean(ocr_camscanner)
        cost_of_error = mean_manual - mean_auto
        
        print("\n### OCR Readability Metric Comparison Table")
        print("| Pipeline Stage Configuration | OCR Average Word Confidence |")
        print("| :--- | :---: |")
        print(f"| **Manual Rectification + Enhancement** | {mean_manual:.2f}% |")
        print(f"| **Fully Automated Scanner Chain**     | {mean_auto:.2f}% |")
        print(f"| **Commercial CamScanner Reference**   | {mean_cam:.2f}% |")
        print(f"\n- **Cost of Corner Prediction Error:** {cost_of_error:.2f}% drop in downstream OCR legibility.")
    else:
        print("\n[OCR Info] pytesseract is not configured. Real-world end-to-end evaluation metrics skipped.")


if __name__ == "__main__":
    main()
# Convolutional Neural Network Applications: Document Scanning & Enhancement
**Course Project - Computer Vision | K. N. Toosi University of Technology**

This repository implements a fully automated, end-to-end document scanning and enhancement pipeline. The system consists of two decoupled deep learning tasks: a **Four-Corner Detection Network** and a **Document Enhancement Network**. By combining these networks, the scanner automatically localizes page boundaries in raw, uncropped smartphone photos, applies perspective rectification (homography warping), and photometrically cleans the flat crops (shadow removal, background whitening, and ink sharpening) to yield scan-quality outputs.

---

## Repository Structure

```text
.
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/
│   ├── Document Scanning Enhancement.pdf   # Project Requirements PDF
│   └── project_report.pdf                  # Completed Academic Project Report
├── src/
│   ├── dataset/
│   │   ├── generator.py                    # 6-step physical degradation generator
│   │   └── loader.py                       # PyTorch dataset loaders & preprocessors
│   ├── models/
│   │   ├── corner_models.py                # Direct Regression & Heatmap U-Net models
│   │   ├── enhancement_model.py            # Custom 4-level deep Enhancement U-Net
│   │   └── losses.py                       # Composite loss (L1 + SSIM + Sobel Edge)
│   ├── training/
│   │   ├── train_corner_heat.py            # Heatmap corner detector training script
│   │   ├── train_corner_reg.py             # Coordinate regression training script
│   │   └── train_enhancement.py            # Enhancement network training script
│   ├── evaluation/
│   │   ├── evaluate_corners.py             # Synthetic and real corner evaluation
│   │   ├── evaluate_end_to_end.py          # Chained automated scanner evaluation
│   │   └── evaluate_enhancement.py         # Enhancement PSNR/SSIM & OCR evaluation
│   ├── inference/
│   │   ├── corner_pipeline.py              # Corner detection inference pipeline
│   │   ├── end_to_end_pipeline.py          # Automated scanner end-to-end pipeline
│   │   └── pipeline.py                     # Document enhancement inference pipeline
│   └── utils/
│       └── ocr_helper.py                   # PyTesseract OCR utility wrapper
├── tests/
│   ├── visualize_check.py                  # Verifies dataset loader & splits
│   ├── visualize_corners.py                # Visualizes predicted corners on real photos
│   └── visualize_enhancement.py            # Visualizes enhanced outputs on real photos
├── requirements.txt                        # Python dependencies
├── LICENSE
└── .gitignore
```

---

## Installation & Setup

### 1. Prerequisites
- **NVIDIA GPU + CUDA:** Highly recommended for local training and inference.
- **Tesseract OCR Binary:** Required for reading legibility evaluations.
  - *Ubuntu:* `sudo apt install tesseract-ocr`
  - *MacOS:* `brew install tesseract`
  - *Windows:* Download the installer and ensure `tesseract` is added to your system's Environment Variables (PATH).

### 2. Environment Setup
Clone the repository and install the dependencies in your virtual environment:
```bash
git clone https://github.com/alikashipazha/Computer-Vision-Project.git
cd Computer-Vision-Project
pip install -r requirements.txt
```

---

## Dataset Layout

Create a folder named `data/` in the root directory (this folder is ignored by Git). Arrange your files in the following structure:
```text
data/
├── raw_scans/               # Place the 50 clean, flat source scans provided by TAs
├── backgrounds/             # Place diverse background images (tables, desks, carpets)
└── real_test/               # Real smartphone test photos
    ├── images/              # Place Roboflow-exported resized images (e.g., 20260803_xx.jpg)
    ├── annotations.json     # Place Roboflow-exported COCO Keypoints JSON file
    └── reference_scans/     # Place corresponding CamScanner template scans (aligned names)
```

---

## Usage Guide

Execute all commands from the root directory using the `-m` module-execution syntax.

### 1. Verification of Data Pipeline
Test the dynamic dataset loader and partition splits. It generates `test_preprocessing_alignment.jpg` showing the raw composite overlaid with target corners:
```bash
python -m tests.visualize_check
```

### 2. Training Models
All training scripts feature **Early Stopping** and **Automatic Checkpoint Resuming** (restoring model weights, optimizer momentum, epoch index, and minimum loss). 

- **Train the Enhancement Network:**
  ```bash
  python -m src.training.train_enhancement
  ```
- **Train the Direct Regression Corner Detector (Approach A):**
  ```bash
  python -m src.training.train_corner_reg
  ```
- **Train the Heatmap Corner Detector (Approach B):**
  ```bash
  python -m src.training.train_corner_heat
  ```
*Note: Checkpoints are saved under `checkpoints/` and training curves are plotted to `docs/`.*

### 3. Running Standalone Visualizations (Phase 2 & 3 Verification)
- **Visualize Corner Predictions on Real Photos (Approach B):**
  Generates `docs/corner_detection_visualization.jpg` with predicted keypoint circles:
  ```bash
  python -m tests.visualize_corners
  ```
- **Visualize Enhanced Crops on Real Photos:**
  Generates `docs/enhancement_visualization.jpg` showing a side-by-side comparison:
  ```bash
  python -m tests.visualize_enhancement
  ```

### 4. System Evaluation
- **Evaluate Enhancement Metrics:**
  Computes synthetic PSNR/SSIM, saves manual-rectification triplets, and measures OCR readability:
  ```bash
  python -m src.evaluation.evaluate_enhancement
  ```
- **Evaluate and Compare Corner Detectors:**
  Computes mean L2 pixel error and success rates ($\le 10$ px) for both Approach A and B:
  ```bash
  python -m src.evaluation.evaluate_corners
  ```
- **Evaluate the Full Automated Scanner Chain (End-to-End):**
  Chains predicted corner warping with the enhancement network. Saves comparative triplets under `docs/real_test_results/end_to_end_xx.jpg` and reports the downstream OCR cost of corner error:
  ```bash
  python -m src.evaluation.evaluate_end_to_end
  ```

---

## Docker Deployment

To deploy the workspace or run isolated pipelines via Docker:
```bash
# Build and run the container service
docker-compose -f docker/docker-compose.yml up --build
```

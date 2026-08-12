# Document Scanning and Enhancement: From Raw Photo to Clean Scan
**Course Project - Computer Vision**

### Student Information
- **Name:** Ali Kashi Pazha
- **Student ID:** 40121723
- **GitHub Repository:** [https://github.com/alikashipazha/Computer-Vision-Project](https://github.com/alikashipazha/Computer-Vision-Project)

### Academic Information
- **University:** K. N. Toosi University of Technology
- **Faculty:** Faculty of Computer Engineering
- **Course:** Computer Vision
- **Instructor:** Dr. Nasihatkon

---

## Abstract
This project implements an end-to-end document scanning and enhancement pipeline. It addresses the common computer vision task of image-to-image translation by converting degraded, shadowed, and perspective-distorted smartphone photographs of documents into crisp, uniform, scan-quality digital copies. The system is designed around two decoupled components: a **Four-Corner Detection Network** and a **Document Enhancement Network**. This report outlines the engineering design, mathematical foundations, and pipeline implementations developed during the first two phases of the project.

---

## 1. Phase 1: Dataset Preparation

Rather than manually annotating thousands of training images, this project leverages a data-centric deep learning paradigm: **generating a supervised dataset synthetically**. By mathematically transforming clean document scans onto arbitrary background images, we obtain high-fidelity training pairs along with pixel-perfect ground truth labels without human annotation.

### 1.1. Real-World Test Set and Annotation
A real-world test set of 10–15 diverse smartphone photographs was collected to evaluate the model's generalization capabilities on entirely unseen documents (such as hand-written notes, textbook pages, and printed homework).
- **Commercial Baseline:** For each test photo, a corresponding high-quality reference scan was acquired using a commercial scanning application (e.g., CamScanner). This serves as the target commercial baseline during final evaluations.
- **Keypoint Annotation:** The four corners of the document in each raw smartphone photo were manually labeled using Roboflow. A strict, consistent ordering of keypoints was maintained:
  1. **Top-Left (TL)**
  2. **Top-Right (TR)**
  3. **Bottom-Right (BR)**
  4. **Bottom-Left (BL)**

| Keypoint Index | Corner Name | Target Orientation | Consistent Ordering Rule |
| :---: | :--- | :---: | :--- |
| **1** | Top-Left (TL) | $(0, 0)$ | Must be the first annotated point across all samples |
| **2** | Top-Right (TR) | $(W-1, 0)$ | Must be the second annotated point |
| **3** | Bottom-Right (BR) | $(W-1, H-1)$ | Must be the third annotated point |
| **4** | Bottom-Left (BL) | $(0, H-1)$ | Must be the fourth annotated point |

Consistent ordering is mathematically critical to ensure the correctness of subsequent perspective rectification transforms.

### 1.2. Synthetic Training Data Generation
The synthetic training pair generator (`src/dataset/generator.py`) simulates the physical pipeline of taking a photograph of a document placed on a surface:

![ASCII flowchart](../ASCII_flowchart.jpg)
*Figure 1: Flowchart of the Synthetic Training Data Generation pipeline.*

1. **Clean Scan Selection:** A pristine, high-resolution document scan is selected as the ground truth target.
2. **Background Selection:** A random textured background image (e.g., table, desk, carpet) is loaded.
3. **Quadrilateral Generation:** A randomized convex quadrilateral is generated within the boundaries of the background image to represent the target document's position and camera perspective distortion.
4. **Homography Warping:** The perspective transform matrix $H \in \mathbb{R}^{3 \times 3}$ is calculated between the pristine scan corners and the chosen target points using:
   $$\mathbf{p}_{\text{target}} = H \mathbf{p}_{\text{source}}$$
   Using OpenCV's `cv2.getPerspectiveTransform` and `cv2.warpPerspective`, the clean scan is projected onto the background.
5. **Compositing:** A binary mask of the warped document is used to blend the scan seamlessly into the background canvas, producing the simulated raw smartphone photo.
6. **Rectification Mapping:** Because the forward transform matrix $H$ is known, the inverse transform $H^{-1}$ can be calculated to warp the simulated raw photo back into a perfectly aligned flat rectangle (Rectified Crop).

---

## 2. Phase 2: Preprocessing and Data Pipeline

Phase 2 focuses on converting raw and synthetic data into a structured format optimized for training neural networks, ensuring coordinate scale invariance, and enforcing robust validation splits.

### 2.1. Annotation Parsing and Resizing
The manually labeled test-set coordinates from Roboflow's COCO Keypoints JSON export are parsed dynamically. 
- **Resizing:** Images are resized to a standardized input resolution of $512 \times 512$ pixels.
- **Coordinate Scaling:** When raw images are resized, the absolute pixel coordinates of their corner labels must scale proportionally to preserve alignment. A resizing scale factor is applied to the keypoint arrays:
  $$x_{\text{new}} = x_{\text{old}} \times \left(\frac{W_{\text{target}}}{W_{\text{original}}}\right), \quad y_{\text{new}} = y_{\text{old}} \times \left(\frac{H_{\text{target}}}{H_{\text{original}}}\right)$$

### 2.2. Normalization and Resolution Independence
To make the learning tasks independent of the input resolution, two normalization steps are implemented:
- **Pixel Intensity Normalization:** Raw image pixel intensities are scaled from $[0, 255]$ to the range $[0.0, 1.0]$ by division by $255.0$. Standard ImageNet channel-wise mean and standard deviation normalization is applied for PyTorch model compatibility:
  $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
- **Keypoint Normalization:** Corner coordinates $(x, y)$ are normalized by dividing by the image's width and height, shifting the labels to a dimensionless coordinate space within the unit range $[0.0, 1.0]^2$.

### 2.3. No-Leakage Dataset Splitting Policy
To ensure rigorous performance tracking, a strict division of the dataset is enforced:
- **Split by Source Scan:** Splitting is performed strictly based on the unique identity of the pristine document scans rather than individual synthetic variations. This prevents **data leakage**, ensuring that degraded variations of the same underlying text page never cross-contaminate different splits.
- **Ratio:** The 50 clean source scans are divided into:
  - **Training Split (80%):** Used for backpropagation.
  - **Validation Split (10%):** Used for hyperparameter tuning and model checkpoint selection.
  - **Test Split (10%):** Held-out for honest synthetic evaluation.

### 2.4. Deterministic Freezing of Validation and Test Splits
Because the dataset generator runs **on-the-fly** (compositing random coordinates and degradations during every `__getitem__` call), a naive implementation would yield varying validation targets every epoch. This introduces noise into the validation metrics and impairs model selection.
- **Deterministic Seed Mechanism:** For validation and test splits, the pseudo-random generators (`random` and `numpy.random`) are seeded deterministically with the index of the requested sample:
  ```python
  if self.frozen:
      random.seed(idx)
      np.random.seed(idx)
  ```
  This ensures that the validation and test datasets are fully "frozen" and identical across all epochs, allowing stable, comparable scoring of model performance.

### 2.5. Data Pipeline Pipeline Verification
To verify the correct execution of the preprocessing pipeline, splitting logic, and coordinate normalization, the validation utility script (`src/dataset/visualize_check.py`) was executed.

**Execution Terminal Log Output:**
```bash
(cv-lab) PS E:\ComputerVision\Computer-Vision-Project> python -m src.dataset.visualize_check
Data split completed successfully (80/10/10).
Source Scans Shuffled & Split Summary:
  - Total Scans Located: 50
  - Training Subset: 40 scans
  - Validation Subset: 5 scans (Deterministic Seed Active)
  - Testing Subset: 5 scans (Deterministic Seed Active)
  
Extracting sample entry index 0 from Training subset...
Reverting PyTorch normalizations...
Mapping normalized coordinates to absolute pixels...
  - Corner 1 (TL) [Normalized]: [0.224, 0.185] -> Pixel: [114, 94]
  - Corner 2 (TR) [Normalized]: [0.781, 0.242] -> Pixel: [400, 124]
  - Corner 3 (BR) [Normalized]: [0.814, 0.821] -> Pixel: [416, 420]
  - Corner 4 (BL) [Normalized]: [0.156, 0.764] -> Pixel: [80, 391]

Generating test plot...
Visualization done. Check 'test_preprocessing_alignment.jpg' for correct alignment.
```

The resulting verification image plots the four normalized coordinates on top of the synthesized document:

![Preprocessing Alignment Check](../test_preprocessing_alignment.jpg)
*Figure 2: Synthesized document raw image with overlaid corner keypoints mapped from normalized target coordinates, demonstrating successful geometric alignment.*

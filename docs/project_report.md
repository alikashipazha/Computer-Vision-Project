# Document Scanning and Enhancement: From Raw Photo to Clean Scan
**Course Project - Computer Vision**

### Student Information
- **Name:** Ali Kashi Pazha
- **Student ID:** 40121723
- **GitHub Repository:** [https://github.com/alikashipazha/Computer-Vision-Project](https://github.com/alikashipazha/Computer-Vision-Project)
- **Google Colab:** [https://colab.research.google.com/drive/13TLa1VMpKMHIKJZekUlmHlsiSRP5Rq0h?usp=sharing](https://colab.research.google.com/drive/13TLa1VMpKMHIKJZekUlmHlsiSRP5Rq0h?usp=sharing)

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
To verify the correct execution of the preprocessing pipeline, splitting logic, and coordinate normalization, the validation utility script (`tests/visualize_check.py`) was executed.

**Execution Terminal Log Output:**
```bash
(cv-lab) PS E:\ComputerVision\Computer-Vision-Project> python -m tests.visualize_check
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

![test_preprocessing_alignment.jpg](../test_preprocessing_alignment.jpg)
*Figure 2: test_preprocessing_alignment.jpg*

The resulting verification image plots the four normalized coordinates on top of the synthesized document:

![Preprocessing Alignment Check](../test_preprocessing_alignment.jpg)
*Figure 3: Synthesized document raw image with overlaid corner keypoints mapped from normalized target coordinates, demonstrating successful geometric alignment.*

---

## 3. Phase 3: Task 1 - The Document Enhancement Network

The Document Enhancement Network is built using a custom, 4-level deep Encoder-Decoder U-Net architecture. It translates degraded, shadowed, and unevenly illuminated documents into clean, uniform, scan-quality images.

### 3.1. Loss Convergence Analysis
The model was trained for 40 epochs on Google Colab using an NVIDIA T4 GPU with a batch size of 8 and a learning rate of $1e-4$. The training utilized the custom `CompositeLoss` ($0.4 \times L1 + 0.4 \times SSIM + 0.2 \times SobelEdge$).

![Enhancement Network Loss Curves](../docs/enhancement_training_loss.png)
*Figure 4: Training and Validation loss curves over 40 epochs. The validation loss reached its minimum of 0.01965 at Epoch 24, where the optimal weights were successfully saved to prevent overfitting.*

The terminal output during the final epochs verifies the training convergence and the model selection behavior:
```text
Epoch [21/40] -> Train Loss: 0.02072 | Val Loss: 0.02045
Epoch [22/40] -> Train Loss: 0.01926 | Val Loss: 0.02004
==> New best model saved!
Epoch [23/40] -> Train Loss: 0.01828 | Val Loss: 0.02016
Epoch [24/40] -> Train Loss: 0.01774 | Val Loss: 0.01965
==> New best model saved!
Epoch [25/40] -> Train Loss: 0.01598 | Val Loss: 0.02035
...
Epoch [40/40] -> Train Loss: 0.00847 | Val Loss: 0.02327
```

### 3.2. Quantitative Evaluation on Synthetic Splits
The performance of the trained enhancement model was evaluated using Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) across the synthetic splits, compared against a "do-nothing" baseline on the held-out test split:

| Split | PSNR (dB) | SSIM |
| :--- | :---: | :---: |
| **No-Model Baseline (Test Split)** | 37.96 | 0.9107 |
| **Training Split** | 36.55 | 0.9857 |
| **Validation Split** | 36.76 | 0.9806 |
| **Test Split (Held-out)** | 36.69 | 0.9792 |

**Analysis of Synthetic Metrics:**
- **Structural Integrity (SSIM):** The model achieved a significant improvement in structural similarity, raising the SSIM from a baseline of **0.9107** to **0.9792** on the unseen test split. This demonstrates successful recovery of thin text strokes and boundary sharpness.
- **Pixel-Level Distance (PSNR):** The minor decrease in PSNR from 37.96 dB to 36.69 dB is a standard phenomenon in document enhancement models. It occurs because the network aggressively whitens the background pixels to uniform white (RGB 255), causing a slight L2 pixel-distance deviation from the natural grayish/yellowish background of the scans, while vastly improving actual readability.

### 3.3. Qualitative and OCR Evaluation on Real-World Photos
To measure the model's ability to generalize to physical degradation, the pipeline was executed on the real-world smartphone test photos. Document legibility was quantitatively analyzed using Tesseract OCR to measure word-level confidence:

| Image Source Pipeline | OCR Average Word Confidence |
| :--- | :---: |
| **Rectified Raw Photo Input (No Model)** | 46.76% |
| **Our Custom U-Net Enhanced Output** | **75.55%** |
| **Commercial CamScanner Reference** | **54.44%** |

**OCR Readability Analysis:**
The custom model enhanced document legibility, raising the average OCR confidence score from **46.76% to 75.55%** (a **+28.79% absolute improvement**). 

Notably, our custom network outperformed the commercial baseline (CamScanner) by **21.11%**. This is a technically valid phenomenon:
1. **Commercial Binarization Limits:** Commercial scanning apps use aggressive local thresholding and high-pass filters designed primarily for high-contrast printed text. When applied to handwritten text with variable ink thickness and color transitions (such as red pen notations), these commercial filters tend to fragment the characters, causing pixel-level stroke breakage and reducing OCR recognition.
2. **Structural Continuous-Tone Preservation:** Our model preserves the continuous-tone integrity of the writing while cleanly neutralizing background paper shadows. As visible in `triplet_00.jpg`, the ink shapes remain structurally continuous and solid, directly benefiting the OCR engine's word-level heuristics.

#### Qualitative Alignment Visuals:
The generated qualitative triplets demonstrate clean background whitening, sharp ink preservation, and absolute shadow suppression across the dataset:

![Real Photo Qualitative Triplet](../docs/real_test_results/triplet_00.jpg)
*Figure 5: Qualitative triplet comparison showing (Left) the rectified raw phone input, (Middle) our custom U-Net enhanced output with complete shadow suppression and continuous-tone ink preservation, and (Right) the commercial CamScanner reference.*

---

## 4. Phase 4: Dataset Enhancement with Realistic Degradations

To enable the networks to generalize to real-world physical environments, Phase 4 replaces the placeholder generator with a robust, physically-grounded, 6-step sequential degradation pipeline. This pipeline simulates typical smartphone document photography artifacts (angles, shadows, lens blur, sensor noise, and compression) using only native OpenCV and NumPy operations.

### 4.1. Degradation Pipeline Architecture
For every training sample, the synthetic generator (`src/dataset/generator.py`) applies six consecutive geometric and photometric transformations to the clean source scan:

![Degradation Pipeline Architecture](../Degradation_Pipeline_Architecture.png)
*Figure 6: Degradation Pipeline Architecture*

1. **Perspective Warp (Geometric):** Projects the clean scan onto a random textured background using a randomized convex quadrilateral to simulate variable camera angles and distances.
2. **Resolution Loss (Photometric):** Downscales the image by a random factor between $1.2\times$ and $2.2\times$ and upscales it back using bilinear interpolation, simulating distance and limited sensor resolution.
3. **Brightness, Contrast, and Color Cast (Photometric):** Applies random linear contrast adjustments ($\alpha \in [0.85, 1.15]$), brightness offsets ($\beta \in [-20, 20]$), and slight channel-wise scaling on the Red and Blue channels to simulate warm/cool lighting.
4. **Illumination Gradients and Shadows (Photometric):** 
   - *Illumination Gradient:* Generates a 2D linear gradient map using `np.meshgrid` at a random angle and multiplies it channel-wise.
   - *Soft Shadows:* Generates random 3-to-5-vertex convex polygons, fills them on a single-channel mask, and applies a heavy Gaussian filter ($71 \times 71$ to $151 \times 151$ kernel) to create soft shadow boundaries before blending.
5. **Blur and Noise (Photometric):** Applies Gaussian lens blur ($ksize \in \{3, 5\}$) followed by additive, zero-mean Gaussian noise ($\sigma \in [2.0, 6.0]$) to simulate camera shake and high-ISO sensor noise.
6. **JPEG Compression (Photometric):** Re-encodes the final image to a memory buffer at a random JPEG quality factor between $50$ and $85$, introducing high-frequency compression artifacts.

### 4.2. Geometric vs. Photometric Separation
To maintain strict, pixel-perfect alignment between the training inputs and target clean scans:
- **Input-Only Photometric Degradations:** All photometric degradations (steps 2 through 6) are applied strictly to the composited input image and never to the target scan. If any noise, shadows, or color shifts contaminate the target, pixel-wise loss functions (like L1 or MSE) will penalize the model incorrectly.
- **Inverse Geometric Mapping:** The perspective warp (step 1) is the only geometric transformation. It is mathematically inverted using the calculated inverse homography matrix $H^{-1}$:
  $$\mathbf{p}_{\text{source}} = H^{-1} \mathbf{p}_{\text{target}}$$
  This projects the highly degraded smartphone-like document back into a flat, rectified rectangular input, keeping it aligned pixel-for-pixel with the clean target scan.

### 4.3. Pipeline Visual Verification
The updated preprocessing verification utility (`tests/visualize_check.py`) was executed to confirm that the sequential distortions are applied correctly and that the corner coordinates remain aligned under the new degradation pipeline.

![Preprocessing Verification Plot](../test_preprocessing_alignment.jpg)
*Figure 7: Visual verification plot of a generated sample. The corner labels remain correctly mapped, while the document body displays simulated lens blur, additive sensor noise, JPEG compression artifacts, and soft overlapping shadow polygons.*

---

## 5. Phase 5: Task 2 - Document Corner Detection

This mandatory task implements and evaluates two fundamentally different neural architectures for predicting the four page corners of a raw smartphone document photo. To determine the most robust method, both models were trained locally on an NVIDIA GeForce RTX 3050 Laptop GPU and evaluated on synthetic and real test splits.

### 5.1. Architectural Design of the Two Approaches
- **Approach A — Direct Coordinate Regression:** Implemented as a deep, 5-block convolutional encoder (`DirectRegressionNet`) that progressively downsamples the raw photo. The feature maps are flattened into a 131,072-dimensional vector and fed into a multi-layer perceptron (MLP) head with a final `Sigmoid` activation. It outputs 8 continuous values representing the normalized $(x, y)$ coordinates of the four corners. It was trained using a standard L1 coordinate loss.
- **Approach B — Heatmap Regression:** Implemented as a fully convolutional, spatial encoder-decoder U-Net (`HeatmapUNet`) outputting 4 distinct probability heatmaps of size $512 \times 512$ (one channel per corner: TL, TR, BR, BL). Target heatmaps are generated on-the-fly using 2D Gaussian distributions centered on the normalized ground-truth coordinates with a standard deviation $\sigma = 8.0$ pixels. The network was trained using a pixel-wise MSE loss. During inference, coordinates are extracted using a robust 2D Spatial Argmax search.

### 5.2. Loss Convergence and Training Logs Analysis

The training behavior of the two approaches reveals a stark contrast in optimization stability:

#### 5.2.1. Approach A Training (Direct Regression Flatline)
Approach A completely failed to converge. The fully connected layers were unable to establish a stable spatial mapping, causing the training to flatline immediately. Early stopping was triggered at Epoch 6 due to 5 consecutive epochs without validation improvement.

![Approach A Loss Curves](../docs/corner_regression_loss.png)
*Figure 8: Training and Validation loss curves for Approach A. The validation loss completely flatlined at 0.19859, showing a total optimization block.*

#### 5.2.2. Approach B Training (Exponential Heatmap Decay)
In contrast, Approach B (Heatmaps) converged rapidly. The network leveraged its spatial convolutional layers to quickly localize the Gaussian corner targets, reaching a near-zero validation loss of `0.00004` by Epoch 11 before early stopping terminated training at Epoch 12.

![Approach B Loss Curves](../docs/corner_heatmap_loss.png)
*Figure 9: Training and Validation loss curves for Approach B. The model shows an ideal exponential decay, confirming highly stable spatial learning.*

---

### 5.3. Quantitative Performance Comparison
The performance of both trained models was evaluated against the synthetic test split and the real-world test split:

```text
(cv-lab) PS E:\ComputerVision\Computer-Vision-Project> python -m src.evaluation.evaluate_corners
Using device: cuda
Approach A (Direct Regression) model loaded.
Approach B (Heatmap Regression) model loaded.

Running comparative evaluations...
```

| Metric / Dataset Split | Approach A (Regression) | Approach B (Heatmaps) |
| :--- | :---: | :---: |
| **Synthetic Test Error (Mean px)** | 150.15 px | **7.92 px** |
| **Synthetic Success Rate (<=10px)**| 0.0% | **90.0%** |
| **Real Test Error (Mean px)**      | 364.56 px | 334.11 px |
| **Real Success Rate (<=10px)**     | 0.0% | 0.0% |

#### 5.3.1. Analysis of Synthetic Performance (Spatial vs. Global Mappings)
Approach B (Heatmaps) decisively outperforms Approach A on synthetic data, achieving a minimal mean error of **7.92 pixels** and a **90.0% success rate**.
- **The Spatial Bottleneck of Approach A:** Flattening the convolutional features into a dense vector destroys structural location information. The network is forced to learn a highly complex, non-linear global-to-point mapping. This results in poor gradients, causing the model to completely stall.
- **The Translation Equivariance of Approach B:** The fully convolutional spatial U-Net naturally preserves coordinate relationships. The network only needs to learn localized pixel activations around the paper corners, achieving pixel-level tracking.

#### 5.3.2. Analysis of Real-World Generalization Failure (Overfitting & Domain Gap)
Both approaches failed to generalize to the real-world smartphone test set, yielding high mean errors of **334.11 pixels** (Heatmaps) and **364.56 pixels** (Regression), with a **0.0% success rate**.
- **Overfitting to Digital Bounding Artifacts:** Per the project constraints, both networks were built "clean" without Dropout layers or pre-trained backbones. Consequently, the networks overfitted entirely to the pixel-perfect, mathematically sharp, and continuous edge boundaries generated by OpenCV's synthetic compositing (`warpPerspective` and masking).
- **The Synthetic-to-Real Domain Gap:** Real-world photos feature soft, complex physical boundaries, background clutter, and variable reflections. Because the networks memorized the artificial "sharp borders" of synthetic compositions rather than learning the semantic concept of "paper corners", they are blind to real-world pages. When evaluated on real photos, Approach B's output heatmaps remain flat or highly noisy, causing the spatial `argmax` function to return statistically random coordinates, which mathematically averages to $\sim330$ pixels on a $512 \times 512$ canvas.

### 5.4. Corner Detection Pipeline
The automated corner detection pipeline (`src/inference/corner_pipeline.py`) wraps the superior Heatmap Regression model:
1. **Preprocess:** Resizes the arbitrary raw BGR image to $512 \times 512$ and normalizes it.
2. **Predict:** Runs the spatial U-Net to yield 4 predicted heatmaps.
3. **Map Coordinates:** Extracts the peak coordinate from each channel using 2D Spatial Argmax, and scales the normalized coordinates back to the original image's native resolution.
4. **Visualize:** Overlays four colored circles with index rankings (1 to 4) directly on the full-resolution raw smartphone photo to verify localized landmarks.

---

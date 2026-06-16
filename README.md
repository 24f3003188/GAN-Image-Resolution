# Plant Leaves Super Resolution Challenge

## Overview

This project implements a high-performance image super-resolution pipeline for the Plant Leaves Super Resolution Challenge. The model learns to reconstruct high-resolution (128×128) plant leaf images from low-resolution (32×32) inputs using a two-stage ESRGAN-inspired training strategy.

The solution combines:

* RRDB (Residual-in-Residual Dense Blocks) Generator
* Multi-Scale GAN Discriminator
* VGG19 Perceptual Loss
* Feature Matching Loss
* Mixed Precision Training (AMP)
* Test-Time Augmentation (TTA)

The training process is divided into two phases:

1. Reconstruction Training
2. Adversarial Fine-Tuning

---

## Architecture

### Generator

The generator is based on the ESRGAN architecture and consists of:

* Initial convolution layer
* 23 RRDB blocks
* Residual trunk connection
* Two-stage upsampling
* Final reconstruction layer

**Input:** 32×32 RGB image

**Output:** 128×128 RGB image

### Discriminator

A Multi-Scale Discriminator is used to improve adversarial training stability.

Features:

* Spectral Normalization
* Instance Normalization
* Multi-scale image evaluation
* Intermediate feature extraction for Feature Matching Loss

---

## Dataset Structure

```text
plant-leaves-super-resolution-challenge/
│
├── train_High_Resolution/
├── train_Low_Resolution/
├── test_Low_Resolution/
└── vgg19_weights.pth
```

---

## Training Pipeline

### Phase 1: Reconstruction Training

**Objective**

```text
Loss = L1 + λ × Perceptual Loss
```

**Configuration**

| Parameter     | Value |
| ------------- | ----- |
| Epochs        | 100   |
| Learning Rate | 2e-4  |
| Batch Size    | 16    |
| Warmup Epochs | 5     |

**Loss Functions**

* L1 Reconstruction Loss
* VGG19 Perceptual Loss

**Goal**

Learn accurate image reconstruction before introducing adversarial objectives.

---

### Phase 2: Adversarial Fine-Tuning

**Objective**

```text
Loss = L1 + Perceptual + GAN + Feature Matching
```

**Configuration**

| Parameter        | Value |
| ---------------- | ----- |
| Epochs           | 60    |
| Generator LR     | 5e-5  |
| Discriminator LR | 1e-4  |
| Warmup Epochs    | 3     |

**Loss Functions**

* L1 Loss
* VGG19 Perceptual Loss
* GAN Loss (LSGAN)
* Feature Matching Loss

**Goal**

Improve texture quality, edge sharpness, and visual realism.

---

## Data Augmentation

The training pipeline includes:

* Horizontal flips
* Vertical flips
* Random 90° rotations
* Mild color jitter

These augmentations improve generalization and robustness.

---

## Perceptual Loss

A pretrained VGG19 network is used to compute feature-space similarity between generated and target images.

```text
Perceptual Loss =
L1(VGG(SR Image), VGG(HR Image))
```

This encourages perceptually realistic reconstructions beyond simple pixel-level similarity.

---

## Test-Time Augmentation (TTA)

Inference uses 8-way Test-Time Augmentation:

* Original image
* Horizontal flip
* Vertical flip
* Horizontal + Vertical flip
* 90° rotation
* Rotated variants with flips

Predictions are averaged to improve final reconstruction quality.

---

## Training Features

### Mixed Precision Training

Uses:

```python
torch.cuda.amp
```

Benefits:

* Faster training
* Lower GPU memory consumption

### Gradient Clipping

```python
max_norm = 1.0
```

Helps stabilize GAN training and prevent gradient explosion.

### Cosine Learning Rate Scheduling

Features:

* Warmup phase
* Cosine decay
* Minimum learning-rate floor

---

## Output Files

### Training Outputs

```text
generator_phase1.pth
generator_phase2.pth
training_curves.png
```

### Inference Outputs

```text
submission.csv
sample_prediction.png
```

---

## Submission Format

The generated submission file follows:

```csv
Id,Pixels
image_001,"..."
image_002,"..."
```

Each prediction is flattened into:

```text
128 × 128 × 3 = 49,152 values
```

stored as a space-separated pixel string.

---

## Evaluation Metric

Validation performance is monitored using:

```text
Mean Absolute Error (MAE)
```

Lower MAE indicates better reconstruction quality.

Additional tracked metrics:

* Generator Loss
* Discriminator Loss
* Validation MAE

---

## Requirements

```bash
pip install torch torchvision numpy pandas matplotlib pillow tqdm
```

Required libraries:

* torch
* torchvision
* numpy
* pandas
* matplotlib
* Pillow
* tqdm

---

## Running the Pipeline

### Training

Run all notebook cells sequentially to:

1. Load datasets
2. Build models
3. Train Phase 1
4. Train Phase 2
5. Save best-performing checkpoints

### Inference

The notebook automatically:

1. Loads the best checkpoint
2. Applies 8-way TTA
3. Generates super-resolved images
4. Creates `submission.csv`

---

## Techniques Used

* ESRGAN-style RRDB Generator
* Multi-Scale PatchGAN Discriminator
* Spectral Normalization
* VGG19 Perceptual Loss
* Feature Matching Loss
* Mixed Precision Training
* Cosine Learning Rate Scheduling
* Test-Time Augmentation
* Gradient Clipping
* Two-Stage Training Strategy

---

## Results

The model is designed to generate high-quality 128×128 plant leaf images from 32×32 inputs while balancing:

* Reconstruction accuracy
* Visual realism
* Training stability
* Generalization performance

The combination of perceptual learning, adversarial fine-tuning, and test-time augmentation provides strong performance for plant image super-resolution tasks.

Plant Leaves Super Resolution Challenge
Overview

This project implements a high-performance image super-resolution pipeline for the Plant Leaves Super Resolution Challenge. The model learns to reconstruct high-resolution (128×128) plant leaf images from low-resolution (32×32) inputs using a two-stage training strategy based on ESRGAN-style architecture.

The solution combines:

RRDB (Residual-in-Residual Dense Blocks) Generator
Multi-Scale GAN Discriminator
VGG19 Perceptual Loss
Feature Matching Loss
Mixed Precision Training (AMP)
Test-Time Augmentation (TTA)

The training process is divided into two phases:

Reconstruction Training
L1 Loss
Perceptual Loss
Adversarial Fine-Tuning
L1 Loss
Perceptual Loss
GAN Loss
Feature Matching Loss
Architecture
Generator

The generator is based on the ESRGAN architecture and consists of:

Initial convolution layer
23 RRDB blocks
Residual trunk connection
Two-stage upsampling
Final reconstruction layer

Input:

32 × 32 RGB image

Output:

128 × 128 RGB image
Discriminator

A Multi-Scale Discriminator is used to improve adversarial training stability.

Features:

Spectral normalization
Instance normalization
Multi-scale image evaluation
Feature extraction for Feature Matching Loss
Dataset Structure
plant-leaves-super-resolution-challenge/
│
├── train_High_Resolution/
├── train_Low_Resolution/
├── test_Low_Resolution/
└── vgg19_weights.pth
Training Pipeline
Phase 1: Reconstruction Training

Objective:

Loss = L1 + λ × Perceptual Loss

Configuration:

Parameter	Value
Epochs	100
Learning Rate	2e-4
Batch Size	16
Warmup Epochs	5

Losses:

L1 Reconstruction Loss
VGG19 Perceptual Loss

Goal:

Learn accurate image reconstruction before adversarial training.
Phase 2: Adversarial Fine-Tuning

Objective:

Loss =
L1 +
Perceptual +
GAN +
Feature Matching

Configuration:

Parameter	Value
Epochs	60
Generator LR	5e-5
Discriminator LR	1e-4
Warmup Epochs	3

Losses:

L1 Loss
VGG19 Perceptual Loss
GAN Loss (LSGAN)
Feature Matching Loss

Goal:

Improve texture realism and fine image details.
Data Augmentation

The training pipeline applies:

Horizontal flips
Vertical flips
Random 90° rotations
Mild color jitter

These augmentations improve generalization and robustness.

Perceptual Loss

The project uses a pretrained VGG19 network.

Feature maps from early VGG layers are extracted and compared using L1 loss:

Perceptual Loss =
L1(VGG(SR Image), VGG(HR Image))

This encourages visually realistic reconstructions.

Test-Time Augmentation (TTA)

Inference uses 8-way Test-Time Augmentation:

Original image
Horizontal flip
Vertical flip
Horizontal + Vertical flip
90° rotation
Rotated variants with flips

Predictions are averaged to improve final performance.

Training Features
Mixed Precision Training

Uses:

torch.cuda.amp

Benefits:

Faster training
Reduced GPU memory usage
Gradient Clipping
max_norm = 1.0

Prevents gradient explosion during GAN training.

Cosine Learning Rate Schedule

Features:

Warmup period
Cosine decay
Minimum learning rate floor
Output Files

During training:

generator_phase1.pth
generator_phase2.pth
training_curves.png

During inference:

submission.csv
sample_prediction.png
Submission Format

The generated submission file contains:

Id,Pixels
image_001,"..."
image_002,"..."

Each image is flattened into:

128 × 128 × 3 = 49,152 pixels

stored as a space-separated string.

Monitoring Metrics

Validation performance is monitored using:

Mean Absolute Error (MAE)

Lower MAE indicates better reconstruction quality.

Training also tracks:

Generator Loss
Discriminator Loss
Validation MAE
Requirements
torch
torchvision
numpy
pandas
matplotlib
Pillow
tqdm

Install dependencies:

pip install torch torchvision numpy pandas matplotlib pillow tqdm
Running the Pipeline
Train

Run all notebook cells sequentially to:

Load data
Build models
Train Phase 1
Train Phase 2
Save best checkpoints
Inference

The notebook automatically:

Loads the best checkpoint
Performs 8-way TTA
Generates super-resolved images
Creates submission.csv
Key Techniques Used
ESRGAN-inspired RRDB Generator
Multi-Scale PatchGAN Discriminator
Spectral Normalization
VGG19 Perceptual Loss
Feature Matching Loss
Mixed Precision Training
Cosine LR Scheduling
Test-Time Augmentation
Gradient Clipping
Two-Stage Training Strategy
Results

The model is designed to generate high-quality 128×128 plant leaf images from 32×32 inputs while balancing:

Reconstruction accuracy
Visual realism
Training stability
Generalization performance

The combination of perceptual learning, adversarial fine-tuning, and test-time augmentation provides strong performance on plant image super-resolution tasks.



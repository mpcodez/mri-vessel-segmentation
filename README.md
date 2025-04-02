# Automated Vessel Segmentation in MRI Data

## Overview
This project implements a deep learning model for the automated segmentation of blood vessels in 3D MRI scans. Manual segmentation of vessels is a time-consuming and inconsistent process. Our approach leverages a U-Net architecture to enhance segmentation accuracy and efficiency, making this task scalable for clinical and research applications.

## Authors
- **Medha Pappula** – Thomas Jefferson High School for Science and Technology
- **Kade Yen** – Thomas Jefferson High School for Science and Technology
- **Instructor:** Dr. Yilmaz

## Problem Statement
Identifying blood vessels in MRI scans is critical for diagnosing neurovascular conditions. However, manual segmentation is labor-intensive, requiring specialized expertise. Our automated model seeks to:
- Improve segmentation accuracy.
- Reduce the time and cost of manual vessel delineation.
- Provide a scalable solution for large-scale neurovascular studies.

## Methodology
### **Data Processing**
- **Dataset**: 600 MRI scans in `.nii` format, converted into 64-voxel cube sub-volumes for efficiency.
- **Features Extracted**: Vessel density, radius, tortuosity, and branching patterns.
- **Data Split**: 80% training, 20% validation; training data further divided (80% training, 20% testing).

### **Model Architecture**
- **Baseline Models**: Compared U-Net against CNN and FCNN implementations.
- **Chosen Model**: **U-Net** – optimized for medical image segmentation.
- **Training Details**:
  - **Epochs**: 100,000 steps
  - **Batch Size**: 1
  - **Learning Rate**: 0.001
  - **Hardware**: Personal GPU setup

### **Evaluation Metrics**
- **Dice Similarity Coefficient (DSC)**
- **False Positive Rate (FPR)**
- **False Negative Rate (FNR)**

| Model        | Dice Score | FPR   | FNR   |
|-------------|-----------|-------|-------|
| Chollet et al. | 0.55  | 0.2   | 0.5   |
| **Our U-Net** | **0.681** | **0.194** | **0.289** |

## Results & Discussion
- **Strengths**: Outperformed previous methods with a higher Dice score and lower false classification rates.
- **Challenges**: False positives created small artifacts; false negatives missed smaller vessels.
- **Future Work**: Further hyperparameter tuning and data augmentation strategies.

## Installation & Usage
### **Prerequisites**
- Python 3.8+
- TensorFlow / PyTorch
- Numpy, Pandas, Scikit-Learn
- Nibabel for `.nii` format handling

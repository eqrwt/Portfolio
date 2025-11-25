# Portfolio

Welcome to my technical portfolio. This repository contains projects demonstrating expertise in 3D Computer Vision, LiDAR processing, and Deep Learning.

## Projects Overview

This repository is organized into branches, each containing a distinct project:

### 1. 3D LiDAR Object Detection (SFA3D)
**Branch:** `main` (Current)

Implementation of **SFA3D (Super Fast and Accurate 3D Object Detection)**. This project focuses on detecting objects in 3D space using LiDAR point clouds and monocular camera data.

**Key Features:**
- 3D Object Detection using Deep Learning
- Monocular 3D detection pipeline
- Visualization of detection results

**Key Files:**
- `sfa_model.ipynb`: Jupyter notebook demonstrating the model architecture and training/inference.
- `SFA3D-Monocular-Detection/`: Source code for the detection framework.

---

### 2. LiDAR-Camera Calibration Tool
**Branch:** [`calibration-folder`](https://github.com/eqrwt/Portfolio/tree/calibration-folder)

A comprehensive toolkit for calibrating LiDAR and Camera systems. This project provides tools to calculate the transformation between LiDAR and Camera coordinate systems and project 3D points onto 2D images.

**Key Features:**
- **Manual Calibration Tool**: Interactive GUI to select corresponding points in Image and LiDAR views.
- **Projection Analysis**: Tools to verify calibration quality by projecting LiDAR points onto images.
- **Density Visualization**: Heatmaps showing point cloud density on the image plane.
- **Ground Truth Comparison**: Compare custom calibration against KITTI ground truth.

**Key Files:**
- `calib_tool.py`: Interactive calibration tool.
- `analyze_projection.py`: Analysis script for projection quality.
- `calibration/README_calibration_usage.md`: Detailed usage guide for the calibration tools.

## Getting Started

To explore a specific project, switch to the corresponding branch:

```bash
# Clone the repository
git clone https://github.com/eqrwt/Portfolio.git
cd Portfolio

# To view the Calibration Tools
git checkout calibration-folder

# To view the SFA3D Project
git checkout main
```

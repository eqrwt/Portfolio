# LiDAR-Camera Calibration Usage Guide

## Overview
This guide explains how to use your calibration file (`calib.txt`) to project 3D LiDAR points onto 2D camera images.

## Files Created

### 1. `calib.txt` - Your Calibration File
Contains the rotation matrix (R) and translation vector (T) that transform LiDAR coordinates to camera coordinates.

### 2. Generated Images
- `projection_result.jpg` - LiDAR points projected using your calibration
- `kitti_projection.jpg` - LiDAR points projected using KITTI ground truth (for comparison)
- `density_visualization.jpg` - Heat map showing point density distribution

## How to Use Your Calibration

### Basic Usage
```python
import numpy as np
import cv2

# Load your calibration
def load_calibration(calib_file):
    R = None
    T = None
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    # ... parsing code ...
    return R, T

# Project LiDAR points to image
def project_lidar_to_image(lidar_points, R, T, K):
    # Transform: P_cam = R * P_lidar + T
    camera_points = (R @ lidar_points.T) + T
    
    # Filter points in front of camera
    valid_mask = camera_points[2, :] > 0
    
    # Project to image plane
    normalized_points = camera_points[:, valid_mask] / camera_points[2, valid_mask]
    image_points_homogeneous = K @ normalized_points
    image_points = image_points_homogeneous[:2, :].T
    
    return image_points, valid_mask
```

### Camera Intrinsic Matrix
Use this matrix for the KITTI dataset:
```python
K = np.array([[721.5377, 0, 609.5593],
              [0, 721.5377, 172.8540],
              [0, 0, 1]])
```

## Testing Your Calibration

### 1. Quick Test
```bash
python test_calibration.py
```
This will:
- Project LiDAR points using your calibration
- Compare with KITTI ground truth
- Save visualization images

### 2. Detailed Analysis
```bash
python analyze_projection.py
```
This provides:
- Point distribution analysis
- Depth statistics
- Coverage metrics
- Quality assessment

### 3. Comparison with Ground Truth
```bash
python compare_calibration.py
```
This compares your calibration with KITTI ground truth and shows:
- Rotation and translation errors
- Your calibration in KITTI format

## Analysis Results

### Your Calibration Performance:
- **Points projected**: 4,416 out of 115,384 LiDAR points (3.8%)
- **Coverage**: 27.6% of image area
- **Point distribution**: Reasonable (5.1 pixels average distance)
- **Depth range**: -2.35m to 2.67m (some negative depths indicate issues)

### Issues Identified:
1. **Low projection rate**: Only 3.8% of LiDAR points project to image
2. **Negative depths**: Some points appear behind camera
3. **Large errors**: 68.62° rotation error, 11.616m translation error vs KITTI

## Improving Your Calibration

### 1. Better Point Selection
- Choose more distinctive features (corners, edges)
- Select points across the entire image
- Ensure points are at different depths
- Use at least 8-10 points for better accuracy

### 2. Coordinate System Check
- Verify LiDAR coordinate system (usually: X=forward, Y=left, Z=up)
- Check camera coordinate system (usually: X=right, Y=down, Z=forward)
- Ensure consistent units (meters)

### 3. Camera Intrinsics
- Use the correct camera matrix for your dataset
- Verify focal length and principal point
- Check for lens distortion

### 4. Manual vs Automatic
- Consider using automatic calibration tools
- Use multiple images for better accuracy
- Implement RANSAC for robust estimation

## Visual Assessment

### Good Calibration Indicators:
- Points align with image features (buildings, roads, objects)
- Even distribution across image
- Reasonable depth values (positive, realistic ranges)
- Points follow expected perspective (closer objects appear larger)

### Poor Calibration Indicators:
- Points don't align with image features
- Clustered in small areas
- Negative or unrealistic depths
- Points outside expected image bounds

## Next Steps

1. **Visual inspection**: Check the generated images to see if points align with features
2. **Improve point selection**: Re-run calibration with better point choices
3. **Check coordinate systems**: Verify your assumptions about sensor orientations
4. **Use ground truth**: Compare with known good calibrations when available
5. **Iterate**: Calibration is often an iterative process

## Troubleshooting

### No points projected:
- Check coordinate system definitions
- Verify camera intrinsics
- Ensure LiDAR points are in correct units

### Points in wrong locations:
- Check rotation matrix signs
- Verify translation direction
- Confirm coordinate frame conventions

### Poor accuracy:
- Use more calibration points
- Choose better distributed points
- Consider automatic calibration methods

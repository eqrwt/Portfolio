import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import glob

def load_velodyne_bin(bin_path):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]

def load_calibration(calib_file):
    """Load calibration from calib.txt file"""
    R = None
    T = None
    
    with open(calib_file, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if line.strip() == 'R:':
            R = np.array([
                [float(x) for x in lines[i+1].strip().strip('[]').split()],
                [float(x) for x in lines[i+2].strip().strip('[]').split()],
                [float(x) for x in lines[i+3].strip().strip('[]').split()]
            ])
        elif line.strip() == 'T:':
            T = np.array([
                [float(lines[i+1].strip().strip('[]'))],
                [float(lines[i+2].strip().strip('[]'))],
                [float(lines[i+3].strip().strip('[]'))]
            ])
    
    return R, T

def project_lidar_to_image(lidar_points, R, T, K):
    """Project 3D LiDAR points to 2D image coordinates"""
    camera_points = (R @ lidar_points.T) + T
    valid_mask = camera_points[2, :] > 0
    normalized_points = camera_points[:, valid_mask] / camera_points[2, valid_mask]
    image_points_homogeneous = K @ normalized_points
    image_points = image_points_homogeneous[:2, :].T
    return image_points, valid_mask

def analyze_projection_quality(image_path, lidar_path, calib_file):
    """Analyze the quality of LiDAR projection"""
    
    # Load data
    img = cv2.imread(image_path)
    lidar_points = load_velodyne_bin(lidar_path)
    R, T = load_calibration(calib_file)
    
    # Camera intrinsic matrix
    K = np.array([[721.5377, 0, 609.5593],
                  [0, 721.5377, 172.8540],
                  [0, 0, 1]])
    
    # Project LiDAR points
    image_points, valid_mask = project_lidar_to_image(lidar_points, R, T, K)
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Filter points within image bounds
    in_bounds = (image_points[:, 0] >= 0) & (image_points[:, 0] < width) & \
                (image_points[:, 1] >= 0) & (image_points[:, 1] < height)
    
    final_points = image_points[in_bounds]
    final_lidar_points = lidar_points[valid_mask][in_bounds]
    
    print("=== PROJECTION QUALITY ANALYSIS ===")
    print(f"Total LiDAR points: {len(lidar_points):,}")
    print(f"Points in front of camera: {np.sum(valid_mask):,} ({np.sum(valid_mask)/len(lidar_points)*100:.1f}%)")
    print(f"Points within image bounds: {len(final_points):,} ({len(final_points)/len(lidar_points)*100:.1f}%)")
    
    # Analyze point distribution
    if len(final_points) > 0:
        print(f"\nPoint distribution in image:")
        print(f"  X range: {final_points[:, 0].min():.1f} to {final_points[:, 0].max():.1f}")
        print(f"  Y range: {final_points[:, 1].min():.1f} to {final_points[:, 1].max():.1f}")
        
        # Analyze depth distribution
        depths = final_lidar_points[:, 2]  # Z coordinate as depth
        print(f"\nDepth analysis:")
        print(f"  Min depth: {depths.min():.2f}m")
        print(f"  Max depth: {depths.max():.2f}m")
        print(f"  Mean depth: {depths.mean():.2f}m")
        print(f"  Median depth: {np.median(depths):.2f}m")
        
        # Check for reasonable depth values
        reasonable_depth = (depths > 0) & (depths < 100)  # 0-100m range
        print(f"  Points with reasonable depth (0-100m): {np.sum(reasonable_depth):,} ({np.sum(reasonable_depth)/len(depths)*100:.1f}%)")
        
        # Analyze point density
        print(f"\nPoint density analysis:")
        # Create a grid to count points
        grid_size = 50
        x_bins = np.linspace(0, width, grid_size)
        y_bins = np.linspace(0, height, grid_size)
        
        hist, _, _ = np.histogram2d(final_points[:, 1], final_points[:, 0], bins=[y_bins, x_bins])
        non_empty_cells = np.sum(hist > 0)
        total_cells = grid_size * grid_size
        
        print(f"  Image coverage: {non_empty_cells}/{total_cells} cells ({non_empty_cells/total_cells*100:.1f}%)")
        print(f"  Average points per non-empty cell: {np.mean(hist[hist > 0]):.1f}")
        print(f"  Max points in a cell: {np.max(hist):.0f}")
        
        # Check for clustering
        if len(final_points) > 100:
            # Sample points for clustering analysis
            sample_indices = np.random.choice(len(final_points), min(1000, len(final_points)), replace=False)
            sample_points = final_points[sample_indices]
            
            # Calculate average distance to nearest neighbor
            from scipy.spatial.distance import cdist
            distances = cdist(sample_points, sample_points)
            np.fill_diagonal(distances, np.inf)  # Exclude self
            min_distances = np.min(distances, axis=1)
            avg_min_distance = np.mean(min_distances)
            
            print(f"  Average distance to nearest neighbor: {avg_min_distance:.1f} pixels")
            
            if avg_min_distance < 5:
                print("  ⚠️  Points are very clustered (might indicate poor calibration)")
            elif avg_min_distance > 50:
                print("  ⚠️  Points are very sparse (might indicate poor calibration)")
            else:
                print("  ✓ Point distribution looks reasonable")
    
    return final_points, final_lidar_points

def create_density_visualization(image_path, lidar_path, calib_file, show=True, output_path="density_visualization.jpg"):
    """Create a density visualization of projected points

    Args:
        image_path: path to image file
        lidar_path: path to lidar .bin file
        calib_file: path to calib.txt file (R/T)
        show: whether to open a GUI window to display result
        output_path: file path to save visualization image
    """
    
    # Load data
    img = cv2.imread(image_path)
    lidar_points = load_velodyne_bin(lidar_path)
    R, T = load_calibration(calib_file)
    
    # Camera intrinsic matrix
    K = np.array([[721.5377, 0, 609.5593],
                  [0, 721.5377, 172.8540],
                  [0, 0, 1]])
    
    # Project LiDAR points
    image_points, valid_mask = project_lidar_to_image(lidar_points, R, T, K)
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Filter points within image bounds
    in_bounds = (image_points[:, 0] >= 0) & (image_points[:, 0] < width) & \
                (image_points[:, 1] >= 0) & (image_points[:, 1] < height)
    
    final_points = image_points[in_bounds]
    
    # Create density map
    density_map = np.zeros((height, width), dtype=np.uint8)
    
    for point in final_points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < width and 0 <= y < height:
            density_map[y, x] += 1
    
    # Normalize density map for visualization
    max_density = np.max(density_map)
    if max_density > 0:
        density_map_normalized = (density_map * 255 / max_density).astype(np.uint8)
    else:
        density_map_normalized = density_map
    
    # Apply colormap
    density_colored = cv2.applyColorMap(density_map_normalized, cv2.COLORMAP_JET)
    
    # Blend with original image
    alpha = 0.7
    blended = cv2.addWeighted(img, 1-alpha, density_colored, alpha, 0)
    
    # Add text overlay
    cv2.putText(blended, f"LiDAR Density Map - {len(final_points)} points", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(blended, f"Max density: {max_density} points/pixel", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save and optionally display
    cv2.imwrite(output_path, blended)
    print(f"Density visualization saved as '{output_path}'")
    
    if show:
        cv2.imshow("LiDAR Density Map", blended)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def run_single(image_path: str, lidar_path: str, calib_file: str, show: bool):
    print(f"\n=== Analyzing ===\nImage: {image_path}\nLiDAR: {lidar_path}\nCalib: {calib_file}")
    points, _ = analyze_projection_quality(image_path, lidar_path, calib_file)
    print("\nCreating density visualization...")
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = f"density_visualization_{stem}.jpg"
    create_density_visualization(image_path, lidar_path, calib_file, show=show, output_path=out_path)
    
    print("\n=== SUMMARY ===")
    if len(points) > 0:
        print("✓ Calibration projects LiDAR points to the image")
        print(f"  - {len(points):,} points projected")
        print("  - Check the saved image for alignment quality")
    else:
        print("✗ No points projected to the image")
        print("  - Check coordinate systems and calibration parameters")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze LiDAR->Image projection quality")
    parser.add_argument("--image", type=str, help="Path to image file (.png)")
    parser.add_argument("--lidar", type=str, help="Path to lidar file (.bin)")
    parser.add_argument("--calib", type=str, default="calib.txt", help="Path to calibration file (R/T)")
    parser.add_argument("--auto", action="store_true", help="Auto-detect frame pairs in current directory")
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI windows; save images only")
    args = parser.parse_args()

    show_gui = not args.no_gui

    if args.auto:
        # Find pairs like ######.png with matching ######.bin
        pngs = sorted(glob.glob("*.png"))
        pairs = []
        for p in pngs:
            stem, _ = os.path.splitext(p)
            b = f"{stem}.bin"
            if os.path.exists(b):
                pairs.append((p, b))
        if not pairs:
            print("No image/LiDAR pairs found in current directory.")
        for img_path, bin_path in pairs:
            run_single(img_path, bin_path, args.calib, show_gui)
    else:
        image_path = args.image if args.image else "000010.png"
        lidar_path = args.lidar if args.lidar else "000010.bin"
        run_single(image_path, lidar_path, args.calib, show_gui)

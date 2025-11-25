import numpy as np
import cv2
import matplotlib.pyplot as plt

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
            # Read the next 3 lines as rotation matrix
            R = np.array([
                [float(x) for x in lines[i+1].strip().strip('[]').split()],
                [float(x) for x in lines[i+2].strip().strip('[]').split()],
                [float(x) for x in lines[i+3].strip().strip('[]').split()]
            ])
        elif line.strip() == 'T:':
            # Read the next 3 lines as translation vector
            T = np.array([
                [float(lines[i+1].strip().strip('[]'))],
                [float(lines[i+2].strip().strip('[]'))],
                [float(lines[i+3].strip().strip('[]'))]
            ])
    
    return R, T

def project_lidar_to_image(lidar_points, R, T, K):
    """
    Project 3D LiDAR points to 2D image coordinates
    
    Args:
        lidar_points: Nx3 array of 3D points in LiDAR coordinates
        R: 3x3 rotation matrix
        T: 3x1 translation vector
        K: 3x3 camera intrinsic matrix
    
    Returns:
        image_points: Nx2 array of 2D image coordinates
        valid_mask: boolean array indicating which points are valid
    """
    # Transform LiDAR points to camera coordinates
    # P_cam = R * P_lidar + T
    camera_points = (R @ lidar_points.T) + T
    
    # Filter points in front of camera (positive Z)
    valid_mask = camera_points[2, :] > 0
    
    # Project to image plane
    # Normalize by Z coordinate
    normalized_points = camera_points[:, valid_mask] / camera_points[2, valid_mask]
    
    # Apply camera intrinsics
    image_points_homogeneous = K @ normalized_points
    image_points = image_points_homogeneous[:2, :].T
    
    return image_points, valid_mask

def visualize_projection(image_path, lidar_path, calib_file):
    """Visualize LiDAR points projected onto the image"""
    
    # Load data
    img = cv2.imread(image_path)
    lidar_points = load_velodyne_bin(lidar_path)
    R, T = load_calibration(calib_file)
    
    print("Loaded calibration:")
    print("R:", R)
    print("T:", T)
    
    # Camera intrinsic matrix (from your script)
    K = np.array([[721.5377, 0, 609.5593],
                  [0, 721.5377, 172.8540],
                  [0, 0, 1]])
    
    # Project LiDAR points to image
    image_points, valid_mask = project_lidar_to_image(lidar_points, R, T, K)
    
    print(f"Total LiDAR points: {len(lidar_points)}")
    print(f"Valid projected points: {np.sum(valid_mask)}")
    print(f"Points in image bounds: {len(image_points)}")
    
    # Filter points within image bounds
    height, width = img.shape[:2]
    in_bounds = (image_points[:, 0] >= 0) & (image_points[:, 0] < width) & \
                (image_points[:, 1] >= 0) & (image_points[:, 1] < height)
    
    final_points = image_points[in_bounds]
    print(f"Points within image bounds: {np.sum(in_bounds)}")
    
    # Create visualization
    img_vis = img.copy()
    
    # Draw projected points
    for point in final_points:
        x, y = int(point[0]), int(point[1])
        cv2.circle(img_vis, (x, y), 1, (0, 255, 0), -1)  # Green dots
    
    # Add text overlay
    cv2.putText(img_vis, f"Projected points: {len(final_points)}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img_vis, f"Total LiDAR: {len(lidar_points)}", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Save and display
    cv2.imwrite("projection_result.jpg", img_vis)
    print("Projection result saved as 'projection_result.jpg'")
    
    # Display
    cv2.imshow("LiDAR Projection", img_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return final_points

def compare_with_kitti_projection(image_path, lidar_path):
    """Compare your calibration with KITTI ground truth"""
    
    # KITTI ground truth calibration
    R_kitti = np.array([
        [6.927964000000e-03, -9.999722000000e-01, -2.757829000000e-03],
        [-1.162982000000e-03, 2.749836000000e-03, -9.999955000000e-01],
        [9.999753000000e-01, 6.931141000000e-03, -1.143899000000e-03]
    ])
    
    T_kitti = np.array([[-2.457729000000e-02], [-6.127237000000e-02], [-3.321029000000e-01]])
    
    # Load data
    img = cv2.imread(image_path)
    lidar_points = load_velodyne_bin(lidar_path)
    
    # Camera intrinsic matrix (KITTI format)
    K = np.array([[721.5377, 0, 609.5593],
                  [0, 721.5377, 172.8540],
                  [0, 0, 1]])
    
    # Project using KITTI calibration
    image_points_kitti, valid_mask_kitti = project_lidar_to_image(lidar_points, R_kitti, T_kitti, K)
    
    # Filter points within image bounds
    height, width = img.shape[:2]
    in_bounds_kitti = (image_points_kitti[:, 0] >= 0) & (image_points_kitti[:, 0] < width) & \
                     (image_points_kitti[:, 1] >= 0) & (image_points_kitti[:, 1] < height)
    
    final_points_kitti = image_points_kitti[in_bounds_kitti]
    
    # Create comparison visualization
    img_comp = img.copy()
    
    # Draw KITTI projected points (blue)
    for point in final_points_kitti:
        x, y = int(point[0]), int(point[1])
        cv2.circle(img_comp, (x, y), 1, (255, 0, 0), -1)  # Blue dots
    
    # Add text overlay
    cv2.putText(img_comp, f"KITTI points: {len(final_points_kitti)}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Save and display
    cv2.imwrite("kitti_projection.jpg", img_comp)
    print("KITTI projection result saved as 'kitti_projection.jpg'")
    
    # Display
    cv2.imshow("KITTI Projection", img_comp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return final_points_kitti

if __name__ == "__main__":
    image_path = "000010.png"
    lidar_path = "000010.bin"
    calib_file = "calib.txt"
    
    print("=== TESTING YOUR CALIBRATION ===")
    print("1. Testing your manual calibration...")
    your_points = visualize_projection(image_path, lidar_path, calib_file)
    
    print("\n2. Testing KITTI ground truth calibration...")
    kitti_points = compare_with_kitti_projection(image_path, lidar_path)
    
    print(f"\n=== COMPARISON RESULTS ===")
    print(f"Your calibration projected {len(your_points)} points")
    print(f"KITTI calibration projected {len(kitti_points)} points")
    
    if len(your_points) > 0:
        print("✓ Your calibration successfully projects points to the image")
    else:
        print("✗ Your calibration failed to project points to the image")
    
    print("\nCheck the saved images to visually compare the projections!")

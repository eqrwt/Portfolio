import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_velodyne_bin(bin_path):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]

image_path = "000000.png"
lidar_path = "000000.bin"

img = cv2.imread(image_path)
lidar_points = load_velodyne_bin(lidar_path)

print("=== SIMPLE CALIBRATION TOOL ===")
print("This tool will guide you to select specific types of points")
print("We'll focus on the most reliable calibration points")
print("=====================================\n")

picked_pixels = []
picked_lidar_points = []

# Define point types and their descriptions
point_types = [
    "Building corner (far left)",
    "Building corner (far right)", 
    "Street sign or traffic light",
    "Road marking or curb",
    "Distinctive object (car, pole)",
    "Another building corner",
    "Another road feature",
    "Any clear geometric feature"
]

def click_image(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(picked_pixels) < len(point_types):
            picked_pixels.append((x, y))
            print(f"[Image] Point {len(picked_pixels)} ({point_types[len(picked_pixels)-1]}): ({x}, {y})")

def click_lidar_simple(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(picked_lidar_points) < len(picked_pixels):
            # Convert display coordinates
            display_x = (x - 400) / 10
            display_y = (400 - y) / 10
            
            # Find nearest point
            distances = np.linalg.norm(lidar_points[:, :2] - np.array([display_x, display_y]), axis=1)
            nearest_idx = np.argmin(distances)
            
            # Check for duplicates
            if len(picked_lidar_points) > 0:
                min_dist = min([np.linalg.norm(lidar_points[nearest_idx] - existing) for existing in picked_lidar_points])
                if min_dist < 1.0:
                    print(f"[LiDAR] Point too close to existing selection, try again")
                    return
            
            picked_lidar_points.append(lidar_points[nearest_idx])
            print(f"[LiDAR] Point {len(picked_lidar_points)}: {lidar_points[nearest_idx]}")

# Step 1: Guided image point selection
print("Step 1: Select points in the image")
print("Follow the prompts to select specific types of features")

cv2.namedWindow("Guided Image Selection", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Guided Image Selection", click_image)

while True:
    img_copy = img.copy()
    
    # Draw selected points
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
    
    for i, (x, y) in enumerate(picked_pixels):
        color = colors[i % len(colors)]
        cv2.circle(img_copy, (x, y), 8, color, -1)
        cv2.circle(img_copy, (x, y), 10, (255, 255, 255), 2)
        cv2.putText(img_copy, str(i+1), (x+12, y+4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show current instruction
    current_point = len(picked_pixels)
    if current_point < len(point_types):
        instruction = f"Select: {point_types[current_point]}"
        cv2.putText(img_copy, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img_copy, f"Progress: {current_point}/{len(point_types)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(img_copy, "All points selected! Press 'q' to continue", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Guided Image Selection", img_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyWindow("Guided Image Selection")

if len(picked_pixels) < 6:
    print(f"Error: You need at least 6 points, but only selected {len(picked_pixels)}")
    exit()

# Step 2: Show 3D LiDAR visualization
print(f"\nStep 2: 3D LiDAR visualization to help you identify the same objects")
print("Look at the 3D views to understand the LiDAR data structure")

# Filter and visualize LiDAR points
ground_height = np.percentile(lidar_points[:, 2], 10)
filtered_mask = (lidar_points[:, 2] > ground_height) & (lidar_points[:, 2] < 10) & \
                (np.abs(lidar_points[:, 0]) < 50) & (np.abs(lidar_points[:, 1]) < 50)
filtered_points = lidar_points[filtered_mask]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Top-down view
axes[0].scatter(filtered_points[:, 0], filtered_points[:, 1], c=filtered_points[:, 2], 
               s=1, cmap='viridis', alpha=0.6)
axes[0].set_xlabel('X (forward)')
axes[0].set_ylabel('Y (left)')
axes[0].set_title('Top-Down View')
axes[0].grid(True)

# Side view
axes[1].scatter(filtered_points[:, 0], filtered_points[:, 2], c=filtered_points[:, 1], 
               s=1, cmap='viridis', alpha=0.6)
axes[1].set_xlabel('X (forward)')
axes[1].set_ylabel('Z (up)')
axes[1].set_title('Side View')
axes[1].grid(True)

# Front view
axes[2].scatter(filtered_points[:, 1], filtered_points[:, 2], c=filtered_points[:, 0], 
               s=1, cmap='viridis', alpha=0.6)
axes[2].set_xlabel('Y (left)')
axes[2].set_ylabel('Z (up)')
axes[2].set_title('Front View')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('lidar_guide.png', dpi=150, bbox_inches='tight')
print("3D LiDAR guide saved as 'lidar_guide.png'")
plt.show()

# Step 3: Guided LiDAR point selection
print(f"\nStep 3: Select the corresponding {len(picked_pixels)} points in LiDAR data")
print("Use the 3D views to help identify the same objects")

# Create height-colored LiDAR display
lidar_display = np.zeros((800, 800, 3), dtype=np.uint8)
lidar_display[:] = 255

# Draw axes
cv2.line(lidar_display, (400, 0), (400, 800), (128, 128, 128), 1)
cv2.line(lidar_display, (0, 400), (800, 400), (128, 128, 128), 1)
cv2.putText(lidar_display, "X (forward)", (780, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
cv2.putText(lidar_display, "Y (left)", (420, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

# Draw points with height coloring
for pt in filtered_points:
    cx = int(pt[0] * 10 + 400)
    cy = int(400 - pt[1] * 10)
    if 0 <= cx < 800 and 0 <= cy < 800:
        height_color = int(255 * (pt[2] - ground_height) / (10 - ground_height))
        height_color = max(0, min(255, height_color))
        cv2.circle(lidar_display, (cx, cy), 1, (height_color, height_color, height_color), -1)

cv2.namedWindow("Guided LiDAR Selection", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Guided LiDAR Selection", click_lidar_simple)

while True:
    display_copy = lidar_display.copy()
    
    # Draw selected points
    for i, point in enumerate(picked_lidar_points):
        display_x = int(point[0] * 10 + 400)
        display_y = int(400 - point[1] * 10)
        
        if 0 <= display_x < 800 and 0 <= display_y < 800:
            color = colors[i % len(colors)]
            cv2.circle(display_copy, (display_x, display_y), 8, color, -1)
            cv2.circle(display_copy, (display_x, display_y), 10, (255, 255, 255), 2)
            cv2.putText(display_copy, str(i+1), (display_x+12, display_y+4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show current instruction
    current_point = len(picked_lidar_points)
    if current_point < len(picked_pixels):
        instruction = f"Select LiDAR point for: {point_types[current_point]}"
        cv2.putText(display_copy, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(display_copy, f"Progress: {current_point}/{len(picked_pixels)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(display_copy, "All points selected! Press 'q' to continue", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Guided LiDAR Selection", display_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyWindow("Guided LiDAR Selection")

# Step 4: Calibration
if len(picked_pixels) != len(picked_lidar_points):
    print(f"Error: Mismatch! {len(picked_pixels)} image points vs {len(picked_lidar_points)} LiDAR points")
    exit()

print(f"\nStep 4: Computing calibration with {len(picked_pixels)} point pairs")

# Use KITTI camera matrix
K = np.array([[721.5377, 0, 609.5593],
              [0, 721.5377, 172.8540],
              [0, 0, 1]])

picked_pixels_np = np.array(picked_pixels, dtype=np.float32)
picked_lidar_points_np = np.array(picked_lidar_points, dtype=np.float32)

# Solve PnP with RANSAC
success, rvec, tvec, inliers = cv2.solvePnPRansac(
    picked_lidar_points_np, picked_pixels_np, K, None,
    flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=2.0
)

if success:
    R_mat, _ = cv2.Rodrigues(rvec)
    
    print(f"Calibration successful! {len(inliers)}/{len(picked_pixels)} points used")
    print("\nRotation Matrix:")
    print(R_mat)
    print("\nTranslation Vector:")
    print(tvec)
    
    # Save calibration
    with open("simple_calib.txt", "w") as f:
        f.write("# Simple guided calibration file\n")
        f.write("R:\n" + np.array2string(R_mat, precision=6) + "\n")
        f.write("T:\n" + np.array2string(tvec, precision=6) + "\n")
    
    print("\nCalibration saved as 'simple_calib.txt'")
    
    # Validation
    print("\nValidation:")
    projected_points, _ = cv2.projectPoints(picked_lidar_points_np, rvec, tvec, K, None)
    projected_points = projected_points.reshape(-1, 2)
    
    errors = np.linalg.norm(projected_points - picked_pixels_np, axis=1)
    print(f"Average reprojection error: {np.mean(errors):.2f} pixels")
    print(f"Max reprojection error: {np.max(errors):.2f} pixels")
    
    if np.mean(errors) < 3:
        print("✓ Excellent calibration!")
    elif np.mean(errors) < 5:
        print("✓ Good calibration")
    else:
        print("⚠️  Calibration may need improvement")
        
else:
    print("❌ Calibration failed! Try selecting different points.")

import numpy as np
import cv2
import matplotlib.pyplot as plt

# -------------------
# Load data
# -------------------
def load_velodyne_bin(bin_path):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]

image_path = "000000.png"
lidar_path = "000000.bin"

img = cv2.imread(image_path)
lidar_points = load_velodyne_bin(lidar_path)
print("Loaded LiDAR points:", lidar_points.shape)

print("\n=== IMPROVED CALIBRATION TOOL ===")
print("This tool will help you select corresponding points more accurately")
print("1. First, select points in the image")
print("2. Then, we'll show you a 3D visualization to help select LiDAR points")
print("3. Finally, we'll compute the calibration")
print("=====================================\n")

picked_pixels = []
picked_lidar_points = []

# -------------------
# Mouse callbacks
# -------------------
def click_image(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        picked_pixels.append((x, y))
        print(f"[Image] Point {len(picked_pixels)}: ({x}, {y})")

def click_lidar_3d(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Find nearest point in 3D space
        display_points, valid_indices = param
        
        # Convert display coordinates to 3D coordinates
        display_x = (x - 400) / 10  # Reverse scaling
        display_y = (400 - y) / 10
        
        # Find nearest point in 3D
        distances = np.linalg.norm(lidar_points[valid_indices, :2] - np.array([display_x, display_y]), axis=1)
        nearest_idx = valid_indices[np.argmin(distances)]
        
        # Check if this point is already selected
        if len(picked_lidar_points) > 0:
            min_dist_to_existing = min([np.linalg.norm(lidar_points[nearest_idx] - existing) for existing in picked_lidar_points])
            if min_dist_to_existing < 0.5:  # Increased threshold
                print(f"[LiDAR] Point too close to existing selection, try again")
                return
        
        picked_lidar_points.append(lidar_points[nearest_idx])
        print(f"[LiDAR] Point {len(picked_lidar_points)}: {lidar_points[nearest_idx]}")

# -------------------
# Step 1: Pick points in image with better visualization
# -------------------
print("Step 1: Click on at least 8 distinctive points in the image")
print("Choose points like:")
print("- Building corners")
print("- Street signs")
print("- Traffic lights")
print("- Distinctive objects")
print("- Road markings")
print("Press 'q' to finish image point selection")

cv2.namedWindow("Image Selection", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Image Selection", click_image)

while True:
    img_copy = img.copy()
    
    # Draw selected points with numbers and colors
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)]
    
    for i, (x, y) in enumerate(picked_pixels):
        color = colors[i % len(colors)]
        cv2.circle(img_copy, (x, y), 8, color, -1)
        cv2.circle(img_copy, (x, y), 10, (255, 255, 255), 2)
        cv2.putText(img_copy, str(i+1), (x+12, y+4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add instructions
    cv2.putText(img_copy, f"Selected: {len(picked_pixels)}/8+ points", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img_copy, "Click on distinctive features", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imshow("Image Selection", img_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyWindow("Image Selection")

if len(picked_pixels) < 6:
    print(f"Error: You need at least 6 points, but only selected {len(picked_pixels)}")
    exit()

# -------------------
# Step 2: Create 3D visualization for LiDAR point selection
# -------------------
print(f"\nStep 2: Now select the corresponding {len(picked_pixels)} points in the LiDAR data")
print("The visualization will help you identify the same objects")

# Create 3D visualization
fig = plt.figure(figsize=(15, 5))

# Filter LiDAR points for better visualization (remove ground and far points)
ground_height = np.percentile(lidar_points[:, 2], 10)
filtered_mask = (lidar_points[:, 2] > ground_height) & (lidar_points[:, 2] < 10) & \
                (np.abs(lidar_points[:, 0]) < 50) & (np.abs(lidar_points[:, 1]) < 50)
filtered_points = lidar_points[filtered_mask]

print(f"Filtered LiDAR points for visualization: {len(filtered_points)}")

# Top-down view (X-Y)
ax1 = fig.add_subplot(131)
ax1.scatter(filtered_points[:, 0], filtered_points[:, 1], c=filtered_points[:, 2], 
           s=1, cmap='viridis', alpha=0.6)
ax1.set_xlabel('X (forward)')
ax1.set_ylabel('Y (left)')
ax1.set_title('Top-Down View (X-Y)')
ax1.grid(True)
ax1.set_aspect('equal')

# Side view (X-Z)
ax2 = fig.add_subplot(132)
ax2.scatter(filtered_points[:, 0], filtered_points[:, 2], c=filtered_points[:, 1], 
           s=1, cmap='viridis', alpha=0.6)
ax2.set_xlabel('X (forward)')
ax2.set_ylabel('Z (up)')
ax2.set_title('Side View (X-Z)')
ax2.grid(True)
ax2.set_aspect('equal')

# Front view (Y-Z)
ax3 = fig.add_subplot(133)
ax3.scatter(filtered_points[:, 1], filtered_points[:, 2], c=filtered_points[:, 0], 
           s=1, cmap='viridis', alpha=0.6)
ax3.set_xlabel('Y (left)')
ax3.set_ylabel('Z (up)')
ax3.set_title('Front View (Y-Z)')
ax3.grid(True)
ax3.set_aspect('equal')

plt.tight_layout()
plt.savefig('lidar_3d_views.png', dpi=150, bbox_inches='tight')
print("3D LiDAR views saved as 'lidar_3d_views.png'")
plt.show()

# -------------------
# Step 3: Interactive LiDAR point selection with multiple views
# -------------------
print("\nStep 3: Interactive LiDAR point selection")
print("You'll see multiple views to help you identify the same objects")

# Create a better top-down display with height information
lidar_display = np.zeros((800, 800, 3), dtype=np.uint8)
lidar_display[:] = 255

# Draw coordinate axes
cv2.line(lidar_display, (400, 0), (400, 800), (128, 128, 128), 1)
cv2.line(lidar_display, (0, 400), (800, 400), (128, 128, 128), 1)
cv2.putText(lidar_display, "X (forward)", (780, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
cv2.putText(lidar_display, "Y (left)", (420, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

# Draw LiDAR points with height-based coloring
for pt in filtered_points:
    cx = int(pt[0] * 10 + 400)
    cy = int(400 - pt[1] * 10)
    if 0 <= cx < 800 and 0 <= cy < 800:
        # Color based on height (Z coordinate)
        height_color = int(255 * (pt[2] - ground_height) / (10 - ground_height))
        height_color = max(0, min(255, height_color))
        cv2.circle(lidar_display, (cx, cy), 1, (height_color, height_color, height_color), -1)

cv2.namedWindow("LiDAR 3D Selection", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("LiDAR 3D Selection", click_lidar_3d, (filtered_points, np.where(filtered_mask)[0]))

while True:
    display_copy = lidar_display.copy()
    
    # Draw selected points with numbers and colors
    for i, point in enumerate(picked_lidar_points):
        display_x = int(point[0] * 10 + 400)
        display_y = int(400 - point[1] * 10)
        
        if 0 <= display_x < 800 and 0 <= display_y < 800:
            color = colors[i % len(colors)]
            cv2.circle(display_copy, (display_x, display_y), 8, color, -1)
            cv2.circle(display_copy, (display_x, display_y), 10, (255, 255, 255), 2)
            cv2.putText(display_copy, str(i+1), (display_x+12, display_y+4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add instructions and info
    cv2.putText(display_copy, f"Selected: {len(picked_lidar_points)}/{len(picked_pixels)}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(display_copy, "Click on SAME objects as in image", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(display_copy, "Use 3D views to help identify objects", 
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imshow("LiDAR 3D Selection", display_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyWindow("LiDAR 3D Selection")

# -------------------
# Step 4: Validation and calibration
# -------------------
if len(picked_pixels) != len(picked_lidar_points):
    print(f"Error: Mismatch! {len(picked_pixels)} image points vs {len(picked_lidar_points)} LiDAR points")
    exit()

print(f"\nStep 4: Computing calibration with {len(picked_pixels)} point pairs")

# Use the KITTI camera matrix for better accuracy
K = np.array([[721.5377, 0, 609.5593],
              [0, 721.5377, 172.8540],
              [0, 0, 1]])

picked_pixels_np = np.array(picked_pixels, dtype=np.float32)
picked_lidar_points_np = np.array(picked_lidar_points, dtype=np.float32)

# Solve PnP with RANSAC for better robustness
success, rvec, tvec, inliers = cv2.solvePnPRansac(
    picked_lidar_points_np, picked_pixels_np, K, None,
    flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=3.0
)

if success:
    R_mat, _ = cv2.Rodrigues(rvec)
    
    print(f"Calibration successful! {len(inliers)}/{len(picked_pixels)} points used")
    print("\nRotation Matrix:")
    print(R_mat)
    print("\nTranslation Vector:")
    print(tvec)
    
    # Save improved calibration
    with open("improved_calib.txt", "w") as f:
        f.write("# Improved calibration file\n")
        f.write("R:\n" + np.array2string(R_mat, precision=6) + "\n")
        f.write("T:\n" + np.array2string(tvec, precision=6) + "\n")
    
    print("\nImproved calibration saved as 'improved_calib.txt'")
    
    # Quick validation
    print("\nQuick validation:")
    projected_points, _ = cv2.projectPoints(picked_lidar_points_np, rvec, tvec, K, None)
    projected_points = projected_points.reshape(-1, 2)
    
    errors = np.linalg.norm(projected_points - picked_pixels_np, axis=1)
    print(f"Average reprojection error: {np.mean(errors):.2f} pixels")
    print(f"Max reprojection error: {np.max(errors):.2f} pixels")
    
    if np.mean(errors) < 5:
        print("✓ Good calibration (low reprojection error)")
    else:
        print("⚠️  Calibration may need improvement (high reprojection error)")
        
else:
    print("❌ Calibration failed! Try selecting different points.")

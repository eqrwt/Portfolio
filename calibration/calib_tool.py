import numpy as np
import cv2
import matplotlib.pyplot as plt

# -------------------
# Load data
# -------------------
def load_velodyne_bin(bin_path):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]

image_path = "000010.png"   # your uploaded image
lidar_path = "000010.bin"   # your uploaded lidar

img = cv2.imread(image_path)
lidar_points = load_velodyne_bin(lidar_path)
print("Loaded LiDAR points:", lidar_points.shape)
print("\n=== CALIBRATION TOOL INSTRUCTIONS ===")
print("1. You will select 6+ corresponding points in the image and LiDAR data")
print("2. Select points in the SAME ORDER in both views")
print("3. Choose distinctive features like corners, edges, or clear objects")
print("4. The tool will show numbered points to help you track selections")
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

def click_lidar(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # find nearest lidar point in top-down projection
        top_down = param
        # Convert display coordinates back to lidar coordinates
        display_x = x - 400
        display_y = 400 - y
        lidar_coords = np.array([display_x / 10, display_y / 10])  # reverse the scaling
        
        # Find nearest point in original lidar coordinates
        distances = np.linalg.norm(lidar_points[:, :2] - lidar_coords, axis=1)
        idx = np.argmin(distances)
        
        # Check if this point is already selected
        if len(picked_lidar_points) > 0:
            min_dist_to_existing = min([np.linalg.norm(lidar_points[idx] - existing) for existing in picked_lidar_points])
            if min_dist_to_existing < 0.1:  # threshold to avoid duplicates
                print(f"[LiDAR] Point too close to existing selection, try again")
                return
        
        picked_lidar_points.append(lidar_points[idx])
        print(f"[LiDAR] Point {len(picked_lidar_points)}: {lidar_points[idx]}")

# -------------------
# Step 1: Pick points in image
# -------------------
print("Step 1: Click on at least 6 points in the image")
print("Press 'q' to finish image point selection")
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Image", click_image)

while True:
    # Create a copy of the image to draw on
    img_copy = img.copy()
    
    # Draw selected points with numbers
    for i, (x, y) in enumerate(picked_pixels):
        cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)  # Green circle
        cv2.putText(img_copy, str(i+1), (x+8, y-8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Blue text
    
    # Show current selection count
    cv2.putText(img_copy, f"Selected: {len(picked_pixels)}/6", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Image", img_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyWindow("Image")

# -------------------
# Step 2: Pick points in LiDAR top-down
# -------------------
print("Step 2: Click on corresponding points in the LiDAR top-down view")
print("Make sure to select the SAME points you selected in the image")
print("Press 'q' to finish LiDAR point selection")
# make top-down projection for selection
lidar_topdown = lidar_points[:, [0, 1]] * 10  # scaled for display
lidar_display = np.zeros((800, 800, 3), dtype=np.uint8)
lidar_display[:] = 255

# Draw coordinate axes for reference
cv2.line(lidar_display, (400, 0), (400, 800), (128, 128, 128), 1)  # Y-axis
cv2.line(lidar_display, (0, 400), (800, 400), (128, 128, 128), 1)  # X-axis
cv2.putText(lidar_display, "X", (780, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
cv2.putText(lidar_display, "Y", (420, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

for pt in lidar_topdown:
    cx = int(pt[0] + 400)
    cy = int(400 - pt[1])
    if 0 <= cx < 800 and 0 <= cy < 800:
        lidar_display[cy, cx] = (0, 0, 0)

cv2.namedWindow("LiDAR TopDown", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("LiDAR TopDown", click_lidar, lidar_topdown)

while True:
    # Create a copy of the display to draw on
    display_copy = lidar_display.copy()
    
    # Draw selected points with numbers
    for i, point in enumerate(picked_lidar_points):
        # Convert 3D point back to display coordinates
        display_x = int(point[0] * 10 + 400)
        display_y = int(400 - point[1] * 10)
        
        if 0 <= display_x < 800 and 0 <= display_y < 800:
            # Draw a colored circle for selected points
            cv2.circle(display_copy, (display_x, display_y), 5, (0, 255, 0), -1)  # Green circle
            # Add point number
            cv2.putText(display_copy, str(i+1), (display_x+8, display_y-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Blue text
    
    # Show current selection count
    cv2.putText(display_copy, f"Selected: {len(picked_lidar_points)}/{len(picked_pixels)}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("LiDAR TopDown", display_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyWindow("LiDAR TopDown")

# -------------------
# Calibration solve
# -------------------
if len(picked_pixels) < 6 or len(picked_lidar_points) < 6:
    raise ValueError(f"You need at least 6 matching points for reliable calibration, got {len(picked_pixels)} & {len(picked_lidar_points)}")

K = np.array([[721.5377, 0, 609.5593],
              [0, 721.5377, 172.8540],
              [0, 0, 1]])

picked_pixels_np = np.array(picked_pixels, dtype=np.float32)
picked_lidar_points_np = np.array(picked_lidar_points, dtype=np.float32)

success, rvec, tvec = cv2.solvePnP(picked_lidar_points_np, picked_pixels_np, K, None)
R_mat, _ = cv2.Rodrigues(rvec)

print("Rotation Matrix:\n", R_mat)
print("Translation Vector:\n", tvec)

with open("calib.txt", "w") as f:
    f.write("# Calibration file\n")
    f.write("R:\n" + np.array2string(R_mat, precision=6) + "\n")
    f.write("T:\n" + np.array2string(tvec, precision=6) + "\n")
print("Calibration file saved: calib.txt")

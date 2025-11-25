import numpy as np

# Your manual calibration results
R_manual = np.array([
    [ 0.933079, -0.359012,  0.021752],
    [ 0.032389,  0.02364,  -0.999196],
    [ 0.358209,  0.933033,  0.033686]
])

T_manual = np.array([[-8.599121], [-1.986499], [-7.927573]])

# KITTI format calibration data (from your input)
# P0, P1, P2, P3 are projection matrices for different cameras
# R0_rect is the rectification matrix
# Tr_velo_to_cam is the transformation from LiDAR to camera
# Tr_imu_to_velo is the transformation from IMU to LiDAR

# Extract Tr_velo_to_cam from KITTI format
Tr_velo_to_cam_kitti = np.array([
    [6.927964000000e-03, -9.999722000000e-01, -2.757829000000e-03, -2.457729000000e-02],
    [-1.162982000000e-03, 2.749836000000e-03, -9.999955000000e-01, -6.127237000000e-02],
    [9.999753000000e-01, 6.931141000000e-03, -1.143899000000e-03, -3.321029000000e-01]
])

# Extract R and T from KITTI Tr_velo_to_cam
R_kitti = Tr_velo_to_cam_kitti[:, :3]
T_kitti = Tr_velo_to_cam_kitti[:, 3].reshape(3, 1)

print("=== CALIBRATION COMPARISON ===\n")

print("1. ROTATION MATRIX COMPARISON:")
print("Manual calibration R:")
print(R_manual)
print("\nKITTI ground truth R:")
print(R_kitti)
print("\nDifference (Manual - KITTI):")
print(R_manual - R_kitti)

print("\n2. TRANSLATION VECTOR COMPARISON:")
print("Manual calibration T:")
print(T_manual)
print("\nKITTI ground truth T:")
print(T_kitti)
print("\nDifference (Manual - KITTI):")
print(T_manual - T_kitti)

print("\n3. METRICS:")
# Calculate rotation error (angle between rotation matrices)
R_diff = R_manual @ R_kitti.T
trace = np.trace(R_diff)
angle_error = np.arccos(np.clip((trace - 1) / 2, -1, 1)) * 180 / np.pi
print(f"Rotation error: {angle_error:.2f} degrees")

# Calculate translation error
translation_error = np.linalg.norm(T_manual - T_kitti)
print(f"Translation error: {translation_error:.3f} meters")

print("\n4. YOUR CALIBRATION IN KITTI FORMAT:")
print("Tr_velo_to_cam (your manual calibration):")
Tr_manual_kitti = np.hstack([R_manual, T_manual])
for row in Tr_manual_kitti:
    print(" ".join([f"{val:12.6e}" for val in row]))

print("\n5. COMPLETE KITTI CALIBRATION FILE:")
print("P0: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 0.000000000000e+00 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00")
print("P1: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 -3.797842000000e+02 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00")
print("P2: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 4.575831000000e+01 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 -3.454157000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 4.981016000000e-03")
print("P3: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 -3.341081000000e+02 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 2.330660000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 3.201153000000e-03")
print("R0_rect: 9.999128000000e-01 1.009263000000e-02 -8.511932000000e-03 -1.012729000000e-02 9.999406000000e-01 -4.037671000000e-03 8.470675000000e-03 4.123522000000e-03 9.999556000000e-01")

# Your manual calibration in KITTI format
print("Tr_velo_to_cam:", end=" ")
for i, val in enumerate(Tr_manual_kitti.flatten()):
    if i > 0:
        print(" ", end="")
    print(f"{val:12.6e}", end="")
print()

print("Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01")

print("\n6. ANALYSIS:")
if angle_error < 5:
    print("✓ Rotation accuracy is good (< 5 degrees)")
else:
    print("✗ Rotation accuracy needs improvement (> 5 degrees)")

if translation_error < 1:
    print("✓ Translation accuracy is good (< 1 meter)")
else:
    print("✗ Translation accuracy needs improvement (> 1 meter)")

print(f"\nOverall assessment: Your manual calibration has a rotation error of {angle_error:.2f}° and translation error of {translation_error:.3f}m")

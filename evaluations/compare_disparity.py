import cv2
import numpy as np

# Load the left and right stereo images
imgL = cv2.imread('datasets/evaluation/offline/direct/4cm/left/2676726237.png', 1)
imgR = cv2.imread('datasets/evaluation/offline/direct/4cm/right/2676726237.png', 1)

# Set the SGBM parameters
win_size = 9
min_disp = 0
num_disp = 64
block_size = 5
disp12_max_diff = 1
uniqueness_ratio = 5
speckle_range = 2
speckle_window_size = 100
mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY

# Create the SGBM object
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 3 * win_size ** 2,
    P2=32 * 3 * win_size ** 2,
    disp12MaxDiff=disp12_max_diff,
    uniquenessRatio=uniqueness_ratio,
    speckleRange=speckle_range,
    speckleWindowSize=speckle_window_size,
    mode=mode
)

# Compute the disparity map
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

# Normalize the disparity map
disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)

# Apply color map to the normalized disparity map
disparity_color = cv2.applyColorMap(disparity_normalized.astype(np.uint8), cv2.COLORMAP_JET)

# Save the colored disparity map as a PNG file
cv2.imwrite('disparity.png', disparity_color)


import cv2
import numpy as np

# Load the image from the .npy file
img = np.load('datasets/evaluation/online/direct/4cm/disparity/1116271382.npy')

# Convert the image to 8-bit unsigned integer
img = cv2.convertScaleAbs(img)

# Save the image as a PNG file
cv2.imwrite('image.png', img)
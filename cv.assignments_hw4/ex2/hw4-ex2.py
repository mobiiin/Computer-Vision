import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

im1 = cv.imread('1-1.jpg')
# Four corners of corridor
pts1 = np.array([[1116, 689], [1828, 797], [448, 1180], [1926, 1484]])
im2 = cv.imread('1-2.jpg')
# Four corners to align
pts2 = np.array([[566, 855], [1268, 753], [634, 1425], [1489, 1273]])
# Calculate Homography
h, _ = cv.findHomography(pts1, pts2)
# Warp source image to destination based on homography
im_out = cv.warpPerspective(im1, h, (im2.shape[1], im2.shape[0]))
# Display
plt.imshow(im_out)
plt.show()
# second image
pts1 = np.array([[566, 855], [1268, 753], [634, 1425], [1489, 1273]])
pts2 = np.array([[500, 500], [500, 1500], [1500, 500], [1500, 1500]])
h, _ = cv.findHomography(pts1, pts2)
im_out2 = cv.warpPerspective(im2, h, (im2.shape[1], im2.shape[0]))
plt.imshow(im_out2)
plt.show()

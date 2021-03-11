import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('image.jpg', 0)
temp = cv.imread('template.jpg', 0)

# Initiate SIFT detector
# sift = cv.xfeatures2d.SIFT_create()
# Initiate ORB detector
orb = cv.ORB_create()

# find the keypoints and descriptors
kp1, des1 = orb.detectAndCompute(img, None)
kp2, des2 = orb.detectAndCompute(temp, None)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort them in the order of their distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 20 matches
img3 = cv.drawMatches(img, kp1, temp, kp2, matches[:20], None)

plt.imshow(img3), plt.show()

akaze = cv.AKAZE_create()
kp1, des1 = akaze.detectAndCompute(img, None)
kp2, des2 = akaze.detectAndCompute(temp, None)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
img4 = cv.drawMatches(img, kp1, temp, kp2, matches[:20], None)
plt.imshow(img4), plt.show()

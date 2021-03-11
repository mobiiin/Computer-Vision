import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop left
    elif not np.sum(frame[:, 0]):
        return trim(frame[:, 1:])
    #crop right
    elif not np.sum(frame[:, -1]):
        return trim(frame[:, :-2])
    return frame


im1 = cv.imread('3-1.jpeg')
gray1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
im2 = cv.imread('3-2.jpeg')
gray2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

akaze = cv.AKAZE_create()
kp1, des1 = akaze.detectAndCompute(gray1, None)
kp2, des2 = akaze.detectAndCompute(gray2, None)

match = cv.BFMatcher(cv.NORM_HAMMING)
matches = match.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)

# cv2.drawMatchesKnn expects list of lists as matches.
im3 = cv.drawMatches(im1, kp1, im2, kp2, good, None, flags=2)
cv.imshow("KNN on ORB", im3)
cv.waitKey(0)

min_match = 10
if len(good) > min_match:
    # construct the two sets of points
    ptsA = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    ptsB = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    h, status = cv.findHomography(ptsB, ptsA, cv.RANSAC, 4)

result = cv.warpPerspective(im2, h, (im1.shape[1] + im2.shape[1], im1.shape[0]))
cv.imshow('result', trim(result))
cv.waitKey(0)
result[0:im1.shape[0], 0:im1.shape[1]] = im1
cv.imshow('result', trim(result))
cv.waitKey(0)

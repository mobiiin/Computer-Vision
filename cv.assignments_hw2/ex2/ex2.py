import cv2 as cv
import numpy as np

img = cv.imread('test_image1.jpg', 1)
img = cv.resize(img, (128, 64))

winSize = (128, 64)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (16, 16)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 0
nlevels = 64
SignedGradients = 0
hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType,
                        L2HysThreshold, gammaCorrection, nlevels, SignedGradients)
hist = hog.compute(img)
print(hist)
np.save('histogram', hist)

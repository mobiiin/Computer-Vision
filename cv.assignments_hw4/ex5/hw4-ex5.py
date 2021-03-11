import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

im1 = cv.imread('4-1.jpg')
gray1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
im2 = cv.imread('4-2.jpg')
gray2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

retval, corners781 = cv.findChessboardCorners(gray1, patternSize=(7, 8))
corners781 = cv.cornerSubPix(gray1, corners781, (11, 11), (-1, -1), criteria)    # refine corners


#plt.imshow(cv.cvtColor(im1, cv.COLOR_BGR2RGB), cmap='gray')
#plt.scatter(corners781[:, 0, 0], corners781[:, 0, 1])
#plt.show()
#img = cv.drawChessboardCorners(im1, (7, 8), corners781, retval)
#cv.imshow('img', img)
#cv.waitKey(0)
retval, corners541 = cv.findChessboardCorners(gray1, patternSize=(5, 4))
corners541 = cv.cornerSubPix(gray1, corners541, (11, 11), (-1, -1), criteria)

#img = cv.drawChessboardCorners(im1, (5, 4), corners541, retval)
#cv.imshow('img', img)
#cv.waitKey(0)
_, corners782 = cv.findChessboardCorners(gray2, patternSize=(7, 8))
corners782 = cv.cornerSubPix(gray2, corners782, (11, 11), (-1, -1), criteria)

_, corners542 = cv.findChessboardCorners(gray2, patternSize=(5, 4))
corners542 = cv.cornerSubPix(gray2, corners542, (11, 11), (-1, -1), criteria)
pts1 = [*np.float32(corners781), *np.float32(corners541)]
pts2 = [*np.float32(corners782), *np.float32(corners542)]
pts1 = np.array(pts1)
pts2 = np.array(pts2)
F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
print(F)

def drawlines(img1, img2, lines, pts1, pts2):
    _, c = img1.shape[0:2]

    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        #img1 = cv.circle(img1, tuple(*pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(*pt2), 5, color, -1)
    return img1, img2


# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(im1, im2, lines1, pts1, pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(im2, im1, lines2, pts2, pts1)

plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img6)
plt.show()

im3 = cv.imread('4-3.jpg')
gray3 = cv.cvtColor(im3, cv.COLOR_BGR2GRAY)
im4 = cv.imread('4-4.jpg')
gray4 = cv.cvtColor(im4, cv.COLOR_BGR2GRAY)

epiline = cv.computeCorrespondEpilines(np.array([265, 305]).reshape(-1, 1, 2), 2, F)
epiline = epiline.squeeze(0)
epiline = epiline.squeeze(0)
_, c = im3.shape[0:2]
r = epiline
x0, y0 = map(int, [0, -r[2]/r[1]])
x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
img_1 = cv.line(im3, (x0, y0), (x1, y1), (0, 255, 100), 1)
img_2 = cv.circle(im4, tuple([265, 305]), 5, (0, 255, 100), -1)
plt.subplot(121), plt.imshow(img_1)
plt.subplot(122), plt.imshow(img_2)
plt.show()

lines3 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines3 = lines3.reshape(-1, 3)
img7, img8 = drawlines(im3, im4, lines3, pts1, pts2)

lines4 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines4 = lines4.reshape(-1, 3)
img9, img10 = drawlines(im4, im3, lines4, pts2, pts1)

plt.subplot(121), plt.imshow(img7)
plt.subplot(122), plt.imshow(img8)
plt.show()

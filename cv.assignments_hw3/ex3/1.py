import tarfile
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

im1 = cv.imread('9960.png')
im1 = cv.cvtColor(im1, cv.COLOR_BGR2RGB)
#im1 = np.float32(im1)
plt.imshow(im1)
plt.show()

# defining a function to scan an image
def sliding_window(image, stepsize, windowsize):
    # slide a window across the image
    if windowsize is None:
        windowsize = [32, 32]
    for y in range(0, image.shape[0], stepsize):
        for x in range(0, image.shape[1], stepsize):
            # yield the current window
            image1 = image[y:y + windowsize[0], x:x + windowsize[1]]
            image[y:y + windowsize[0], x:x + windowsize[1]] = image1
            yield x, y, image1


scores = []
box = []
for (x1, y1, window) in sliding_window(im1, stepsize=4, windowsize=None):
    if window.shape[0] != 32 or window.shape[1] != 32:
        continue
    score = model.evaluate(window)
    if score[1] >= .9:
        # saving the score and prospective window
        scores.append(score[1])
        # reversing the coordinates to draw the rectangle in the original image
        # (d is the downscale, p is the downscale sessions)
        box.append([y1, x1, (y1 + 32), (x1 + 32)])
        cv.rectangle(im1, (x1, y1), ((x1 + 32), (y1 + 32)), (0, 0, 255), 1)
        cv.imshow('1', im1)
        cv.waitKey(0)
        cv.imwrite('im.jpg', im1)
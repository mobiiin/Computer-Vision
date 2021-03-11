import cv2 as cv
import numpy as np
import sys


def mouse_handler(event, x, y, flags, data):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(data['im'], (x, y), 3, (0, 0, 255), 5, 16)
        cv.imshow("Image", data['im'])
        if len(data['points']) < 4:
            data['points'].append([x, y])


def get_four_points(im):
    # Set up data to send to mouse handler
    data = {}
    data['im'] = im.copy()
    data['points'] = []

    # Set the callback function for any mouse event
    cv.imshow("Image", im)
    cv.setMouseCallback("Image", mouse_handler, data)
    cv.waitKey(0)

    # Convert array to np.array
    points = np.vstack(data['points']).astype(float)

    return points


im1 = cv.imread('2-1.jpg')
im2 = cv.imread('2-2.jpg')
im2 = cv.resize(im2, (1920, 1080))
im3 = cv.imread('2-3.jpg')
im3 = cv.resize(im3, (1920, 1080))

size = im3.shape
# Create a vector of source points.
pts_src = np.array([[0, 0], [size[1] - 1, 0], [size[1] - 1, size[0] - 1],
                    [0, size[0] - 1]], dtype=float)

# Get four corners of the billboard
pts_dst = get_four_points(im2)
# Calculate Homography between source and destination points
h, status = cv.findHomography(pts_src, pts_dst)
im_temp = cv.warpPerspective(im3, h, (im2.shape[1], im2.shape[0]))
# Black out polygonal area in destination image.
cv.fillConvexPoly(im2, pts_dst.astype(int), 0, 16)
# Add warped source image to destination image.
im_out = im2 + im_temp
# Display image.
cv.imshow("Image", im_out)
cv.imwrite('2_out.jpg', im_out)
cv.waitKey(0)
# second image
pts_dst2 = get_four_points(im1)
h, _ = cv.findHomography(pts_src, pts_dst2)
im_temp2 = cv.warpPerspective(im3, h, (im1.shape[1], im1.shape[0]))
cv.fillConvexPoly(im1, pts_dst2.astype(int), 0, 16)
im_out2 = im1 + im_temp2
cv.imshow("Image2", im_out2)
cv.imwrite('1_out.jpg', im_out2)
cv.waitKey(0)

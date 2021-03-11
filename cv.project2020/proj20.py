import cv2 as cv
import numpy as np
import time
import sys
from centroidtracker import centroidtracker

ct = centroidtracker()
cap = cv.VideoCapture('video1.mp4')
# // extracting the background
#rndframe = cap.get(cv.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=30)  # Randomly chooses 30 frames
#frames = []  # create an empty array
#for i in rndframe:
#    cap.set(cv.CAP_PROP_POS_FRAMES, i)
#    (ret, frame) = cap.read()
#    frames.append(frame)  # adds each frame to the end of the frame arrays

# Calculates the middle frame
#medianF = np.median(frames, axis=0).astype(dtype=np.uint8)
#cv.imshow('frame', medianF)
#cv.imwrite('medianF.jpg', medianF)
#medianF = cv.cvtColor(medianF, cv.COLOR_BGR2GRAY)
##
medianF = cv.imread('medianF.jpg', 0)
old_gray = medianF.copy()

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    for i in boxB:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou, i

# fist four arrays are bounding box coordinates
# the fifth value is for the frame the vehicle has been seen in the original video
Mat = np.array([[[0,0,0,0,0]]], dtype=object)
boxB = [0,0,0,0]
rects = []
frame_count = 0
bboxes = []
#point = ()
#old_points = np.array([[]])
#vehicles = np.vstack([])
while cap.isOpened():
    (ret, frame1) = cap.read()
    frame_count += 1
    if ret == 1:
        frame2 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        backg_sub = cv.absdiff(medianF, frame2)  # subtracts backG from the rest of the frames
        backg_sub = cv.GaussianBlur(backg_sub, (5, 5), 0)
        rett, thresh = cv.threshold(backg_sub, 30, 255, 0)
        element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        dilated = cv.dilate(thresh, kernel=element)
        # fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
        # dilated = cv.dilate(thresh, None, iterations=3)

        #f = medianF + 100*dilated
        #cv.imshow('f',f)
        #for i in range(3):
         #   for x in range(dilated.shape[0]):
          #      for y in range(dilated.shape[1]):
           #         medianF[x,y,i] = dilated[x,y,i]

        contours, hierarchy = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        #cv.drawContours(frame1, contours, -1, (255, 0, 0), 3)
        for contour in contours:

            (x, y, w, h) = cv.boundingRect(contour)
            if cv.contourArea(contour) < 50:
                continue
            #valid_cntrs.append(contour)
            #cv.putText(frame1, "vehicles detected: " + str(len(valid_cntrs)), (55, 15),
            #           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)
            boxA = [x, y, x + w, y + h]
            boxA.append(frame_count)
            new = np.array([boxA], dtype=object)
            Mat = np.vstack((Mat, new))
            # objects = ct.update(boxA)
            #print(boxA)
            rects.append(boxA)
            #print(rects)
            #car = np.array()
            #if frame_count < 1:
             #   continue
            
            iou , i= bb_intersection_over_union(boxA, rects)
            if iou > .9:


            #boxB = [x, y, x + w, y + h]
            cv.rectangle(frame1, (boxA[0], boxA[1]), (boxA[2], boxA[3]), (0, 255, 2), 2)
            #cv.putText(frame1, '%s' % (int(n / 30)), (x - 2, y - 2), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            #points = np.array([[x+int(w/2), y+int(h/2)]], dtype=np.float32)

            # Lucas kanade params
            #lk_params = dict(winSize=(15, 15), maxLevel=4,
             #                criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

            #new_points, status, error = cv.calcOpticalFlowPyrLK(old_gray, frame2, points, None, **lk_params)
            #old_gray = frame2.copy()

            #x1, y1 = new_points.ravel()
          #  cv.circle(frame1, (x1, y1), 5, (0, 255, 0), -1)


        # loop over the tracked objects
        #for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
        #    text = "ID {}".format(objectID)
        #    cv.putText(frame1, text, (centroid[0] - 10, centroid[1] - 10),
        #                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #    cv.circle(frame1, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        cv.imshow('cars', frame1)

        if cv.waitKey(20) == 27:
            break
    else:
        break


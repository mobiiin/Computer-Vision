import cv2 as cv
import numpy as np

# first we load the haar feature files
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier('haarcascade_smile.xml')

img = cv.imread('img1.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# first we detect faces
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    img = cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    # then we detect eyes and smiles within the cropped out face pictures
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    smiles = smile_cascade.detectMultiScale(roi_gray, 1.5, minNeighbors=20)
    for (sx, sy, sw, sh) in smiles:
        cv.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)

img2 = cv.imread('img2.jpg')
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray2, 1.5, minNeighbors=2)
for (x, y, w, h) in faces:
    img2 = cv.rectangle(img2, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray2[y:y+h, x:x+w]
    roi_color = img2[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    smiles = smile_cascade.detectMultiScale(roi_gray, 2.5, minNeighbors=30)
    for (sx, sy, sw, sh) in smiles:
        cv.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)

img3 = cv.imread('img3.jpg')
gray3 = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray3, minNeighbors=5)
for (x, y, w, h) in faces:
    img3 = cv.rectangle(img3, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray3[y:y+h, x:x+w]
    roi_color = img3[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.5)
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    smiles = smile_cascade.detectMultiScale(roi_gray, 2.5, minNeighbors=30)
    for (sx, sy, sw, sh) in smiles:
        cv.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)

img4 = cv.imread('img4.jpg')
gray4 = cv.cvtColor(img4, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray4, 1.5, minNeighbors=2)
for (x, y, w, h) in faces:
    img4 = cv.rectangle(img4, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray4[y:y+h, x:x+w]
    roi_color = img4[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray, 1.5, minNeighbors=18)
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    smiles = smile_cascade.detectMultiScale(roi_gray, 2.5, minNeighbors=30)
    for (sx, sy, sw, sh) in smiles:
        cv.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)

img5 = cv.imread('img5.png')
gray5 = cv.cvtColor(img5, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray5, 1.5, minNeighbors=2)
for (x, y, w, h) in faces:
    img5 = cv.rectangle(img5, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_gray = gray5[y:y+h, x:x+w]
    roi_color = img5[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    smiles = smile_cascade.detectMultiScale(roi_gray, minNeighbors=30)
    for (sx, sy, sw, sh) in smiles:
        cv.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)

imgs = [img, img2, img3, img4, img5]
for i in range(5):
    cv.imshow('img', imgs[i])
    cv.waitKey(0)
    cv.destroyAllWindows()

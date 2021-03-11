from itertools import chain
from matplotlib import pyplot as plt
from matplotlib.pyplot import annotate
from sklearn.datasets import fetch_lfw_people, make_classification
from skimage import data, color, transform, feature
from sklearn.feature_extraction.image import PatchExtractor
import numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm, metrics, datasets
from sklearn.svm import SVC, LinearSVC
import tensorflow as tf

# loading faces in the wild dataset
faces = fetch_lfw_people()
# saving the faces as positive patches
positive_patches = faces.images
# calling some categories from the database as negative patches
imgs_to_use = ['camera', 'text', 'coins', 'moon', 'page', 'clock', 'immunohistochemistry',
               'chelsea', 'coffee', 'hubble_deep_field']
# loading the negative patches
images = [color.rgb2gray(getattr(data, name)()) for name in imgs_to_use]


# defining a function to extract patches and reshape them to the images size
def extract_patches(img, N, scale=1.0, patch_size=positive_patches[0].shape):
    extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))
    extractor = PatchExtractor(patch_size=extracted_patch_size, max_patches=N, random_state=0)
    patches = extractor.transform(img[np.newaxis])
    if scale != 1:
        patches = np.array([transform.resize(patch, patch_size) for patch in patches])
    return patches


negative_patches = np.vstack([extract_patches(im, 1000, scale) for im in images for scale in [.5, 1.0, 2.0]])
# merging two parts together
tot_patches = np.concatenate([positive_patches, negative_patches])

# extracting the HOG feature of the whole database
X = np.array([feature.hog(img) for img in chain(tot_patches)])

#        Labeling The Dataset
p_label = np.ones(positive_patches.shape[0])
n_label = np.zeros(negative_patches.shape[0])
label = np.concatenate([p_label, n_label])

#        Splitting the dataset
X_train, X_test, l_train, l_test = train_test_split(X, label, train_size=.8)

####        Training SVM Classifier
params = {'C': [.01, .1, 1, 2]}
# applying the gridsearchcv to find the optimum parameters
clf = GridSearchCV(LinearSVC(), params)
clf.fit(X_train, l_train)

####       Applying the Classifier to dataset
l_pred = clf.predict(X_test)
# obtaining classification accuracy
print("Accuracy:", metrics.accuracy_score(l_test, l_pred))
# got classification accuracy of 99 percent

# Applying the classifier to test_images
im1 = cv.imread('test_image1.jpg', 1)
im2 = cv.imread('test_image2.jpg', 1)
im3 = cv.imread('test_image3.jpg', 1)
im4 = cv.imread('test_image4.png', 1)
im5 = cv.imread('test_image5.jpg', 1)
imgs = [im1, im2, im3, im4, im5]


# defining a function to scan an image
def sliding_window(image, stepsize, windowsize):
    # slide a window across the image
    if windowsize is None:
        windowsize = [62, 47]
    for y in range(0, image.shape[0], stepsize):
        for x in range(0, image.shape[1], stepsize):
            # yield the current window
            image1 = image[y:y + windowsize[0], x:x + windowsize[1]]
            image[y:y + windowsize[0], x:x + windowsize[1]] = image1
            yield x, y, image1


# obtaining image pyramids to perform face detection
d = 4
pyramid = tuple(transform.pyramid_gaussian(imgs[1], max_layer=-1, downscale=d, multichannel=True))
score = []
box = []
for p in range(2, len(pyramid)):
    py = pyramid[p]
    for (x1, y1, window) in sliding_window(py, stepsize=4, windowsize=None):
        if window.shape[0] != 62 or window.shape[1] != 47:
            continue
        # saving the hof feature vector of the window
        X_t = [feature.hog(window)]
        # feeding the vector to the trained classifier
        l_p = clf.predict(X_t)
        # scoring the candidate window based on its probability and accuracy
        d_s = clf.decision_function(X_t)
        if l_p == 1 and d_s >= 0:
            # saving the score and prospective window
            score.append(d_s[0])
            # reversing the coordinates to draw the rectangle in the original image
            # (d is the downscale, p is the downscale sessions)
            box.append([y1 * d * p, x1 * d * p, (y1 + 62) * d * p, (x1 + 47) * d * p])
            cv.rectangle(imgs[1], (x1 * d * p, y1 * d * p), ((x1 + 47) * d * p, (y1 + 62) * d * p), (0, 0, 255), 1)
            cv.imshow('1', imgs[1])
            cv.waitKey(0)
            cv.imwrite('im.jpg', imgs[1])
        else:
            continue
# converting the python lists to tensor
#sess = tf.compat.v1.Session()
scores = tf.convert_to_tensor(score, np.float32)
boxes = tf.convert_to_tensor(box, np.float32)
# applying non maximum sup. to pick the window which best encompasses the overlapping windows
NMS = tf.image.non_max_suppression(boxes, scores, max_output_size=10, iou_threshold=0.5, score_threshold=float('-inf'))
# saving the coordinates of box
#new_boxes = tf.cast(tf.gather(boxes, NMS), tf.int32)
new_boxes = tf.gather(boxes, NMS)
#a = sess.run(new_boxes)
a = new_boxes[0]
cv.rectangle(imgs[1], (a[1], a[0]), (a[3] - a[1], a[2] - a[0]), (0, 255, 0), 2)
cv.imshow('1', imgs[1])
cv.waitKey(0)

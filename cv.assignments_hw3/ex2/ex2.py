from skimage import data, transform
from skimage.feature import local_binary_pattern
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# loading the necessary images
brick = data.brick()
grass = data.grass()
gravel = data.gravel()
# setting local binary pattern parameters
radius = 2
n_points = 8 * radius

# defining a scoring function
def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))

# defining a function to create histogram based on found features in the descriptors
# then compare the histogram to the criteria lbp to determine the group
def match(refs, img):
    best_score = 10
    best_name = None
    lbp = local_binary_pattern(img, n_points, radius, 'uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    for name, ref in refs.items():
        ref_hist, _ = np.histogram(ref, density=True, bins=n_bins,
                                   range=(0, n_bins))
        score = kullback_leibler_divergence(hist, ref_hist)
        if score < best_score:
            best_score = score
            best_name = name
    return best_name

# extracting and saving the LBP descriptors
refs = {
    'brick': local_binary_pattern(brick, n_points, radius, 'uniform'),
    'grass': local_binary_pattern(grass, n_points, radius, 'uniform'),
    'gravel': local_binary_pattern(gravel, n_points, radius, 'uniform')}

# classify rotated textures and testing its robustness to rotation
print('Rotated images matched against references using LBP:')
print('original: brick, rotated: 30deg, match result: ',
      match(refs, transform.rotate(brick, angle=30, resize=False)))
print('original: brick, rotated: 70deg, match result: ',
      match(refs, transform.rotate(brick, angle=70, resize=False)))
print('original: grass, rotated: 145deg, match result: ',
      match(refs, transform.rotate(grass, angle=145, resize=False)))


# plot histograms of LBP of textures
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3,
                                                       figsize=(9, 6))
plt.gray()

ax1.imshow(brick)
ax1.axis('off')
ax4.hist(refs['brick'])
ax4.set_ylabel('Percentage')

ax2.imshow(grass)
ax2.axis('off')
ax5.hist(refs['grass'])
ax5.set_xlabel('Uniform LBP values')

ax3.imshow(gravel)
ax3.axis('off')
ax6.hist(refs['gravel'])

plt.show()
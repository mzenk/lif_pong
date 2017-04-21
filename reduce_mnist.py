from __future__ import division
import numpy as np
import copy, cPickle, gzip
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

def downsample(set):
    result = np.zeros((set.shape[0], int((set.shape[1]/2)**2)))
    for i,img in enumerate(set):
        tmp = gaussian_filter(img, 1.)
        tmp = tmp[::2,::2]
        result[i,:] = tmp.flatten()
    print np.min(result), np.max(result)\
    return result

# Load MNIST and make smaller
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
# first test on smaller data set
train_images =  np.reshape(train_set[0], (train_set[0].shape[0], 28,28))
test_images =  np.reshape(test_set[0], (test_set[0].shape[0], 28,28))
valid_images =  np.reshape(valid_set[0], (valid_set[0].shape[0], 28,28))

small_train_set = downsample(train_images)
np.save("small_train", small_train_set)
small_test_set = downsample(test_images)
np.save("small_test", small_test_set)
small_valid_set = downsample(valid_images)
np.save("small_valid", small_valid_set)

inspect data    
plt.imshow(train_images.reshape(train_images.shape[0], 28,28)[10,:],\
 interpolation='Nearest', cmap='gray')
plt.figure()
plt.imshow(small_train_set.reshape(train_images.shape[0], 14,14)[10,:],\
 interpolation='Nearest', cmap='gray')
plt.show()
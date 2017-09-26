# script for analyzing sampled data
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../')
from training.rbm import RBM, CRBM
from utils.data_mgmt import make_figure_folder, load_images, load_rbm, get_data_path

# Load rbm and data

# MNIST
img_shape = (28, 28)
import gzip
f = gzip.open('../shared_data/datasets/mnist.pkl.gz', 'rb')
_, _, test_set = np.load(f)
f.close()
rbm = load_rbm('mnist_disc_rbm')
nv = rbm.n_visible
nl = rbm.n_labels
sample_file = get_data_path('lif_classification') + \
    'mnist_classif_500samples.npz'

n_pixels = np.prod(img_shape)


with np.load(sample_file) as d:
    # samples.shape: ([n_instances], n_samples, n_units)
    samples = d['samples'].astype(float)
    if len(samples.shape) == 3:
        # take only one of the instances
        samples = samples[0]
    print('Loaded sample array with shape {}'.format(samples.shape))

vis_samples = samples[..., :n_pixels]
hid_samples = samples[..., nv:]
sample_imgs = vis_samples.reshape(vis_samples.shape[0], *img_shape)

test_img = np.ones(img_shape)
counter = 0
for i, s in enumerate(sample_imgs[1:]):
    if not np.all(np.isclose(s, test_img)):
        # if counter > 20:
        #     print('More than 20 wrong...')
        #     break
        # plt.figure()
        # plt.imshow(s - test_img, interpolation='Nearest', cmap='gray')
        # plt.colorbar()
        # plt.savefig(make_figure_folder() + 'wrong' + str(i) + '.png')
        # plt.close()
        counter += 1
print('{} of {} samples were clamped incorrectly'.format(
    counter, len(sample_imgs)))

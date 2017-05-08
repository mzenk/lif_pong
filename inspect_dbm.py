from __future__ import division
import numpy as np
import cPickle, gzip
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import util
from bm import Rbm, ClassRbm


# Load rbm and data
with open('saved_rbms/mnist_dbm.pkl', 'rb') as f:
    testdbm = cPickle.load(f)
f = gzip.open('datasets/mnist.pkl.gz', 'rb')
_, _, test_set = np.load(f)
f.close()
img_shape = (28, 28)
n_pixels = np.prod(img_shape)

# # For visual inspection of filters and samples
# tiled_filters = util.tile_raster_images(X=testdbm.w.T[:25, :n_pixels],
#                                         img_shape=img_shape,
#                                         tile_shape=(5, 5),
#                                         scale_rows_to_unit_interval=True,
#                                         output_pixel_vals=False)
# plt.figure()
# plt.imshow(tiled_filters, interpolation='Nearest', cmap='gray')
# plt.savefig('figures/filters.png')

# samples = testdbm.draw_samples(1e5)
# tiled_samples = util.tile_raster_images(samples[500::1000, :n_pixels],
#                                         img_shape=img_shape,
#                                         tile_shape=(10, 10),
#                                         scale_rows_to_unit_interval=True,
#                                         output_pixel_vals=False)

# plt.figure()
# plt.imshow(tiled_samples, interpolation='Nearest', cmap='gray')
# plt.savefig('./figures/samples.png')

# samples with partially clamped inputs
n_pxls = int(np.sqrt(testdbm.n_visible))
n_samples = 25
test_img = test_set[0][321]
# take half MNIST image
clamped_input = np.zeros((n_pxls, n_pxls))
clamped_input[:, :n_pxls//2] = 1
clamped_input = clamped_input.flatten()
clamped_ind = np.nonzero(clamped_input == 1)[0]
clamped_input = test_img[clamped_ind]

clamped = [None] * (testdbm.n_visible + len(testdbm.hidden_layers))
clamped[0] = clamped_ind
clamped_val = [None] * (testdbm.n_visible + len(testdbm.hidden_layers))
clamped_val[0] = clamped_input

samples = testdbm.draw_samples(100 + n_samples, clamped=clamped,
                               clamped_val=clamped_val)

tiled_clamped = util.tile_raster_images(samples[100:],
                                        img_shape=(n_pxls, n_pxls),
                                        tile_shape=(5, 5),
                                        scale_rows_to_unit_interval=True,
                                        output_pixel_vals=False)

plt.figure()
plt.imshow(tiled_clamped, interpolation='Nearest', cmap='gray')
plt.savefig('./figures/clamped.png')

inferred_img = np.tile(samples[100], (3, 1)).T
inferred_img[:, [0, 2]] = 0
inferred_img[clamped_ind, :] = np.tile(test_img[clamped_ind], (3, 1)).T
inferred_img = inferred_img.reshape((n_pxls, n_pxls, 3))
plt.imshow(inferred_img, cmap='gray', interpolation='Nearest')
plt.savefig('figures/clamped_rgb.png')

# # sample_with_clamped_units from labels --- only for crbms
# n_pxls = int(np.sqrt(testdbm.n_inputs))
# clamped_samples = testdbm.sample_from_label(9,1000)[::40]
# tiled_clamped = util.tile_raster_images(clamped_samples.astype(float),
#                         img_shape=(n_pxls,n_pxls),
#                         tile_shape=(5,5),
#                         scale_rows_to_unit_interval=True,
#                         output_pixel_vals=False)

# plt.figure()
# plt.imshow(tiled_clamped, interpolation='Nearest', cmap='gray')
# plt.savefig('./figures/clamped.png')

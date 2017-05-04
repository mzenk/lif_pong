from __future__ import division
import numpy as np
import cPickle, gzip
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import util
from bm import Rbm, ClassRbm


# Load rbm and data
with open('saved_rbms/mnist_dbn.pkl', 'rb') as f:
    testrbm = cPickle.load(f)
# for dbn look at bottom BM
# testrbm = testrbm.rbms[0]
f = gzip.open('datasets/mnist.pkl.gz', 'rb')
_, _, test_set = np.load(f)
f.close()
img_shape = (28, 28)
n_pixels = np.prod(img_shape)
# # weight histogram
# plt.figure()
# plt.hist(testrbm.w.flatten(), 50)
# plt.savefig('./figures/weight_histo.png')

# # For visual inspection of filters and samples
# tiled_filters = util.tile_raster_images(X=testrbm.w.T[:25, :n_pixels],
#                                         img_shape=img_shape,
#                                         tile_shape=(5, 5),
#                                         scale_rows_to_unit_interval=True,
#                                         output_pixel_vals=False)
# plt.figure()
# plt.imshow(tiled_filters, interpolation='Nearest', cmap='gray')
# plt.savefig('figures/filters.png')

samples = testrbm.draw_samples(1e5)
tiled_samples = util.tile_raster_images(samples[500::1000, :n_pixels],
                                        img_shape=img_shape,
                                        tile_shape=(10, 10),
                                        scale_rows_to_unit_interval=True,
                                        output_pixel_vals=False)

plt.figure()
plt.imshow(tiled_samples, interpolation='Nearest', cmap='gray')
plt.savefig('./figures/samples.png')

# # samples with partially clamped inputs
# n_pxls = int(np.sqrt(testrbm.n_visible))
# n_samples = 25

# # design clamped image pixels
# clamped_input = np.zeros((n_pxls, n_pxls))
# # clamped_input[n_pxls//2 - 1: n_pxls//2 + 1, n_pxls//2 - 3: n_pxls//2 + 3] = 1
# clamped_input[n_pxls//2 - 4: n_pxls//2 + 4, n_pxls//2 - 7: n_pxls//2 - 5] = 1
# clamped_input = clamped_input.flatten()
# clamped_ind = np.nonzero(clamped_input == 1)[0]
# clamped_input = clamped_input[np.nonzero(clamped_input)]

# clamped_samples = \
#     testrbm.sample_with_clamped_units(100 + n_samples, clamped_ind,
#                                       clamped_input)[100:]
# inferred_imgs = np.zeros((n_samples, n_pxls**2))
# inferred_imgs[:, clamped_ind] = clamped_input
# inferred_imgs[:, np.setdiff1d(np.arange(testrbm.n_visible),
#                               clamped_ind)] = clamped_samples

# tiled_clamped = util.tile_raster_images(inferred_imgs.astype(float),
#                                         img_shape=(n_pxls, n_pxls),
#                                         tile_shape=(5, 5),
#                                         scale_rows_to_unit_interval=True,
#                                         output_pixel_vals=False)

# plt.figure()
# plt.imshow(tiled_clamped, interpolation='Nearest', cmap='gray')
# plt.savefig('./figures/clamped.png')

# # sample_with_clamped_units from labels --- only for crbms
# n_pxls = int(np.sqrt(testrbm.n_inputs))
# clamped_samples = testrbm.sample_from_label(9,1000)[::40]
# tiled_clamped = util.tile_raster_images(clamped_samples.astype(float),
#                         img_shape=(n_pxls,n_pxls),
#                         tile_shape=(5,5),
#                         scale_rows_to_unit_interval=True,
#                         output_pixel_vals=False)

# plt.figure()
# plt.imshow(tiled_clamped, interpolation='Nearest', cmap='gray')
# plt.savefig('./figures/clamped.png')

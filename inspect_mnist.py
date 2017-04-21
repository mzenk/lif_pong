from __future__ import division
import numpy as np
import cPickle, gzip
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import util
from bm import Rbm, ClassRbm


# Load rbm
with open('mnist_gen_rbm.pkl', 'rb') as f:
    testrbm = cPickle.load(f)

# pong pattern completion
# Load Pong data
f = gzip.open('datasets/mnist.pkl.gz', 'rb')
_, _, test_set = np.load(f)
f.close()

# # weight histogram
# plt.figure()
# plt.hist(testrbm.w.flatten(), 50)
# plt.savefig('weight_histo.png')

# # For visual inspection of filters and samples
# #filters
# tiled_filters = util.tile_raster_images(X=testrbm.w.T[:25,:],
#                         img_shape=(n_pxls,n_pxls),
#                         tile_shape=(5,5),
#                         scale_rows_to_unit_interval=True,
#                         output_pixel_vals=False)
# plt.figure()
# plt.imshow(tiled_filters, interpolation='Nearest', cmap='gray')
# plt.savefig('filters.png')

n_pxls = int(np.sqrt(testrbm.n_visible))
# samples
samples = testrbm.draw_samples(10000, ast=True)[500::100, :testrbm.n_visible]
tiled_samples = util.tile_raster_images(samples,
                                        img_shape=(n_pxls, n_pxls),
                                        tile_shape=(10, samples.shape[0]//10),
                                        scale_rows_to_unit_interval=True,
                                        output_pixel_vals=False)

plt.figure()
plt.imshow(tiled_samples, interpolation='Nearest', cmap='gray')
plt.savefig('samples.png')

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
# plt.savefig('clamped.png')

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
# plt.savefig('clamped.png')

from __future__ import division
import numpy as np
import cPickle, gzip
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import util


# Load rbm and data
with open('saved_rbms/mnist_gen_rbm.pkl', 'rb') as f:
    testrbm = cPickle.load(f)
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
#                                         scale_rows_to_unit_interval=False,
#                                         output_pixel_vals=False)
# plt.figure()
# plt.imshow(tiled_filters, interpolation='Nearest', cmap='gray')
# plt.savefig('figures/filters.png')

# samples = testrbm.draw_samples(1e5)
# tiled_samples = util.tile_raster_images(samples[500::1000, :n_pixels],
#                                         img_shape=img_shape,
#                                         tile_shape=(10, 10),
#                                         scale_rows_to_unit_interval=False,
#                                         output_pixel_vals=False)

# plt.figure()
# plt.imshow(tiled_samples, interpolation='Nearest', cmap='gray')
# plt.savefig('./figures/samples.png')

# samples with partially clamped inputs
pxls_x = int(np.sqrt(testrbm.n_visible))
n_samples = 100
burn_in = 100
# # design clamped image pixels
# clamped_input = np.zeros((pxls_x, pxls_x))
# # clamped_input[pxls_x//2 - 1: pxls_x//2 + 1, pxls_x//2 - 3: pxls_x//2 + 3] = 1
# clamped_input[pxls_x//2 - 4: pxls_x//2 + 4, pxls_x//2 - 7: pxls_x//2 - 5] = 1
# clamped_input = clamped_input.flatten()
# clamped_ind = np.nonzero(clamped_input == 1)[0]
# clamped_input = clamped_input[np.nonzero(clamped_input)]

# take half MNIST image
test_img = test_set[0][test_set[1] == 6][1]
clamped_input = np.zeros((pxls_x, pxls_x))
clamped_input[:pxls_x//2, :] = 1
clamped_input = clamped_input.flatten()
clamped_ind = np.nonzero(clamped_input == 1)[0]
clamped_input = test_img[clamped_ind]

clamped_samples = \
    testrbm.sample_with_clamped_units(burn_in + n_samples, clamped_ind,
                                      clamped_input)[burn_in:]

inferred_imgs = np.zeros((n_samples, pxls_x**2))
inferred_imgs[:, clamped_ind] = clamped_input
inferred_imgs[:, np.setdiff1d(np.arange(testrbm.n_visible),
                              clamped_ind)] = clamped_samples

tiled_clamped = util.tile_raster_images(inferred_imgs,
                                        img_shape=(pxls_x, pxls_x),
                                        tile_shape=(5, 5),
                                        scale_rows_to_unit_interval=False,
                                        output_pixel_vals=False)

plt.figure()
plt.imshow(tiled_clamped, interpolation='Nearest', cmap='gray')
plt.savefig('./figures/clamped.png')

inferred_img = np.tile(inferred_imgs[-1], (3, 1)).T
inferred_img[:, [0, 2]] = 0
inferred_img[clamped_ind, :] = np.tile(test_img[clamped_ind], (3, 1)).T
inferred_img = inferred_img.reshape((pxls_x, pxls_x, 3))
plt.figure()
plt.subplot(121)
plt.imshow(inferred_img, cmap='gray', interpolation='Nearest')
plt.subplot(122)
plt.imshow(test_img.reshape(pxls_x, pxls_x),
           cmap='gray', interpolation='Nearest')
plt.savefig('figures/clamped_rgb.png')

# # sample_with_clamped_units from labels --- only for crbms
# pxls_x = int(np.sqrt(testrbm.n_inputs))
# clamped_samples = testrbm.sample_from_label(0, 1000)
# for i in range(1, 10):
#     new_samples = testrbm.sample_from_label(i, 1000)
#     clamped_samples = np.concatenate((clamped_samples, new_samples), axis=0)

# tiled_clamped = util.tile_raster_images(clamped_samples[100::200],
#                                         img_shape=(pxls_x, pxls_x),
#                                         tile_shape=(5, 10),
#                                         scale_rows_to_unit_interval=False,
#                                         output_pixel_vals=False)

# plt.figure()
# plt.imshow(tiled_clamped, interpolation='Nearest', cmap='gray')
# plt.savefig('./figures/clamped_labels.png')

# # classification performance
# test_data = test_set[0]
# test_targets = test_set[1]
# prediction = testrbm.classify(test_data)
# crate = 100 * np.average(prediction == test_targets)
# print('Correct predictions: {:.2f} %'.format(crate))

from __future__ import division
import numpy as np
import cPickle, gzip
import matplotlib
import util
matplotlib.use('agg')
import matplotlib.pyplot as plt


# Load rbm and data
# MNIST
with open('saved_rbms/mnist_cdbm_mf.pkl', 'rb') as f:
    testdbm = cPickle.load(f)
img_shape = (28, 28)
f = gzip.open('datasets/mnist.pkl.gz', 'rb')
train_set, _, test_set = np.load(f)
f.close()

# # Pong
# with open('saved_rbms/mnist_cdbm.pkl', 'rb') as f:
#     testdbm = cPickle.load(f)

# img_shape = (36, 48)
# data_name = 'gauss_var_start{}x{}'.format(*img_shape)
# with np.load('datasets/' + data_name + '.npz') as d:
#     train_set, _, test_set = d[d.keys()[0]]

n_pixels = np.prod(img_shape)

# Visual inspection of filters and samples
w_l0 = testdbm.weights[0]
tiled_filters = util.tile_raster_images(X=w_l0.T[:25],
                                        img_shape=img_shape,
                                        tile_shape=(5, 5),
                                        scale_rows_to_unit_interval=True,
                                        output_pixel_vals=False)
plt.figure()
plt.imshow(tiled_filters, interpolation='Nearest', cmap='gray')
plt.savefig('figures/filters_l0.png')

samples = testdbm.draw_samples(1e5)
tiled_samples = util.tile_raster_images(samples[500::1000, :n_pixels],
                                        img_shape=img_shape,
                                        tile_shape=(10, 10),
                                        scale_rows_to_unit_interval=True,
                                        output_pixel_vals=False)

plt.figure()
plt.imshow(tiled_samples, interpolation='Nearest', cmap='gray')
plt.savefig('./figures/samples.png')

# # samples with partially clamped inputs
# imgsize_x = int(np.sqrt(testdbm.n_visible))
# n_samples = 1000
# test_img = test_set[0][416]
# # take half MNIST image
# clamped_input = np.zeros((imgsize_x, imgsize_x))
# clamped_input[:-10, :] = 1
# clamped_input = clamped_input.flatten()
# clamped_ind = np.nonzero(clamped_input == 1)[0]
# clamped_input = test_img[clamped_ind]

# clamped = [None] * (testdbm.n_layers + 1)
# clamped[0] = clamped_ind
# clamped_val = [None] * (testdbm.n_layers + 1)
# clamped_val[0] = clamped_input

# samples = testdbm.draw_samples(100 + n_samples, clamped=clamped,
#                                clamped_val=clamped_val)

# tiled_clamped = util.tile_raster_images(samples[100::40],
#                                         img_shape=(imgsize_x, imgsize_x),
#                                         tile_shape=(5, 5),
#                                         scale_rows_to_unit_interval=True,
#                                         output_pixel_vals=False)

# plt.figure()
# plt.imshow(tiled_clamped, interpolation='Nearest', cmap='gray')
# plt.savefig('./figures/clamped.png')

# inferred_img = np.tile(samples[-1], (3, 1)).T
# inferred_img[:, [0, 2]] = 0
# inferred_img[clamped_ind, :] = np.tile(test_img[clamped_ind], (3, 1)).T
# inferred_img = inferred_img.reshape((imgsize_x, imgsize_x, 3))
# plt.imshow(inferred_img, cmap='gray', interpolation='Nearest')
# plt.savefig('figures/clamped_rgb.png')

# clamped labels --- only for cdbm
n_samples = 3000
clamped = [None] * (1 + testdbm.n_layers)
clamped[-1] = np.arange(testdbm.hidden_layers[-1])
clamped_val = [None] * (1 + testdbm.n_layers)
clamped_val[-1] = util.to_1_of_c(np.array([0]), 10)
# init_v = test_set[0][test_set[1] == 4][0]
samples = testdbm.draw_samples(n_samples, clamped=clamped,
                               clamped_val=clamped_val)
for i in range(1, 10):
    clamped_val[-1] = util.to_1_of_c(np.array([i]), 10)
    new_samples = testdbm.draw_samples(100 + n_samples, clamped=clamped,
                                       clamped_val=clamped_val)
    samples = np.concatenate((samples, new_samples), axis=0)

tiled_clamped = util.tile_raster_images(samples[::600],
                                        img_shape=img_shape,
                                        tile_shape=(5, 10),
                                        scale_rows_to_unit_interval=True,
                                        output_pixel_vals=False)

plt.figure()
plt.imshow(tiled_clamped, interpolation='Nearest', cmap='gray')
plt.savefig('./figures/clamped_labels.png')

# # classification performance
# test_data = test_set[0]
# test_targets = test_set[1]
# prediction = testdbm.classify(test_data)
# crate = 100 * np.average(prediction == test_targets)
# print('Correct predictions: {:.2f} %'.format(crate))

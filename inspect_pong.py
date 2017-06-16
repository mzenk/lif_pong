from __future__ import division
from __future__ import print_function
import numpy as np
import cPickle
from util import tile_raster_images, to_1_of_c
from rbm import RBM, CRBM
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt


# Load Pong data
img_shape = (36, 48)
data_name = 'gauss_var_start{}x{}'.format(*img_shape)
data_name = 'pong_var_overlap'
with np.load('datasets/' + data_name + '.npz') as d:
    train_set, _, test_set = d[d.keys()[0]]

assert np.prod(img_shape) == train_set[0].shape[1]

print('Number of samples: {}, {}'.format(train_set[0].shape[0],
                                         test_set[0].shape[0]))


# # testing of L2
# fname = 'gauss_uncover{}w{}s'.format(100, 100)
# with np.load('figures/' + fname + '.npz') as d:
#     correct_predictions, distances, dist_std, \
#         img_diff, img_diff_std = d[d.keys()[0]]

# imgs = train_set[0]
# l2_diffa = np.sqrt(np.sum((imgs - np.random.rand(*imgs.shape))**2, axis=1))/imgs.shape[1]
# l2_diffb = np.sqrt(np.sum(np.diff(imgs, axis=1)**2, axis=1))/imgs.shape[1]

# plt.figure(figsize=(14, 7))
# plt.subplot(121)
# plt.title('||train images - uniform random noise||2')
# plt.hist(l2_diffa, bins=100)
# plt.subplot(122)
# plt.title('||train image - next train image||2')
# plt.hist(l2_diffb, bins=100)
# plt.plot(img_diff, [400]*20, 'ro')
# plt.savefig('testl2.png')

# # inspect data
# samples = tile_raster_images(train_set[0][:1600:100],
#                              img_shape=img_shape,
#                              tile_shape=(4, 4),
#                              tile_spacing=(1, 1),
#                              scale_rows_to_unit_interval=True,
#                              output_pixel_vals=False)

# plt.figure()
# plt.imshow(samples, interpolation='Nearest', cmap='gray', origin='lower')
# plt.savefig('figures/' + data_name + '_trainsamples.png')

# # Load rbm
# discriminative = True
# if discriminative:
#     rbm_name = data_name + '_crbm.pkl'
# else:
#     rbm_name = data_name + '_rbm.pkl'
# with open('saved_rbms/' + rbm_name, 'rb') as f:
#     testrbm = cPickle.load(f)

# # filters
# rand_ind = np.random.randint(testrbm.n_hidden, size=25)
# tiled_filters = tile_raster_images(X=testrbm.w.T[rand_ind, :testrbm.n_inputs],
#                                    img_shape=img_shape,
#                                    tile_shape=(5, 5),
#                                    tile_spacing=(1, 1),
#                                    scale_rows_to_unit_interval=False,
#                                    output_pixel_vals=False)
# plt.figure()
# plt.imshow(tiled_filters, interpolation='Nearest', cmap='gray')
# plt.colorbar()
# plt.savefig('figures/' + data_name + '_filters.png')

# # weight histogram
# plt.figure(figsize=(14, 7))
# plt.subplot(121)
# plt.hist(testrbm.w[:testrbm.n_inputs].flatten(), 50)
# plt.title('Visible weights')
# plt.subplot(122)
# plt.hist(testrbm.w[testrbm.n_inputs:].flatten(), 50)
# plt.title('Label weights')
# plt.savefig('figures/' + data_name + 'weights_histo.png')

# # inspect label placement
# imgs = train_set[0].reshape((train_set[0].shape[0], img_shape[0], img_shape[1]))
# print('n_labels = ' + str(train_set[1].shape[1]))
# labels = np.repeat(train_set[1], 3, axis=1)
# a = np.concatenate((imgs, np.expand_dims(labels, 2)), axis=2)

# tiled_a = tile_raster_images(a[:20],
#                              img_shape=a.shape[1:],
#                              tile_shape=(4, 5),
#                              tile_spacing=(1, 1),
#                              scale_rows_to_unit_interval=False,
#                              output_pixel_vals=False)

# plt.figure()
# plt.imshow(tiled_a, interpolation='Nearest', cmap='gray', origin='lower')
# plt.savefig('figures/asdf.png')

# Check the probability of each hidden unit being active given instances of a
# minibatch
# tbd

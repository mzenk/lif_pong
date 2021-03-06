from __future__ import division
import numpy as np
import os
import sys
from lif_pong.utils import tile_raster_images
from lif_pong.utils.data_mgmt import load_images, get_rbm_dict, make_figure_folder
from lif_pong.sampling.lif_inspect_samples import plot_samples
import rbm as rbm_pkg
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def draw_samples(rbm, data, n_pixels, n_chains, n_samples, ast=False):
    if data is None:
        v_init = None
    else:
        np.random.seed(123456)
        rand_idx = np.random.choice(range(len(data)), size=n_chains,
                                    replace=False)
        v_init = np.hstack((data[rand_idx], np.zeros((len(rand_idx), rbm.n_labels))))

    # for i in range(len(v_init)):
    #     plt.figure()
    #     plt.imshow(v_init[i, :n_pixels].reshape(img_shape))
    #     plt.savefig('test{}.png'.format(i))
    if ast:
        samples = np.zeros((n_samples, n_chains, n_pixels))
        for i in range(n_chains):
            tmp = testrbm.draw_samples_ast(
                n_samples, v_init=v_init[i])[:, :n_pixels]
            samples[:, i] = tmp
    else:
        # # for testing
        # samples = np.zeros((n_samples, n_chains, n_pixels))
        # for i in range(n_chains):
        #     tmp = testrbm.draw_samples(
        #         n_samples, v_init=v_init[i])[:, :n_pixels]
        #     samples[:, i] = tmp
        samples = testrbm.draw_samples(n_samples,
                                       n_chains=n_chains, v_init=v_init)
    return samples


# # Load rbm and data
# import gzip, cPickle
# with open('../shared_data/saved_rbms/mnist_crbm.pkl', 'rb') as f:
#     rbm_dict = cPickle.load(f)
# testrbm = rbm_pkg.load(rbm_dict)
# with gzip.open('../shared_data/datasets/mnist.pkl.gz', 'rb') as f:
#     _, _, test_set = np.load(f)
# img_shape = (28, 28)

img_shape = (40, 48)
rbm_name = 'pong_lw5_40x48_crbm'
data_name = 'pong_lw5_40x48'
_, _, test_set = load_images(data_name)
testrbm = rbm_pkg.load(get_rbm_dict(rbm_name))
n_pixels = np.prod(img_shape)

testrbm.set_seed(1234)
n_chains = 5
n_cols = 10
samples_per_tile = 1000
tile_shape = (n_chains, n_cols)

# dreaming
n_samples = samples_per_tile*n_cols
samples = draw_samples(testrbm, test_set[0], n_pixels, n_chains, n_samples, ast=False)

# or load other sample file
plot_samples(samples, tile_shape, img_shape, samples_per_tile=samples_per_tile,
             titles=['chain {:1d}'.format(i + 1) for i in range(n_chains)],
             savename='pong_gibbs_samples')


# # samples with partially clamped inputs
# pxls_x = int(np.sqrt(testrbm.n_visible))
# n_samples = 100
# burn_in = 100
# # # design clamped image pixels
# # clamped_input = np.zeros((pxls_x, pxls_x))
# # # clamped_input[pxls_x//2 - 1: pxls_x//2 + 1, pxls_x//2 - 3: pxls_x//2 + 3] = 1
# # clamped_input[pxls_x//2 - 4: pxls_x//2 + 4, pxls_x//2 - 7: pxls_x//2 - 5] = 1
# # clamped_input = clamped_input.flatten()
# # clamped_ind = np.nonzero(clamped_input == 1)[0]
# # clamped_input = clamped_input[np.nonzero(clamped_input)]

# # take half MNIST image
# test_img = test_set[0][test_set[1] == 6][1]
# clamped_input = np.zeros((pxls_x, pxls_x))
# clamped_input[:pxls_x//2, :] = 1
# clamped_input = clamped_input.flatten()
# clamped_ind = np.nonzero(clamped_input == 1)[0]
# clamped_input = test_img[clamped_ind]

# clamped_samples = \
#     testrbm.sample_with_clamped_units(burn_in + n_samples, clamped_ind,
#                                       clamped_input)[burn_in:]

# inferred_imgs = np.zeros((n_samples, pxls_x**2))
# inferred_imgs[:, clamped_ind] = clamped_input
# inferred_imgs[:, np.setdiff1d(np.arange(testrbm.n_visible),
#                               clamped_ind)] = clamped_samples

# tiled_clamped = tile_raster_images(inferred_imgs,
#                                         img_shape=(pxls_x, pxls_x),
#                                         tile_shape=(5, 5),
#                                         scale_rows_to_unit_interval=False,
#                                         output_pixel_vals=False)

# plt.figure()
# plt.imshow(tiled_clamped, interpolation='Nearest', cmap='gray')
# plt.savefig('./figures/clamped.png')

# inferred_img = np.tile(inferred_imgs[-1], (3, 1)).T
# inferred_img[:, [0, 2]] = 0
# inferred_img[clamped_ind, :] = np.tile(test_img[clamped_ind], (3, 1)).T
# inferred_img = inferred_img.reshape((pxls_x, pxls_x, 3))
# plt.figure()
# plt.subplot(121)
# plt.imshow(inferred_img, cmap='gray', interpolation='Nearest')
# plt.subplot(122)
# plt.imshow(test_img.reshape(pxls_x, pxls_x),
#            cmap='gray', interpolation='Nearest')
# plt.savefig('figures/clamped_rgb.png')

# # sample_with_clamped_units from labels --- only for crbms
# pxls_x = int(np.sqrt(testrbm.n_inputs))
# clamped_samples = testrbm.sample_from_label(0, 1000)
# for i in range(1, 10):
#     new_samples = testrbm.sample_from_label(i, 1000)
#     clamped_samples = np.concatenate((clamped_samples, new_samples), axis=0)

# tiled_clamped = tile_raster_images(clamped_samples[100::200],
#                                         img_shape=(pxls_x, pxls_x),
#                                         tile_shape=(5, 10),
#                                         scale_rows_to_unit_interval=False,
#                                         output_pixel_vals=False)

# plt.figure()
# plt.imshow(tiled_clamped, interpolation='Nearest', cmap='gray')
# plt.savefig('./figures/clamped_labels.png')

# # classification performance
# test_data = test_set[0]
# if len(test_set[1].shape) == 2:
#     test_targets = np.argmax(test_set[1], axis=1)
# else:
#     test_targets = test_set[1]
# prediction = testrbm.classify(test_data)
# crate = 100 * np.average(prediction == test_targets)
# print('Correct predictions: {:.2f} %'.format(crate))

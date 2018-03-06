from __future__ import division
from __future__ import print_function
import numpy as np
import os
from lif_pong.utils import tile_raster_images, to_1_of_c
from lif_pong.utils.data_mgmt import load_images, get_rbm_dict, make_figure_folder
import rbm as rbm_pkg
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rcParams['font.size'] = 12


def plot_labvis_filters(rbm, label_idxs, img_shape, name='lab_filters',
                        title='Label filters', ax=None):
    hidden_act = 1./(1 + np.exp(-rbm.wv - rbm.hbias))
    filters = np.dot(rbm.wl, hidden_act.T) + rbm.lbias.reshape((-1, 1))
    tiled_filters = tile_raster_images(X=filters[label_idxs],
                                       img_shape=img_shape,
                                       tile_shape=(1, len(label_idxs)),
                                       tile_spacing=(1, 1),
                                       spacing_val=filters[label_idxs].min(),
                                       scale_rows_to_unit_interval=False,
                                       output_pixel_vals=False)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    im = ax.imshow(tiled_filters, interpolation='Nearest', cmap='gray')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fmin = np.ceil(tiled_filters.min())
    fmax = np.floor(tiled_filters.max())
    plt.colorbar(im, cax=cax, ticks=[fmin, .5*(fmin + fmax), fmax])
    # plt.colorbar(im, orientation='horizontal')
    ax.set_title(title)
    ax.tick_params(left='off', right='off', bottom='off',
                   labelleft='off', labelright='off', labelbottom='off')
    if fig is not None:
        fig.tight_layout()
        plt.savefig(os.path.join(make_figure_folder(), name + '_filters.png'))


def plot_filters(rbm, hidden_idxs, img_shape, tile_shape=(4, 4),
                 name='filters', title='Receptive fields'):
    tiled_filters = tile_raster_images(X=rbm.w.T[hidden_idxs, :rbm.n_inputs],
                                       img_shape=img_shape,
                                       tile_shape=tile_shape,
                                       tile_spacing=(1, 1),
                                       scale_rows_to_unit_interval=False,
                                       output_pixel_vals=False)
    fig, ax = plt.subplots()
    im = ax.imshow(tiled_filters, interpolation='Nearest', cmap='gray')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    # plt.colorbar(im)
    ax.set_title(title)
    ax.tick_params(left='off', right='off', bottom='off',
                   labelleft='off', labelright='off', labelbottom='off')
    fig.tight_layout()
    plt.savefig(os.path.join(make_figure_folder(), name + '_filters.png'))


def plot_histograms(rbm, name='histo'):
    # weight histogram
    plt.figure(figsize=(14, 7))
    plt.subplot(121)
    plt.hist(rbm.w[:rbm.n_inputs].flatten(), bins='auto')
    plt.title('Visible weights')
    plt.subplot(122)
    plt.hist(rbm.w[rbm.n_inputs:].flatten(), bins='auto')
    plt.title('Label weights')
    plt.tight_layout()
    plt.savefig(os.path.join(make_figure_folder(), name + '_weights.png'))

    # bias histogram
    plt.figure(figsize=(14, 7))
    plt.subplot(121)
    plt.hist(rbm.ibias, bins='auto', alpha=.7)
    plt.hist(rbm.lbias, bins='auto', alpha=.7)
    plt.title('Visible biases')
    plt.subplot(122)
    plt.hist(rbm.hbias, bins='auto')
    plt.title('Hidden biases')
    plt.tight_layout()
    plt.savefig(os.path.join(make_figure_folder(), name + '_biases.png'))


# Load data -- Pong
img_shape = (40, 48)
rbm_name = 'pong_lw5_40x48_crbm'
# data_name = 'gauss_var_start{}x{}'.format(*img_shape)
# train_set, _, test_set = load_images(data_name)
# assert np.prod(img_shape) == train_set[0].shape[1]
testrbm = rbm_pkg.load(get_rbm_dict(rbm_name))
# RBM-specific plots
rand_ind = np.random.choice(np.arange(testrbm.n_hidden), size=12, replace=False)
# plot_filters(testrbm, rand_ind, img_shape, tile_shape=(3, 4), name='gauss',
#              title='Gau\ss')
fig, ax = plt.subplots(2, 1, figsize=(14, 7))
plot_labvis_filters(testrbm, np.arange(2, 8, 2), img_shape, title='Pong',
                    ax=ax[0])

rbm_name = 'gauss_lw5_40x48_crbm'
testrbm = rbm_pkg.load(get_rbm_dict(rbm_name))
plot_labvis_filters(testrbm, np.arange(2, 8, 2), img_shape, title='Gau\ss',
                    ax=ax[1])
fig.tight_layout()
fig.savefig(os.path.join(make_figure_folder(), 'lab_filters.png'))
# plot_histograms(testrbm)

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

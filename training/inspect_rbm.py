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


def plot_labvis_filters(rbm, label_idxs, img_shape, name='lab_filters',
                        title='Label filters', ax=None):
    hidden_act = 1./(1 + np.exp(-rbm.wv - rbm.hbias))
    filters = np.dot(rbm.wl, hidden_act.T) + rbm.lbias.reshape((-1, 1))
    labwidth = int(img_shape[0] / rbm.n_labels)
    gray_imgs = filters[label_idxs].reshape((-1,) + img_shape)
    maxval = gray_imgs.max()
    minval = gray_imgs.min()
    ext_imgs = []
    for i, gimg in enumerate(gray_imgs):
        labvec = maxval * np.ones((gimg.shape[0], 1))
        labvec[label_idxs[i]*labwidth:(label_idxs[i] + 1)*labwidth] = minval
        ext_imgs.append(np.hstack((gimg, labvec)).flatten())
    ext_imgs = np.array(ext_imgs)
    tiled_filters = tile_raster_images(X=ext_imgs,
                                       img_shape=(img_shape[0], img_shape[1] + 1),
                                       tile_shape=(1, len(label_idxs)),
                                       tile_spacing=(1, 1),
                                       spacing_val=maxval,
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
    ax.tick_params(left='off', right='off', bottom='off', top='off',
                   labelleft='off', labelright='off', labelbottom='off')
    if fig is not None:
        plt.tight_layout()
        plt.savefig(os.path.join(make_figure_folder(), name + '_filters.png'))
        plt.savefig(os.path.join(make_figure_folder(), name + '_filters.pdf'))


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
    ax.tick_params(left='off', right='off', bottom='off', top='off',
                   labelleft='off', labelright='off', labelbottom='off')
    plt.tight_layout()
    plt.savefig(os.path.join(make_figure_folder(), name + '_filters.png'),
                bbox_inches='tight')
    plt.savefig(os.path.join(make_figure_folder(), name + '_filters.pdf'),
                bbox_inches='tight')


def plot_histograms(rbm, name='histo'):
    plt.style.use('mthesis_style')
    # weight histogram
    fig = plt.figure()
    fig.set_figheight(fig.get_figwidth()/1.8)
    plt.subplot(121)
    plt.hist(rbm.w[:rbm.n_inputs].flatten(), bins='auto', histtype='stepfilled')
    print('Max: {}, Min: {}'.format(rbm.w[:rbm.n_inputs].max(), rbm.w[:rbm.n_inputs].min()))
    plt.title('Visible to hidden')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.subplot(122)
    plt.hist(rbm.w[rbm.n_inputs:].flatten(), bins='auto', histtype='stepfilled')
    plt.title('Label to hidden')
    plt.xlabel('Weight')
    # plt.ylabel('Frequency')
    plt.subplots_adjust(wspace=0.3, right=.9, bottom=.12)
    plt.savefig(os.path.join(make_figure_folder(), name + '_weights.png'))
    plt.savefig(os.path.join(make_figure_folder(), name + '_weights.pdf'))

    # bias histogram
    fig = plt.figure()
    fig.set_figheight(fig.get_figwidth()/1.8)
    plt.subplot(121)
    plt.hist(rbm.ibias, bins='auto', histtype='stepfilled')
    # plt.hist(rbm.lbias, bins='auto', alpha=.9)
    plt.title('Visible neurons')
    plt.xlabel('Bias')
    plt.ylabel('Frequency')
    plt.subplot(122)
    plt.hist(rbm.hbias, bins='auto', histtype='stepfilled')
    plt.title('Hidden neurons')
    plt.xlabel('Bias')
    # plt.ylabel('Frequency')
    plt.subplots_adjust(wspace=0.3, right=.9, bottom=.12)
    plt.savefig(os.path.join(make_figure_folder(), name + '_biases.png'))
    plt.savefig(os.path.join(make_figure_folder(), name + '_biases.pdf'))


def visualize_biases(rbm, img_shape, name='test'):
    plt.style.use('mthesis_style')
    try:
        biases = rbm.ibias
    except AttributeError:
        biases = rbm.vbias
    fig = plt.figure()
    fig.set_figheight(.66*fig.get_figheight())
    fig.set_figwidth(.66*fig.get_figwidth())
    plt.title('Visible biases (Flat)')
    plt.imshow(biases.reshape(img_shape), interpolation='Nearest', cmap='gray_r')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(make_figure_folder(), name + '_visbiases.png'))
    plt.savefig(os.path.join(make_figure_folder(), name + '_visbiases.pdf'))


# Load data -- Pong
img_shape = (40, 48)
rbm_name = 'pong_lw5_40x48_crbm'
# data_name = 'gauss_var_start{}x{}'.format(*img_shape)
# train_set, _, test_set = load_images(data_name)
# assert np.prod(img_shape) == train_set[0].shape[1]
testrbm = rbm_pkg.load(get_rbm_dict(rbm_name))
# plot_histograms(testrbm, 'pong')
# visualize_biases(testrbm, img_shape)
rand_ind = np.random.choice(np.arange(testrbm.n_hidden), size=12, replace=False)
plot_filters(testrbm, rand_ind, img_shape, tile_shape=(3, 4), name='pong',
             title='Flat')
fig, ax = plt.subplots(2, 1, figsize=(12, 7))
plot_labvis_filters(testrbm, np.arange(2, 8, 2), img_shape, title='Flat',
                    ax=ax[0])

# rbm_name = 'gauss_lw5_40x48_crbm'
# testrbm = rbm_pkg.load(get_rbm_dict(rbm_name))
# # rand_ind = np.random.choice(np.arange(testrbm.n_hidden), size=12, replace=False)
# # plot_filters(testrbm, rand_ind, img_shape, tile_shape=(3, 4), name='gauss',
# #          title='Hill')
# plot_labvis_filters(testrbm, np.arange(2, 8, 2), img_shape, title='Hill',
#                     ax=ax[1])
# plt.tight_layout()
# fig.savefig(os.path.join(make_figure_folder(), 'lab_filters.png'))

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

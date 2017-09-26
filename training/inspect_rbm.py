from __future__ import division
from __future__ import print_function
import numpy as np
import cPickle
from utils import tile_raster_images, to_1_of_c
from utils.data_mgmt import load_images, load_rbm, make_figure_folder
from rbm import RBM, CRBM
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rcParams['font.size'] = 12


def plot_data(data, show_idx, img_shape, tile_shape=(5, 5), name='data'):
    samples = tile_raster_images(train_set[0][idx],
                                 img_shape=img_shape,
                                 tile_shape=(4, 4),
                                 tile_spacing=(1, 1),
                                 scale_rows_to_unit_interval=True,
                                 output_pixel_vals=False)

    plt.figure()
    plt.imshow(samples, interpolation='Nearest', cmap='gray')
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(make_figure_folder() + name + '.png')


def plot_receptive_fields(rbm, hidden_idx, img_shape, tile_shape=(4, 4),
                          name='filters', title='Receptive fields'):
    tiled_filters = tile_raster_images(X=rbm.w.T[hidden_idx, :rbm.n_inputs],
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
    ax.set_title(title)
    ax.tick_params(left='off', right='off', bottom='off',
                   labelleft='off', labelright='off', labelbottom='off')
    fig.tight_layout()
    plt.savefig(make_figure_folder() + name + '.png')


def plot_histograms(rbm, name='histo'):
    # weight histogram
    plt.figure(figsize=(14, 7))
    plt.subplot(121)
    plt.hist(testrbm.w[:testrbm.n_inputs].flatten(), bins='auto')
    plt.title('Visible weights')
    plt.subplot(122)
    plt.hist(testrbm.w[testrbm.n_inputs:].flatten(), bins='auto')
    plt.title('Label weights')
    plt.tight_layout()
    plt.savefig(make_figure_folder() + name + '_weights.pdf')

    # bias histogram
    plt.figure(figsize=(14, 7))
    plt.subplot(121)
    plt.hist(testrbm.ibias, bins='auto', alpha=.7)
    plt.hist(testrbm.lbias, bins='auto', alpha=.7)
    plt.title('Visible biases')
    plt.subplot(122)
    plt.hist(testrbm.hbias, bins='auto')
    plt.title('Hidden biases')
    plt.tight_layout()
    plt.savefig(make_figure_folder() + name + '_biases.pdf')


# # Load data -- Pong
# img_shape = (36, 48)
# data_name = 'pong_var_start{}x{}'.format(*img_shape)
# train_set, _, test_set = load_images(data_name)
# testrbm = load_rbm(data_name + '_crbm')
# Load data -- MNIST
import gzip
img_shape = (28, 28)
with gzip.open('../shared_data/datasets/mnist.pkl.gz', 'rb') as f:
    train_set, _, test_set = np.load(f)
testrbm = load_rbm('mnist_disc500_rbm')

assert np.prod(img_shape) == train_set[0].shape[1]

print('Number of samples: {}, {}'.format(train_set[0].shape[0],
                                         test_set[0].shape[0]))

# inspect data
np.random.seed(42)
idx = np.random.choice(np.arange(len(train_set[0])), size=25, replace=False)
idx = np.arange(25)
plot_data(train_set, idx, img_shape)

# RBM-specific plots
rand_ind = np.random.randint(testrbm.n_hidden, size=16)
# plot_receptive_fields(testrbm, rand_ind, img_shape)
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
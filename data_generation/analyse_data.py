from __future__ import division
from __future__ import print_function
import numpy as np
import os
import sys
from lif_pong.utils import tile_raster_images
from lif_pong.utils.data_mgmt import load_images, make_figure_folder
import matplotlib.pyplot as plt
plt.style.use('mthesis_style')


def plot_data_samples(data, data_idx, img_shape, name='data',
                      tile_shape=(4, 4), binary=False):
    if binary:
        data = (data > .5)*1.
        name_mod = '_bin'
    else:
        name_mod = ''
    samples = tile_raster_images(data,
                                 img_shape=img_shape,
                                 tile_shape=tile_shape,
                                 tile_spacing=(1, 1),
                                 scale_rows_to_unit_interval=False,
                                 output_pixel_vals=False)

    plt.figure()
    plt.imshow(samples, interpolation='Nearest', cmap='gray_r', origin='lower')
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(make_figure_folder(),
                             name + '_samples' + name_mod + '.png'))


def plot_pos_dist(data, img_shape, col):
    # compute histograms of positions at a specific column
    data_col = data.reshape((-1,) + img_shape)[..., col]

    positions = np.average(np.tile(np.arange(img_shape[0]), (len(data_col), 1)),
                           weights=data_col, axis=1)

    # compute histogram
    plot_startend(positions)


def plot_startend(positions):
    plt.figure()
    positions /= (positions.max() - positions.min())
    plt.hist(positions, bins='auto')
    plt.xlabel('Position in units of field height')
    plt.ylabel('Number')
    plt.savefig(os.path.join(make_figure_folder(), 'pos_histo.png'))


def plot_anglehisto(angles):
    plt.figure()
    plt.hist(angles, bins='auto')
    plt.xlabel('Position')
    plt.ylabel('Number')
    plt.savefig(os.path.join(make_figure_folder(), 'angle_histo.png'))


def plot_mean_data(data, img_shape):
    # plot the mean of all data samples
    plt.figure()
    plt.imshow(data.mean(axis=0).reshape(img_shape),
               interpolation='Nearest', cmap='gray_r')
    plt.savefig(os.path.join(make_figure_folder(), 'mean_img.png'))


if len(sys.argv) != 2:
    print('Wrong number of arguments. Please provide path to data-file.')
    sys.exit()

# load data and statistics
img_shape = (40, 48)
stat_file = '.'.join(sys.argv[1].split('.')[:-1]) + '_stats.npz'
with np.load(sys.argv[1]) as d:
    train_data = d['train_data']
    valid_data = d['valid_data']
    test_data = d['test_data']

with np.load(stat_file) as d:
    init_pos = d['init_pos']
    init_angles = d['init_angles']
    end_pos = d['end_pos']

ntrain = len(train_data)
nvalid = len(valid_data)
ntest = len(test_data)

print('Number of samples in train/valid/test set: {}, {}, {}'.format(
    ntrain, nvalid, ntest))

# inspect data
np.random.seed(12345)
idx = np.random.choice(np.arange(ntrain), size=min(25, ntrain), replace=False)
# plot_data_samples(train_data, idx, img_shape, binary=False, name='test')
# plot_startend(init_pos)
# plot_anglehisto(init_angles)
plot_mean_data(train_data, img_shape)

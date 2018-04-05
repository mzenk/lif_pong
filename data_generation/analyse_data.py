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
                             name + '_samples' + name_mod + '.pdf'))


def plot_pos_dist(data, img_shape, col):
    # compute histograms of positions at a specific column
    data_col = data.reshape((-1,) + img_shape)[..., col]

    positions = np.average(np.tile(np.arange(img_shape[0]), (len(data_col), 1)),
                           weights=data_col, axis=1)

    # compute histogram
    plot_startend(positions)


def plot_startend(positions, label=None):
    positions /= (positions.max() - positions.min())
    plt.hist(positions, bins='auto', histtype='stepfilled', label=label)
    plt.xlabel('Position in units of field height')
    plt.ylabel('Frequency')


def plot_anglehisto(angles, label=None):
    plt.hist(angles, bins='auto', histtype='stepfilled')
    plt.xlabel('Angle [degrees]')
    plt.ylabel('Frequency')


def plot_mean_data(data, img_shape, title=None):
    # plot the mean of all data samples
    fig = plt.figure()
    if title is not None:
        plt.title(title)
    fig.set_figheight(.66*fig.get_figheight())
    fig.set_figwidth(.66*fig.get_figwidth())
    plt.imshow(data.mean(axis=0).reshape(img_shape),
               interpolation='Nearest', cmap='gray_r')
    plt.tight_layout()
    plt.savefig(os.path.join(make_figure_folder(), 'mean_img.pdf'))


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

train_start_pos = init_pos[:ntrain]
train_end_pos = end_pos[:ntrain]
test_start_pos = init_pos[-ntest:]
test_end_pos = end_pos[-ntest:]

print('Number of samples in train/valid/test set: {}, {}, {}'.format(
    ntrain, nvalid, ntest))

# inspect data
np.random.seed(12345)
idx = np.random.choice(np.arange(ntrain), size=min(25, ntrain), replace=False)
# plot_data_samples(train_data, idx, img_shape, binary=False, name='test')

# plot start position histogram
fig = plt.figure()
fig.set_figheight(.5*fig.get_figheight())
fig.set_figwidth(.5*fig.get_figwidth())
plt.title('Start positions (Kink)')
plot_startend(train_start_pos, label='train')
plot_startend(test_start_pos, label='test')
plt.tight_layout()
# plt.legend(loc=4)
plt.savefig(os.path.join(make_figure_folder(), 'start_histo.pdf'))

# plot end position histogram
fig = plt.figure()
fig.set_figheight(.5*fig.get_figheight())
fig.set_figwidth(.5*fig.get_figwidth())
plt.title('End positions (Kink)')
plot_startend(train_end_pos, label='train')
plot_startend(test_end_pos, label='test')
plt.tight_layout()
# plt.legend(loc=4)
plt.savefig(os.path.join(make_figure_folder(), 'end_histo.pdf'))

# plot initial angles
fig = plt.figure()
plt.title('Start angles (Pong)')
fig.set_figheight(.66*fig.get_figheight())
fig.set_figwidth(.66*fig.get_figwidth())
plot_anglehisto(init_angles)
plt.tight_layout()
plt.savefig(os.path.join(make_figure_folder(), 'angle_histo.pdf'))

# plot mean data image
plot_mean_data(train_data, img_shape, title='Pong')

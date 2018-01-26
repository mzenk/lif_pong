from __future__ import division
from __future__ import print_function
import numpy as np
import os
from lif_pong.utils import tile_raster_images, to_1_of_c
from lif_pong.utils.data_mgmt import load_images, get_rbm_dict, make_figure_folder
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

mpl.rcParams['font.size'] = 12


def plot_data(data_set, show_idx, img_shape, tile_shape=(5, 5), binary=False):
    if binary:
        data = (data_set[0][idx] > .4)*1.
        name_mod = '_bin'
    else:
        data = data_set[0][idx]
        name_mod = ''
    samples = tile_raster_images(data,
                                 img_shape=img_shape,
                                 tile_shape=(4, 4),
                                 tile_spacing=(1, 1),
                                 scale_rows_to_unit_interval=True,
                                 output_pixel_vals=False)

    plt.figure()
    plt.imshow(samples, interpolation='Nearest', cmap='gray', origin='lower')
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(make_figure_folder(),
                             data_name + '_samples' + name_mod + '.png'))


# Load data -- Pong
img_shape = (36, 48)
data_name = 'knick_pos0.5_ampl0.5_var_start36x48'
train_set, valid_set, test_set = load_images(data_name)
# # Load data -- MNIST
# import gzip
# img_shape = (28, 28)
# with gzip.open('../shared_data/datasets/mnist.pkl.gz', 'rb') as f:
#     train_set, _, test_set = np.load(f)

assert np.prod(img_shape) == train_set[0].shape[1]

print('Number of samples in train/valid/test set: {}, {}, {}'.format(
    len(train_set[0]), len(valid_set[0]), len(test_set[0])))

# inspect data
data_set = test_set
np.random.seed(42)
idx = np.random.choice(np.arange(len(data_set[0])),
                       size=min(25, len(data_set[0])), replace=False)
plot_data(data_set, idx, img_shape, binary=True)

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

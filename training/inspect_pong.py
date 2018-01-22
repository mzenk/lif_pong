from __future__ import division
from __future__ import print_function
import numpy as np
from lif_pong.utils import tile_raster_images, to_1_of_c
from lif_pong.utils.data_mgmt import load_images, get_rbm_dict, make_figure_folder
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

mpl.rcParams['font.size'] = 12


def plot_data(data, show_idx, img_shape, tile_shape=(5, 5)):
    samples = tile_raster_images(train_set[0][idx],
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
    plt.savefig(make_figure_folder() + data_name + '_samples.png')


# Load data -- Pong
img_shape = (36, 48)
data_name = 'knick_pos0.5_ampl0.6_var_start36x48'
train_set, _, test_set = load_images(data_name)
# # Load data -- MNIST
# import gzip
# img_shape = (28, 28)
# with gzip.open('../shared_data/datasets/mnist.pkl.gz', 'rb') as f:
#     train_set, _, test_set = np.load(f)

assert np.prod(img_shape) == train_set[0].shape[1]

print('Number of samples: {}, {}'.format(train_set[0].shape[0],
                                         test_set[0].shape[0]))

# inspect data
np.random.seed(42)
idx = np.random.choice(np.arange(len(train_set[0])),
                       size=min(25, len(train_set[0])), replace=False)
plot_data(train_set, idx, img_shape)
print(train_set[0].shape)

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

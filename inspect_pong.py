from __future__ import division
from __future__ import print_function
import numpy as np
import cPickle
from util import tile_raster_images, to_1_of_c
from bm import Rbm, ClassRbm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


# Load Pong data
img_shape = (36, 48)
data_name = 'gauss_fixed_start{}x{}'.format(*img_shape)
with np.load('datasets/' + data_name + '.npz') as d:
    train_set, _, test_set = d[d.keys()[0]]

assert np.prod(img_shape) == train_set[0].shape[1]

print('Number of samples: {}, {}'.format(train_set[0].shape[0],
                                         test_set[0].shape[0]))

# inspect data
samples = tile_raster_images(train_set[0][:16],
                             img_shape=img_shape,
                             tile_shape=(4, 4),
                             tile_spacing=(1, 1),
                             scale_rows_to_unit_interval=True,
                             output_pixel_vals=False)

plt.figure()
plt.imshow(samples, interpolation='Nearest', cmap='gray', origin='lower')
plt.savefig(data_name + 'samples.png')

# Load rbm
discriminative = True
if discriminative:
    rbm_name = data_name + '_crbm.pkl'
else:
    rbm_name = data_name + '_rbm.pkl'
with open('saved_rbms/' + rbm_name, 'rb') as f:
    testrbm = cPickle.load(f)

# filters
rand_ind = np.random.randint(testrbm.n_hidden, size=25)
tiled_filters = tile_raster_images(X=testrbm.w.T[rand_ind, :testrbm.n_inputs],
                                   img_shape=img_shape,
                                   tile_shape=(5, 5),
                                   tile_spacing=(1, 1),
                                   scale_rows_to_unit_interval=False,
                                   output_pixel_vals=False)
plt.figure()
plt.imshow(tiled_filters, interpolation='Nearest', cmap='gray')
plt.colorbar()
plt.savefig('filters.png')

# # weight histogram
# plt.figure()
# plt.hist(testrbm.w[:testrbm.n_inputs].flatten(), 50)
# plt.savefig('weight_histo.png')

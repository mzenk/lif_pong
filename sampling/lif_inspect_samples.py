# script for analyzing sampled data
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
# sys.path.insert(0, '../')
from training.rbm import RBM, CRBM
from utils import tile_raster_images
from utils.data_mgmt import make_figure_folder, load_images, load_rbm, get_data_path

# Load rbm and data

show_label = False
# # MNIST
# img_shape = (28, 28)
# import gzip
# f = gzip.open('../shared_data/datasets/mnist.pkl.gz', 'rb')
# _, _, test_set = np.load(f)
# f.close()
# rbm = load_rbm('mnist_disc_rbm')
# nv = rbm.n_visible
# nl = rbm.n_labels
# sample_file = get_data_path('lif_dreaming') + \
#     'mnist_samples.npz'
# Pong
img_shape = (36, 48)
data_name = 'pong_var_start{}x{}'.format(*img_shape)
_, _, test_set = load_images(data_name)
rbm = load_rbm(data_name + '_crbm')
nv = rbm.n_visible
nl = rbm.n_labels
sample_file = get_data_path('lif_clamp_window') + \
    'pong_smooth8_win48_all_chunk002.npz'

# # tests
# sample_file = get_data_path('playground') + 'clamp_test_samples.npz'
# img_shape = (3, 4)
# nv = np.prod(img_shape)

n_pixels = np.prod(img_shape)


with np.load(sample_file) as d:
    # samples.shape: ([n_instances], n_samples, n_units)
    samples = d['samples'].astype(float)
    if len(samples.shape) == 2:
        samples = np.expand_dims(samples, 0)
    data_idx = d['data_idx']
    print('Loaded sample array with shape {}'.format(samples.shape))
    test_data = test_set[0][data_idx]
    test_targets = test_set[1][data_idx]

vis_samples = samples[..., :n_pixels]
hid_samples = samples[..., nv:]
if show_label:
    lab_samples = samples[..., nv - nl:nv]
    # Compute classification performance
    labels = np.argmax(lab_samples.sum(axis=1), axis=1)
    print('Correct predictions: {}'.format((labels == test_targets).mean()))

# # marginal visible probabilities can be calculated from hidden states
# nh = hid_samples.shape[-1]
# vis_probs = rbm.sample_v_given_h(hid_samples.reshape(-1, nh))[0][:, :n_pixels]
# sample_imgs = vis_probs.reshape(vis_probs.shape[0], *img_shape)
sample_imgs = vis_samples.reshape(-1, *img_shape)
if show_label:
    active_lab = np.argmax(lab_samples, axis=2).flatten()

# # plot images for quick inspection
# tiled_samples = tile_raster_images(sample_imgs[::50],
#                                    img_shape=img_shape,
#                                    tile_shape=(4, 5),
#                                    tile_spacing=(1, 1),
#                                    scale_rows_to_unit_interval=False,
#                                    output_pixel_vals=False)

# plt.figure()
# plt.imshow(tiled_samples, interpolation='Nearest', cmap='gray')
# plt.savefig(make_figure_folder() + 'samples.png', bbox_inches='tight')


# video
def update_fig(i):
    idx = i % sample_imgs.shape[0]
    if show_label:
        lab_text.set_text('Active label: {}'.format(active_lab[idx]))
    im.set_data(sample_imgs[idx])
    return im,  lab_text

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(np.zeros(img_shape), vmin=0, vmax=1., interpolation='Nearest',
               cmap='gray', animated=True)
lab_text = ax.text(0.95, 0.01, '', va='bottom', ha='right',
                   transform=ax.transAxes, color='green')

ani = animation.FuncAnimation(fig, update_fig, interval=20., blit=True,
                              repeat_delay=2000)

# ani.save('figures/sampling.mp4')
plt.show()

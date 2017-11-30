# script for analyzing sampled data
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
sys.path.insert(0, '../')
from training.rbm import RBM, CRBM
from utils import tile_raster_images
from utils.data_mgmt import make_figure_folder, load_images, load_rbm, get_data_path

# Load rbm and data

show_label = False
# testing
img_shape = (2, 2)
n_pixels = np.prod(img_shape)
n_labels = 0
# sample_file = get_data_path('lif_clamp_stp') + 'test.npz'
sample_file = '/wang/users/mzenk/cluster_home/experiment/simulations/TestSweep/0.001_1000.0/samples.npz'
# # MNIST
# img_shape = (28, 28)
# n_pixels = np.prod(img_shape)
# import gzip
# f = gzip.open('../shared_data/datasets/mnist.pkl.gz', 'rb')
# _, _, test_set = np.load(f)
# f.close()
# rbm = load_rbm('mnist_disc_rbm')
# nv = rbm.n_visible
# n_labels = rbm.n_labels
# sample_file = get_data_path('playground') + \
#     'test_bias_neurons.npz'
# # Pong
# img_shape = (36, 48)
# n_pixels = np.prod(img_shape)
# n_labels = img_shape[0] // 3
# data_name = 'pong_var_start{}x{}'.format(*img_shape)
# _, _, test_set = load_images(data_name)
# sample_file = get_data_path('lif_clamp_stp') + \
#     'pong_test_chunk000.npz'

with np.load(sample_file) as d:
    # samples.shape: ([n_instances], n_samples, n_units)
    samples = d['samples'].astype(float)
    if len(samples.shape) == 2:
        samples = np.expand_dims(samples, 0)
    if 'data_idx' in d.keys():
        data_idx = d['data_idx']
    n_samples = samples.shape[1]
    n_imgs = samples.shape[0]
    print('Loaded sample array with shape {}'.format(samples.shape))


vis_samples = samples[..., :n_pixels]
hid_samples = samples[..., n_pixels + n_labels:]
if show_label:
    test_targets = test_set[1][data_idx]
    if len(test_targets.shape) == 2:
        test_targets = np.argmax(test_targets, axis=1)
    lab_samples = samples[..., n_pixels:n_pixels + n_labels]
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
    time_text.set_text('{:4d}/{} (img {})'.format(i % n_samples, n_samples,
                                                  (i // n_samples) % n_imgs))
    im.set_data(sample_imgs[idx])
    return im, time_text, lab_text

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(np.zeros(img_shape), vmin=0, vmax=1., interpolation='Nearest',
               cmap='gray', animated=True)
lab_text = ax.text(0.95, 0.01, '', va='bottom', ha='right',
                   transform=ax.transAxes, color='green')
time_text = ax.text(0.05, 0.01, '', va='bottom', ha='left',
                    transform=ax.transAxes, color='white', fontsize=12)

ani = animation.FuncAnimation(fig, update_fig, interval=20., blit=True,
                              repeat_delay=2000)

# ani.save(make_figure_folder() + 'sampling.mp4')
plt.show()

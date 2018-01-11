# script for analyzing sampled data
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
from lif_pong.utils import tile_raster_images
from lif_pong.utils.data_mgmt import make_figure_folder, load_images, get_rbm_dict, get_data_path
plt.rcParams['animation.ffmpeg_path'] = u'/home/hd/hd_hd/hd_kq433/ffmpeg-3.4.1-64bit-static/ffmpeg'

n_imgs = 2
sample_file = None
if len(sys.argv) >= 2:
    sample_file = sys.argv[1]
if len(sys.argv) == 3:
    n_imgs = int(sys.argv[2])

show_label = False
# # testing
# img_shape = (2, 2)
# n_pixels = np.prod(img_shape)
# n_labels = 0
# # sample_file = get_data_path('lif_clamp_stp') + 'test.npz'
# sample_file = '/wang/users/mzenk/cluster_home/experiment/simulations/TestSweep/0.001_2000.0/samples.npz'
# # MNIST
# img_shape = (28, 28)
# n_pixels = np.prod(img_shape)
# import gzip
# f = gzip.open('../shared_data/datasets/mnist.pkl.gz', 'rb')
# _, _, test_set = np.load(f)
# f.close()
# sample_file = get_data_path('playground') + \
#     'test_bias_neurons.npz'
# Pong
img_shape = (36, 48)
n_pixels = np.prod(img_shape)
n_labels = img_shape[0] // 3
data_name = 'pong_var_start{}x{}'.format(*img_shape)
_, _, test_set = load_images(data_name)
if sample_file is None:
    print('no sample file give. load default.')
    sample_file = get_data_path('lif_clamp_stp') + \
        'pong_test_chunk000.npz'

with np.load(sample_file) as d:
    # samples.shape: ([n_instances], n_samples, n_units)
    samples = d['samples'].astype(float)[:n_imgs]
    if len(samples.shape) == 2:
        samples = np.expand_dims(samples, 0)
    if 'data_idx' in d.keys():
        data_idx = d['data_idx']
    n_samples = samples.shape[1]
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
# vis_samples = rbm.sample_v_given_h(hid_samples.reshape(-1, nh))[0][:, :n_pixels]

# # running average over samples?
# from scipy.ndimage import convolve1d
# kwidth = 20
# kernel = np.ones(kwidth)/kwidth
# avg_samples = convolve1d(vis_samples, kernel, axis=1)
avg_samples = vis_samples

frames = avg_samples.reshape(-1, *img_shape)
samples_per_frame = frames.shape[1] / (img_shape[1] + 1)
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
def update_fig(i_frame):
    i, frame = i_frame
    # idx = i % frames.shape[0]
    # if show_label:
    #     lab_text.set_text('Active label: {}'.format(active_lab[idx]))
    time_text.set_text('{:4d}/{} (img {})'.format(i % n_samples, n_samples,
                                                  (i // n_samples) % n_imgs))
    im.set_data(frame)
    return im, time_text, lab_text


fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(np.zeros(img_shape), vmin=0, vmax=1., interpolation='Nearest',
               cmap='gray', animated=True)
lab_text = ax.text(0.95, 0.01, '', va='bottom', ha='right',
                   transform=ax.transAxes, color='green')
time_text = ax.text(0.05, 0.01, '', va='bottom', ha='left',
                    transform=ax.transAxes, color='white', fontsize=12)

ani = animation.FuncAnimation(fig, update_fig, frames=zip(range(len(frames)), frames),
                              interval=10., blit=True, repeat=False)

ani.save(make_figure_folder() + 'test.mp4', writer='ffmpeg')
# plt.show()

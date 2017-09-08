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

# MNIST
img_shape = (28, 28)
import gzip
f = gzip.open('../shared_data/datasets/mnist.pkl.gz', 'rb')
_, _, test_set = np.load(f)
f.close()
rbm = load_rbm('mnist_disc_rbm')
has_label = type(rbm) is CRBM
nv = rbm.n_visible
nl = rbm.n_labels
sample_file = get_data_path('lif_sampling') + 'mnist_samples.npz'
# # Pong
# img_shape = (36, 48)
# data_name = 'pong_var_start{}x{}'.format(*img_shape)
# _, _, test_set = load_images(data_name)
# rbm = load_rbm(data_name + '_crbm')
# nv = rbm.n_visible
# nl = rbm.n_labels
# has_label = type(rbm) is CRBM
# sample_file = get_data_path('lif_sampling') + \
#     'pong_samples.npz'
# # tests
# img_shape = (2, 2)
# sample_file = get_data_path('lif_sampling') + 'toyrbm_samples.npz'
# nv = 4
# nl = 0
# has_label = False

n_pixels = np.prod(img_shape)


with np.load(sample_file) as d:
    samples = d['samples'][0].astype(float)
    # samples.shape: (n_instances, n_samples, n_z)

vis_samples = samples[:, :n_pixels]
print(vis_samples.shape)
print((vis_samples != 0).sum())
hid_samples = samples[:, nv:]
if has_label:
    lab_samples = samples[:, nv - nl:nv]
    active_lab = np.argmax(lab_samples, axis=1)

# # marginal visible probabilities can be calculated from hidden states
# vis_probs = rbm.sample_v_given_h(hid_samples)[0][:, :n_pixels]
# sample_imgs = vis_probs.reshape(vis_probs.shape[0], *img_shape)
sample_imgs = vis_samples.reshape(vis_samples.shape[0], *img_shape)

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

# test_img = np.ones(img_shape)
# test_img = test_img.flatten()
# # test_img = np.round(test_set[0][13])
# counter = 0
# for i, s in enumerate(vis_samples[1:]):
#     if np.any(s != test_img):
#         # if counter > 20:
#         #     print('More than 20 wrong...')
#         #     break
#         # plt.figure()
#         # plt.imshow(s.reshape(*img_shape) - test_img.reshape(*img_shape),
#         #            interpolation='Nearest', cmap='gray')
#         # plt.colorbar()
#         # plt.savefig('wrong' + str(i) + '.png')
#         # plt.close()
#         counter += 1
# print('{} of {} samples were clamped incorrectly'.format(
#     counter, len(vis_samples)))


# video
def update_fig(i):
    idx = i % sample_imgs.shape[0]
    if has_label:
        lab_text.set_text('Active label: {}'.format(active_lab[idx]))
    im.set_data(sample_imgs[idx])
    # im.set_data(hid_samples.reshape(hid_samples.shape[0], 20, 20)[idx])
    return im,  lab_text

fig = plt.figure()
ax = fig.add_subplot(111)
# if a homogeneous picture is used here it does not work -> wtf
im = ax.imshow(np.random.rand(*img_shape), interpolation='Nearest',
               cmap='gray', animated=True)
lab_text = ax.text(0.95, 0.01, '', va='bottom', ha='right',
                   transform=ax.transAxes, color='green')

ani = animation.FuncAnimation(fig, update_fig, interval=50., blit=True,
                              repeat_delay=2000)

# ani.save('figures/sampling.mp4')
plt.show()

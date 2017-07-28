# script for analyzing sampled data
import matplotlib as mpl
# mpl.use( "Agg" )
import numpy as np
import cPickle
import gzip
import sys
sys.path.insert(0, '../gibbs_rbm')
from rbm import RBM, CRBM
from util import tile_raster_images
# import sbs
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load rbm and data

# MNIST
img_shape = (28, 28)
f = gzip.open('../datasets/mnist.pkl.gz', 'rb')
_, _, test_set = np.load(f)
f.close()
with open('../gibbs_rbm/saved_rbms/mnist_disc_rbm.pkl', 'rb') as f:
    rbm = cPickle.load(f)
sample_file = 'mnist_samples.npy'
# # Pong
# img_shape = (36, 48)
# n_pxls = np.prod(img_shape)
# data_name = 'pong_var_start{}x{}'.format(*img_shape)
# with np.load('../datasets/' + data_name + '.npz') as d:
#     train_set, _, test_set = d[d.keys()[0]]
# with open('../gibbs_rbm/saved_rbms/' + data_name + '_crbm.pkl', 'rb') as f:
#     rbm = cPickle.load(f)
# sample_file = 'pong_clamped_samples.npy'

n_pixels = np.prod(img_shape)
clamped_label = False


samples = np.load(sample_file)
vis_samples = samples[:, :n_pixels]
hid_samples = samples[:, rbm.n_visible:]
if clamped_label:
    lab_samples = samples[:, rbm.n_inputs:rbm.n_visible]
    active_lab = np.argmax(lab_samples, axis=1)

# marginal visible probabilities can be calculated from hidden states
vis_probs = rbm.sample_v_given_h(hid_samples)[0][:, :n_pixels]
sample_imgs = vis_probs.reshape(vis_probs.shape[0], *img_shape)
# sample_imgs = vis_samples.reshape(vis_samples.shape[0], *img_shape)

# plot images for quick inspection
tiled_samples = tile_raster_images(vis_probs[::vis_samples.shape[0]//20],
                                   img_shape=img_shape,
                                   tile_shape=(4, 5),
                                   tile_spacing=(1, 1),
                                   scale_rows_to_unit_interval=False,
                                   output_pixel_vals=False)

plt.figure()
plt.imshow(tiled_samples, interpolation='Nearest', cmap='gray')
plt.savefig('Figures/samples.png', bbox_inches='tight')

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

# # video
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # if a homogeneous picture is used here it does not work -> wtf
# im = ax.imshow(np.random.rand(*img_shape), interpolation='Nearest',
#                cmap='gray', animated=True)
# lab_text = ax.text(0.95, 0.01, '', va='bottom', ha='right',
#                    transform=ax.transAxes, color='green')


# def update_fig(i):
#     idx = i % sample_imgs.shape[0]
#     if clamped_label:
#         lab_text.set_text('Active label: {}'.format(active_lab[idx]))
#     im.set_data(sample_imgs[idx])
#     # im.set_data(hid_samples.reshape(hid_samples.shape[0], 20, 20)[idx])
#     return im,  lab_text

# ani = animation.FuncAnimation(fig, update_fig, interval=50., blit=True,
#                               repeat_delay=2000)

# # ani.save('figures/sampling.mp4')
# plt.show()

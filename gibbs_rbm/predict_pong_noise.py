from __future__ import division
from __future__ import print_function
import numpy as np
import cPickle
from util import tile_raster_images, to_1_of_c, get_windowed_image_index
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def add_traj_noise(img_patch, sigma=3.):
    # compute com
    col = img_patch[:, -1]
    noised_com = np.average(np.arange(len(col)), weights=col) + \
        sigma * np.random.randn()
    fp_ind = noised_com - .5
    # cic assignment (cf. trajectory.py::add_to_image; here 1d, h=1 and
    # actually boundary conditions not necessary (?))
    fp_ind = max(0, fp_ind)
    fp_ind = min(len(col) - 1, fp_ind)
    w_lower = np.ceil(fp_ind) - fp_ind
    w_upper = 1 - w_lower
    noised_col = np.zeros_like(col)
    noised_col[int(np.ceil(fp_ind))] = w_upper
    noised_col[int(np.floor(fp_ind))] = w_lower
    img_patch[:, -1] = noised_col
    return img_patch.flatten()


# # add noise to whole image set -> tbd, use in uncover routine
# def add_traj_noise_set(img_set, interval, sigma=3.):
#     # compute com
#     col = img_patch[:, -1]
#     noised_com = np.average(np.arange(len(col)), weights=col) + \
#         sigma * np.random.randn()
#     fp_ind = noised_com - .5
#     # cic assignment (cf. trajectory.py::add_to_image; here 1d, h=1 and
#     # actually boundary conditions not necessary (?))
#     fp_ind = max(0, fp_ind)
#     fp_ind = min(len(col) - 1, fp_ind)
#     w_lower = np.ceil(fp_ind) - fp_ind
#     w_upper = 1 - w_lower
#     noised_col = np.zeros_like(col)
#     noised_col[int(np.ceil(fp_ind))] = w_upper
#     noised_col[int(np.floor(fp_ind))] = w_lower
#     img_patch[:, -1] = noised_col
#     return img_patch.flatten()


def add_background_noise(image, sigma=.05):
    # add Gaussian noise (other distributions possible) and normalize image
    noisy = image + np.random.randn(*image.shape) * sigma
    return np.clip(noisy, 0., 1.)

# pong pattern completion
# Load Pong data
img_shape = (36, 48)
n_pxls = np.prod(img_shape)
data_name = 'pong_var_start{}x{}'.format(*img_shape)
# data_name = 'label1_fixed'
discriminative = True
with np.load('../datasets/' + data_name + '.npz') as d:
    train_set, _, test_set = d[d.keys()[0]]

if discriminative:
    rbm_name = data_name + '_crbm.pkl'
else:
    rbm_name = data_name + '_rbm.pkl'

# rbm_name = 'pong_mf.pkl'
# Load rbm
with open('saved_rbms/' + rbm_name, 'rb') as f:
    rbm = cPickle.load(f)

# Produce visual example for a pattern completion
v_init = np.zeros(rbm.n_visible)
# np.random.randint(2, size=rbm.n_visible)
burnIn = int(20)
winsize = 48
image = test_set[0][42]
for fraction in np.arange(1, img_shape[1]):
    # first get the prediction
    clamped = get_windowed_image_index(img_shape, fraction,
                                       window_size=winsize)
    clamped_input = image[clamped]

    # add noise to rightmost clamped_column
    clamped_input = add_traj_noise(clamped_input.reshape(img_shape[0], -1),
                                   sigma=.05 * img_shape[0])
    image[clamped] = clamped_input
    # background noise is reset after each step
    # clamped_input = add_background_noise(clamped_input, sigma=3e-2)

    # RBM
    unclampedinp = np.setdiff1d(np.arange(rbm.n_inputs), clamped)
    samples = \
        rbm.sample_with_clamped_units(100 + burnIn, clamped, clamped_input,
                                      v_init=v_init, binary=True)[burnIn:]
    prediction = np.average(samples, axis=0)
    # # carry previous state over
    # burnIn = 0
    # unclampedvis = np.setdiff1d(np.arange(rbm.n_visible), clamped)
    # v_init[unclampedvis] = samples[-1]
    # v_init[clamped] = clamped_input

    trajectory = prediction[:-rbm.n_labels]
    labels = prediction[-rbm.n_labels:]

    inferred_img = np.zeros(np.prod(img_shape))
    # use rgb to make clamped part distinguishable from unclamped part
    inferred_img[clamped] = clamped_input
    inferred_img = np.tile(inferred_img, (3, 1)).T
    inferred_img[unclampedinp, 0] = trajectory
    # ---

    # # DBM
    # clamped_list = [None] * (1 + rbm.n_layers)
    # clamped_list[0] = clamped
    # clamped_val = [None] * (1 + rbm.n_layers)
    # clamped_val[0] = clamped_input
    # samples = rbm.draw_samples(100 + burnIn, clamped=clamped_list,
    #                            clamped_val=clamped_val)[10:]
    # inferred_img = np.average(samples, axis=0)
    # inferred_img = np.tile(inferred_img, (3, 1)).T
    # if clamped.size != n_pxls:
    #     inferred_img[np.setdiff1d(np.arange(rbm.n_visible), clamped), 1:] = 0
    # # ---

    inferred_img = inferred_img.reshape((img_shape[0], img_shape[1], 3))
    # plotting - though ugly, this is the best working implementation I found
    if fraction % 3 == 0:
        # fig = plt.figure()
        # width = .7
        # ax1 = fig.add_axes([.05, .2, width, width*3/4])
        # ax2 = fig.add_axes([width - .02, .2, .2, width*3/4])
        fig, ax1 = plt.subplots()
        ax1.imshow(inferred_img, interpolation='Nearest', cmap='gray',
                   origin='lower')
        ax1.tick_params(left='off', right='off', bottom='off', labelleft='off',
                        labelright='off', labelbottom='off')
        # ax2.barh(np.arange(inferred_img.shape[0]) - .5, inferred_img[:, -1, 0],
        #          height=np.ones(inferred_img.shape[0]), color='r')
        # ax2.set_ylim([-.5, inferred_img.shape[0] - .5])
        # ax2.xaxis.set_ticks([0., 0.5, 1.])
        # ax2.tick_params(left='off', right='off', labelleft='off', labelright='off')
        fig.savefig('figures/noise{}.png'.format(fraction),
                    bbox_inches='tight')
        plt.close(fig)

# # Test the "classification performance", i.e. how much of the picture does
# # the RBM need to predict the correct outcome
# test_set[0] = test_set[0][:10]
# test_set[1] = test_set[1][:10]
# imgs = test_set[0]
# labels = np.argmax(test_set[1], axis=1)
# targets = np.average(np.tile(np.arange(rbm.n_labels), (len(imgs), 1)),
#                      weights=test_set[1], axis=1)
# # fractions = np.arange(0, img_shape[1] + 1, 3)
# fractions = np.linspace(0., 1., 20)
# win_size = 48
# burnIn = 20
# n_sampl = 100
# v_init = np.random.randint(2, size=(imgs.shape[0], rbm.n_visible))

# correct_predictions = np.zeros_like(fractions, dtype=float)
# distances = np.zeros_like(fractions, dtype=float)
# dist_std = np.zeros_like(fractions, dtype=float)
# img_diff = np.zeros_like(fractions, dtype=float)
# img_diff_std = np.zeros_like(fractions, dtype=float)

# print('Window size: {}, #samples: {}'.format(win_size, n_sampl))
# for i, fraction in enumerate(fractions):
#     clamped_ind = \
#         get_windowed_image_index(img_shape, fraction,
#                                  fractional=True, window_size=win_size)

#     # due to memory requirements not all instances can be put into an array
#     n_chunks = int(np.ceil(8*(n_sampl+burnIn)*imgs.shape[0]*n_pxls / 3e9))
#     n_each, remainder = imgs.shape[0] // n_chunks, imgs.shape[0] % n_chunks
#     chunk_sizes = np.array([0] + [n_each] * n_chunks)
#     chunk_sizes[1:(remainder + 1)] += 1
#     chunk_ind = np.cumsum(chunk_sizes)
#     for j, chunk in enumerate(np.array_split(imgs, n_chunks)):
#         if v_init is None:
#             chunk_init = None
#         else:
#             chunk_init = v_init[chunk_ind[j]:chunk_ind[j+1]]
#         samples = \
#             rbm.sample_with_clamped_units(burnIn + n_sampl,
#                                           clamped_ind=clamped_ind,
#                                           clamped_val=chunk[:, clamped_ind],
#                                           v_init=chunk_init)
#         # minimal "burn-in"? omit first few samples
#         tmp_vis = np.mean(samples[burnIn:, :, :-rbm.n_labels], axis=0)
#         tmp_lab = np.mean(samples[burnIn:, :, -rbm.n_labels:], axis=0)
#         if j == 0:
#             vis_samples = tmp_vis
#             lab_samples = tmp_lab
#         else:
#             vis_samples = np.vstack((vis_samples, tmp_vis))
#             lab_samples = np.vstack((lab_samples, tmp_lab))

#     # if the gibbs chain should be continued
#     unclampedvis = np.setdiff1d(np.arange(rbm.n_visible), clamped_ind)
#     v_init[:, unclampedvis] = np.hstack((vis_samples[-1], lab_samples[-1]))

#     pred_labels = np.argmax(lab_samples, axis=1)
#     pred_pos = np.average(np.tile(np.arange(rbm.n_labels), (imgs.shape[0], 1)),
#                           weights=lab_samples, axis=1)
#     unclamped = np.setdiff1d(np.arange(rbm.n_inputs), clamped_ind)

#     correct_predictions[i] = np.mean(pred_labels == labels)
#     distances[i] = np.mean(np.abs(pred_pos - targets))
#     dist_std[i] = np.std(np.abs(pred_pos - targets))
#     if unclamped.size != 0:
#         # this is the L2 norm of the difference image normalized to one pixel
#         l2_diff = np.sqrt(np.sum((imgs[:, unclamped] - vis_samples)**2, axis=1))
#         img_diff[i] = np.mean(l2_diff / unclamped.size)
#         img_diff_std[i] = np.std(l2_diff / unclamped.size)

# # save data
# np.savez_compressed(
#     'figures/' + data_name[:4] + '_uncover{}w{}s'.format(win_size, n_sampl),
#     (correct_predictions, distances, dist_std, img_diff, img_diff_std))

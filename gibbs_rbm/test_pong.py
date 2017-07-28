from __future__ import division
from __future__ import print_function
import numpy as np
import cPickle
from util import tile_raster_images, to_1_of_c, get_windowed_image_index
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


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

# # Produce visual example for a pattern completion
# v_init = np.zeros(rbm.n_visible)
# # np.random.randint(2, size=rbm.n_visible)
# burnIn = int(1e1)
# image = test_set[0][141]
# for fraction in np.linspace(.1, .9, 9):
#     # first get the prediction
#     clamped = get_windowed_image_index(img_shape, fraction,
#                                        window_size=100, fractional=True)
#     clamped_input = image[clamped]

#     # RBM
#     unclampedinp = np.setdiff1d(np.arange(rbm.n_inputs), clamped)
#     samples = \
#         rbm.sample_with_clamped_units(100 + burnIn, clamped, clamped_input,
#                                       v_init=v_init, binary=True)[burnIn:]
#     prediction = np.average(samples, axis=0)
#     # # carry previous state over
#     # burnIn = 0
#     # unclampedvis = np.setdiff1d(np.arange(rbm.n_visible), clamped)
#     # v_init[unclampedvis] = samples[-1]
#     # v_init[clamped] = clamped_input

#     trajectory = prediction[:-rbm.n_labels]
#     labels = prediction[-rbm.n_labels:]

#     inferred_img = np.zeros(np.prod(img_shape))
#     # use rgb to make clamped part distinguishable from unclamped part
#     inferred_img[clamped] = clamped_input
#     inferred_img = np.tile(inferred_img, (3, 1)).T
#     inferred_img[unclampedinp, 0] = trajectory
#     # ---

#     # # DBM
#     # clamped_list = [None] * (1 + rbm.n_layers)
#     # clamped_list[0] = clamped
#     # clamped_val = [None] * (1 + rbm.n_layers)
#     # clamped_val[0] = clamped_input
#     # samples = rbm.draw_samples(100 + burnIn, clamped=clamped_list,
#     #                            clamped_val=clamped_val)[10:]
#     # inferred_img = np.average(samples, axis=0)
#     # inferred_img = np.tile(inferred_img, (3, 1)).T
#     # if clamped.size != n_pxls:
#     #     inferred_img[np.setdiff1d(np.arange(rbm.n_visible), clamped), 1:] = 0
#     # # ---

#     inferred_img = inferred_img.reshape((img_shape[0], img_shape[1], 3))
#     # plotting - though ugly, this is the best working implementation I found
#     fig = plt.figure()
#     width = .7
#     ax1 = fig.add_axes([.05, .2, width, width*3/4])
#     ax2 = fig.add_axes([width - .02, .2, .2, width*3/4])
#     ax1.imshow(inferred_img, interpolation='Nearest', cmap='gray',
#                origin='lower')
#     ax2.barh(np.arange(inferred_img.shape[0]) - .5, inferred_img[:, -1, 0],
#              height=np.ones(inferred_img.shape[0]), color='r')
#     ax2.set_ylim([-.5, inferred_img.shape[0] - .5])
#     ax2.xaxis.set_ticks([0., 0.5, 1.])
#     ax2.tick_params(left='off', right='off', labelleft='off', labelright='off')
#     fig.savefig('figures/windowed_traj{:.1f}.png'.format(fraction),
#                 bbox_inches='tight')

# Test the "classification performance", i.e. how much of the picture does
# the RBM need to predict the correct outcome
test_set[0] = test_set[0][:10]
test_set[1] = test_set[1][:10]
imgs = test_set[0]
labels = np.argmax(test_set[1], axis=1)
targets = np.average(np.tile(np.arange(rbm.n_labels), (len(imgs), 1)),
                     weights=test_set[1], axis=1)
# fractions = np.arange(0, img_shape[1] + 1, 3)
fractions = np.linspace(0., 1., 20)
win_size = 48
burnIn = 20
n_sampl = 100
v_init = np.random.randint(2, size=(imgs.shape[0], rbm.n_visible))

correct_predictions = np.zeros_like(fractions, dtype=float)
distances = np.zeros_like(fractions, dtype=float)
dist_std = np.zeros_like(fractions, dtype=float)
img_diff = np.zeros_like(fractions, dtype=float)
img_diff_std = np.zeros_like(fractions, dtype=float)

print('Window size: {}, #samples: {}'.format(win_size, n_sampl))
for i, fraction in enumerate(fractions):
    clamped_ind = \
        get_windowed_image_index(img_shape, fraction,
                                 fractional=True, window_size=win_size)

    # due to memory requirements not all instances can be put into an array
    n_chunks = int(np.ceil(8*(n_sampl+burnIn)*imgs.shape[0]*n_pxls / 3e9))
    n_each, remainder = imgs.shape[0] // n_chunks, imgs.shape[0] % n_chunks
    chunk_sizes = np.array([0] + [n_each] * n_chunks)
    chunk_sizes[1:(remainder + 1)] += 1
    chunk_ind = np.cumsum(chunk_sizes)
    for j, chunk in enumerate(np.array_split(imgs, n_chunks)):
        if v_init is None:
            chunk_init = None
        else:
            chunk_init = v_init[chunk_ind[j]:chunk_ind[j+1]]
        samples = \
            rbm.sample_with_clamped_units(burnIn + n_sampl,
                                          clamped_ind=clamped_ind,
                                          clamped_val=chunk[:, clamped_ind],
                                          v_init=chunk_init)
        # minimal "burn-in"? omit first few samples
        tmp_vis = np.mean(samples[burnIn:, :, :-rbm.n_labels], axis=0)
        tmp_lab = np.mean(samples[burnIn:, :, -rbm.n_labels:], axis=0)
        if j == 0:
            vis_samples = tmp_vis
            lab_samples = tmp_lab
        else:
            vis_samples = np.vstack((vis_samples, tmp_vis))
            lab_samples = np.vstack((lab_samples, tmp_lab))

    # if the gibbs chain should be continued
    unclampedvis = np.setdiff1d(np.arange(rbm.n_visible), clamped_ind)
    v_init[:, unclampedvis] = np.hstack((vis_samples[-1], lab_samples[-1]))

    pred_labels = np.argmax(lab_samples, axis=1)
    pred_pos = np.average(np.tile(np.arange(rbm.n_labels), (imgs.shape[0], 1)),
                          weights=lab_samples, axis=1)
    unclamped = np.setdiff1d(np.arange(rbm.n_inputs), clamped_ind)

    correct_predictions[i] = np.mean(pred_labels == labels)
    distances[i] = np.mean(np.abs(pred_pos - targets))
    dist_std[i] = np.std(np.abs(pred_pos - targets))
    if unclamped.size != 0:
        # this is the L2 norm of the difference image normalized to one pixel
        l2_diff = np.sqrt(np.sum((imgs[:, unclamped] - vis_samples)**2, axis=1))
        img_diff[i] = np.mean(l2_diff / unclamped.size)
        img_diff_std[i] = np.std(l2_diff / unclamped.size)

# save data
np.savez_compressed(
    'figures/' + data_name[:4] + '_uncover{}w{}s'.format(win_size, n_sampl),
    (correct_predictions, distances, dist_std, img_diff, img_diff_std))

# plotting...
if win_size < img_shape[1]:
    xlabel = 'Window position'
else:
    xlabel = 'Uncovered fraction'
plt.figure(figsize=(14, 7))
plt.subplot(121)
plt.errorbar(fractions, distances, fmt='ro', yerr=dist_std)
plt.ylabel('Distance to correct label')
# plt.ylim([0, 3])
plt.xlabel(xlabel)
plt.twinx()
plt.plot(fractions, correct_predictions, 'bo')
plt.ylabel('Correct predictions')
# plt.ylim([0, 1])
plt.gca().spines['right'].set_color('blue')
plt.gca().spines['left'].set_color('red')
plt.title('#samples: {}, window size: {}'.format(n_sampl, win_size))

plt.subplot(122)
plt.errorbar(fractions, img_diff, fmt='ro', yerr=img_diff_std)
plt.ylabel('L2 image dissimilarity')
plt.xlabel(xlabel)
plt.tight_layout()
plt.savefig('figures/pong_uncover{}w{}s.pdf'.format(win_size, n_sampl))

# # Dreaming
# samples = rbm.draw_samples(int(1e5), ast=True)
# # rand_ind = np.random.randint(0, samples.shape[0], size=20)
# tiled_samples = tile_raster_images(samples[::samples.shape[0]//20, :n_pxls],
#                                    img_shape=img_shape,
#                                    tile_shape=(5, 4),
#                                    tile_spacing=(1, 1),
#                                    scale_rows_to_unit_interval=True,
#                                    output_pixel_vals=False)

# plt.figure()
# plt.imshow(tiled_samples, interpolation='Nearest', cmap='gray')
# plt.savefig('figures/samples.pdf', bbox_inches='tight')

# # whole set performance
# my_set = test_set
# labels = np.argmax(my_set[1], axis=1)

# # For the wrong cases, how do the class probabilities look?
# class_prob = rbm.classify(my_set[0], class_prob=True)
# prediction = np.argmax(class_prob, axis=1)
# print('Correct predictions: {}'.format(np.average(prediction == labels)))
# wrong_ind = np.where(np.argmax(class_prob, axis=1) != my_set[1])[0]
# plt.figure()
# for i in range(9):
#     plt.subplot(3, 3, i + 1)
#     plt.bar(np.arange(rbm.n_labels), class_prob[wrong_ind][i], width=1)
# plt.tight_layout()
# plt.savefig('class_prob.png')

# # How much off are the labels?
# # also check the distance to second best label guess
# sorted_ind = np.argsort(class_prob[wrong_ind], axis=1)
# difference1 = my_set[1][wrong_ind] - sorted_ind[:, -1]
# difference2 = my_set[1][wrong_ind] - sorted_ind[:, -2]
# plt.figure()
# plt.hist(difference1, align='left', color='g', label='1st guess',
#          bins=np.arange(-rbm.n_labels + 1, rbm.n_labels - 1))
# plt.hist(difference2, align='left', color='b', label='2nd guess', alpha=.7,
#          bins=np.arange(-rbm.n_labels + 1, rbm.n_labels - 1))
# plt.xlim([-5, 5])
# plt.legend()
# plt.xlabel('Correct label - predicted label')
# plt.savefig('wrong_difference.png')

# # ...and the difference in probability between first and second guess
# sorted_probs = np.sort(class_prob[wrong_ind], axis=1)
# prob_diffs = sorted_probs[:, -1] - sorted_probs[:, -2]
# plt.figure()
# plt.hist(prob_diffs, align='left', bins=10)
# plt.xlabel('Probability difference between 1st and 2nd guess')
# plt.savefig('prob_differences.png')

# # inspect some wrong predictions
# wrong_ind = np.where(prediction != my_set[1])[0]
# wrong_imgs = my_set[0][wrong_ind]
# wrong_labels = my_set[1][wrong_ind]
# rep_labels = np.repeat(2*(to_1_of_c(my_set[1][wrong_ind], rbm.n_labels) -
#                           to_1_of_c(prediction[wrong_ind], rbm.n_labels)),
#                        3, axis=1)

# # visualize labels
# picz = np.concatenate((wrong_imgs.reshape((wrong_ind.size,) + img_shape),
#                        rep_labels.reshape(wrong_ind.size, img_shape[0], 1)),
#                       axis=2).reshape(wrong_ind.size, img_shape[0],
#                                       img_shape[1] + 1)
# wrong_examples = tile_raster_images(picz[:9],
#                                     img_shape=(img_shape[0], img_shape[1] + 1),
#                                     tile_shape=(3, 3),
#                                     tile_spacing=(1, 1),
#                                     scale_rows_to_unit_interval=False,
#                                     output_pixel_vals=False)

# plt.figure()
# plt.imshow(wrong_examples, interpolation='Nearest', cmap='gray')
# plt.colorbar()
# plt.title('Black: predicted; white: correct')
# plt.savefig('wrong_examples.png')

# # Are there any labels that are harder to predict?
# # -> if more detail needed: confusion matrix
# plt.figure()
# histo = np.histogram(wrong_labels, bins=rbm.n_labels)[0] /\
#     np.histogram(my_set[1], bins=rbm.n_labels)[0]
# plt.bar(np.arange(rbm.n_labels), height=histo)
# plt.xlabel('Class label')
# plt.ylabel('Wrong predictions %')
# plt.savefig('wrong_histo.png')

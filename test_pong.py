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
data_name = 'pong_var_start{}x{}'.format(*img_shape)
# data_name = 'label1_fixed'
discriminative = True
with np.load('datasets/' + data_name + '.npz') as d:
    train_set, _, test_set = d[d.keys()[0]]

if discriminative:
    rbm_name = data_name + '_crbm.pkl'
else:
    rbm_name = data_name + '_rbm.pkl'

rbm_name = 'pong_cdbm.pkl'
# Load rbm
with open('saved_rbms/' + rbm_name, 'rb') as f:
    testrbm = cPickle.load(f)

# Produce visual example for a pattern completion
my_test = test_set[0][141]
for fraction in np.linspace(.1, .9, 9):
    # first get the prediction
    clamped_ind = get_windowed_image_index(img_shape, fraction,
                                           fractional=True, window_size=100)
    clamped_input = my_test[clamped_ind]
    # RBM - the interface is a bit different for DBM; if there's time,
    # make them equal
    # if discriminative:
    #     end_visible = -testrbm.n_labels
    #     index_range = testrbm.n_inputs
    # else:
    #     end_visible = testrbm.n_visible
    #     index_range = testrbm.n_visible
    # samples = \
    #     testrbm.sample_with_clamped_units(100, clamped_ind,
    #                                       clamped_input)[10:, :end_visible]

    # inferred_img = np.zeros(np.prod(img_shape))
    # # use rgb to make the clamped part distinguishable from the unclamped part
    # inferred_img[clamped_ind] = clamped_input
    # inferred_img = np.tile(inferred_img, (3, 1)).T
    # if not clamped_ind.size == np.prod(img_shape):
    #     inferred_img[np.setdiff1d(np.arange(index_range), clamped_ind), 0] = \
    #         np.average(samples, axis=0)

    # DBM
    clamped = [None] * (1 + testrbm.n_layers)
    clamped[0] = clamped_ind
    clamped_val = [None] * (1 + testrbm.n_layers)
    clamped_val[0] = clamped_input
    samples = testrbm.draw_samples(100, clamped=clamped,
                                   clamped_val=clamped_val)[10:]
    inferred_img = np.average(samples, axis=0)
    inferred_img = np.tile(inferred_img, (3, 1)).T
    if not clamped_ind.size == np.prod(img_shape):
        inferred_img[np.setdiff1d(np.arange(testrbm.n_visible),
                                  clamped_ind), 1:] = 0

    inferred_img = inferred_img.reshape((img_shape[0], img_shape[1], 3))
    # plotting - though ugly, this is the best working implementation I found
    fig = plt.figure()
    width = .7
    ax1 = fig.add_axes([.05, .2, width, width*3/4])
    ax2 = fig.add_axes([width - .02, .2, .2, width*3/4])
    ax1.imshow(inferred_img, interpolation='Nearest', cmap='gray',
               origin='lower')
    ax2.barh(np.arange(inferred_img.shape[0]) - .5, inferred_img[:, -1, 0],
             height=np.ones(inferred_img.shape[0]), color='r')
    ax2.set_ylim([-.5, inferred_img.shape[0] - .5])
    ax2.xaxis.set_ticks([0., 0.5, 1.])
    ax2.tick_params(left='off', right='off', labelleft='off', labelright='off')
    fig.savefig('figures/windowed_traj{:.1f}.png'.format(fraction),
                bbox_inches='tight')

# # Test the "classification performance", i.e. how much of the picture does
# # the RBM need to predict the correct outcome
# if discriminative:
#     imgs = test_set[0]
#     labels = test_set[1]
#     # fractions = np.arange(0, img_shape[1] + 1, 3)
#     fractions = np.linspace(0., 1., 20)
#     win_size = int(sys.argv[1])
#     n_sampl = int(sys.argv[2])
#     correct_predictions = np.zeros_like(fractions, dtype=float)
#     pred_err = np.zeros_like(fractions, dtype=float)
#     distances = np.zeros_like(fractions, dtype=float)
#     dist_err = np.zeros_like(fractions, dtype=float)

#     print('Window size: {}, #samples: {}'.format(win_size, n_sampl))
#     for i, fraction in enumerate(fractions):
#         clamped_ind = \
#             get_windowed_image_index(img_shape, fraction,
#                                      fractional=True, window_size=win_size)
#         samples = \
#             testrbm.sample_with_clamped_units(n_sampl, clamped_ind=clamped_ind,
#                                               clamped_val=imgs[:, clamped_ind])
#         # minimal "burn-in"? omit first few samples
#         prediction = np.argmax(np.sum(samples[5:, :, -testrbm.n_labels:],
#                                       axis=0), axis=1)

#         correct_predictions[i] = np.mean(prediction == labels)
#         pred_err[i] = np.std(prediction == labels)
#         distances[i] = np.mean(np.abs(prediction - labels))
#         dist_err[i] = np.std(np.abs(prediction - labels))

#     plt.figure()
#     plt.errorbar(fractions, distances, fmt='ro', yerr=dist_err)
#     plt.ylabel('Distance to correct label')
#     # plt.ylim([0, 3])
#     plt.xlabel('Uncovered fraction')
#     plt.twinx()
#     plt.plot(fractions, correct_predictions, 'bo')
#     plt.ylabel('Correct predictions')
#     # plt.ylim([0, 1])
#     plt.gca().spines['right'].set_color('blue')
#     plt.gca().spines['left'].set_color('red')
#     plt.title('#samples: {}, window size: {}'.format(n_sampl, win_size))
    # plt.savefig('uncovering{}window{}samples.png'.format(win_size, n_sampl),
    #             bbox_inches = 'tight')

# # Dreaming
# samples = testrbm.draw_samples(int(1e4), ast=False)
# # rand_ind = np.random.randint(0, samples.shape[0], size=20)
# tiled_samples = tile_raster_images(samples[::samples.shape[0]//20, :index_range],
#                                    img_shape=img_shape,
#                                    tile_shape=(5, 4),
#                                    tile_spacing=(1, 1),
#                                    scale_rows_to_unit_interval=True,
#                                    output_pixel_vals=False)

# plt.figure()
# plt.imshow(tiled_samples, interpolation='Nearest', cmap='gray')
# plt.savefig('samples.png', bbox_inches = 'tight')

# # whole set performance
# my_set = test_set

# # For the wrong cases, how do the class probabilities look?
# class_prob = testrbm.classify(my_set[0], class_prob=True)
# prediction = np.argmax(class_prob, axis=1)
# print('Correct predictions: '
#       '{}'.format(np.average(prediction == my_set[1])))
# wrong_ind = np.where(np.argmax(class_prob, axis=1) != my_set[1])[0]
# plt.figure()
# for i in range(9):
#     plt.subplot(3, 3, i + 1)
#     plt.bar(np.arange(testrbm.n_labels), class_prob[wrong_ind][i], width=1)
# plt.tight_layout()
# plt.savefig('class_prob.png')

# # How much off are the labels?
# # also check the distance to second best label guess
# sorted_ind = np.argsort(class_prob[wrong_ind], axis=1)
# difference1 = my_set[1][wrong_ind] - sorted_ind[:, -1]
# difference2 = my_set[1][wrong_ind] - sorted_ind[:, -2]
# plt.figure()
# plt.hist(difference1, align='left', color='g', label='1st guess',
#          bins=np.arange(-testrbm.n_labels + 1, testrbm.n_labels - 1))
# plt.hist(difference2, align='left', color='b', label='2nd guess', alpha=.7,
#          bins=np.arange(-testrbm.n_labels + 1, testrbm.n_labels - 1))
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
# rep_labels = np.repeat(2*(to_1_of_c(my_set[1][wrong_ind], testrbm.n_labels) -
#                           to_1_of_c(prediction[wrong_ind], testrbm.n_labels)),
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
# histo = np.histogram(wrong_labels, bins=testrbm.n_labels)[0] /\
#     np.histogram(my_set[1], bins=testrbm.n_labels)[0]
# plt.bar(np.arange(testrbm.n_labels), height=histo)
# plt.xlabel('Class label')
# plt.ylabel('Wrong predictions %')
# plt.savefig('wrong_histo.png')

from __future__ import division
from __future__ import print_function
import numpy as np
import cPickle
from util import tile_raster_images, to_1_of_c
from bm import Rbm, ClassRbm
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


# since v_x > 0 in Pong, we can simply uncover the images from the left
def get_windowed_image_index(img_shape, end_index,
                             window_size=-1, fractional=False):
    mask = np.zeros(img_shape)
    if fractional:
        end_index = int(end_index * img_shape[1])
    if window_size < 0:
        window_size = end_index
    start_index = max(0, end_index - window_size)
    mask[:, start_index:end_index] = 1
    uncovered_ind = np.nonzero(mask.flatten())[0]
    return uncovered_ind


# pong pattern completion
# Load Pong data
img_shape = (36, 48)
data_name = 'pong_var_start{}x{}'.format(*img_shape)
discriminative = True
with np.load('datasets/' + data_name + '.npz') as d:
    train_set, _, test_set = d[d.keys()[0]]

if discriminative:
    rbm_name = data_name + '_crbm.pkl'
else:
    rbm_name = data_name + '_rbm.pkl'

# Load rbm
with open('saved_rbms/' + rbm_name, 'rb') as f:
    testrbm = cPickle.load(f)


if discriminative:
    end_visible = -testrbm.n_labels
    index_range = testrbm.n_inputs
else:
    end_visible = testrbm.n_visible
    index_range = testrbm.n_visible

# train set performance
prediction = testrbm.classify(train_set[0])
print('Correct predictions:'
      '{}'.format(np.average(prediction == train_set[1])))

# How much off are the labels?
dist = prediction - train_set[1]
plt.hist(dist[dist != 0], bins=np.arange(-testrbm.n_labels + 1,
                                         testrbm.n_labels - 1), align='left')
plt.xlim([-5, 5])
plt.xlabel('Correct label - predicted label')
plt.savefig('wrong_difference.png')

# inspect some wrong predictions
wrong_ind = np.where(prediction != train_set[1])[0]
wrong_imgs = train_set[0][wrong_ind]
wrong_labels = train_set[1][wrong_ind]
rep_labels = np.repeat(2*(to_1_of_c(train_set[1][wrong_ind], testrbm.n_labels) -
                          to_1_of_c(prediction[wrong_ind], testrbm.n_labels)),
                       3, axis=1)

# visualize labels
picz = np.concatenate((wrong_imgs.reshape((wrong_ind.size,) + img_shape),
                       rep_labels.reshape(wrong_ind.size, img_shape[0], 1)),
                      axis=2).reshape(wrong_ind.size, img_shape[0],
                                      img_shape[1] + 1)
wrong_examples = tile_raster_images(picz[:9],
                                    img_shape=(img_shape[0], img_shape[1] + 1),
                                    tile_shape=(3, 3),
                                    tile_spacing=(1, 1),
                                    scale_rows_to_unit_interval=False,
                                    output_pixel_vals=False)

plt.figure()
plt.imshow(wrong_examples, interpolation='Nearest', cmap='gray', origin='lower')
plt.colorbar()
plt.title('Black: predicted; white: correct')
plt.savefig('wrong_examples.png')

# Are there any labels that are harder to predict?
# -> if more detail needed: confusion matrix
plt.figure()
histo = np.histogram(wrong_labels, bins=testrbm.n_labels)[0] /\
    np.histogram(train_set[1], bins=testrbm.n_labels)[0]
plt.bar(np.arange(testrbm.n_labels), height=histo)
plt.xlabel('Class label')
plt.ylabel('Wrong predictions %')
plt.savefig('wrong_histo.png')

# # Produce visual example for a pattern completion
# my_test = test_set[0][123]
# for fraction in np.linspace(.1, .9, 9):
#     # first get the prediction
#     clamped_ind = get_windowed_image_index(img_shape, fraction,
#                                            fractional=True, window_size=4)
#     clamped_input = my_test[clamped_ind]
#     samples = \
#         testrbm.sample_with_clamped_units(100, clamped_ind,
#                                           clamped_input)[10:, :end_visible]

#     inferred_img = np.zeros(np.prod(img_shape))
#     # use rgb to make the clamped part distinguishable from the unclamped part
#     inferred_img[clamped_ind] = clamped_input
#     inferred_img = np.tile(inferred_img, (3, 1)).T
#     if not clamped_ind.size == np.prod(img_shape):
#         inferred_img[np.setdiff1d(np.arange(index_range), clamped_ind), 1] = \
#             np.average(samples, axis=0)
#     inferred_img = inferred_img.reshape((img_shape[0], img_shape[1], 3))

#     # plotting - though ugly, this is the best working implementation I found
#     fig = plt.figure()
#     width = .7
#     ax1 = fig.add_axes([.05, .2, width, width*3/4])
#     ax2 = fig.add_axes([width - .02, .2, .2, width*3/4])
#     ax1.imshow(inferred_img, interpolation='Nearest', cmap='gray',
#                origin='lower')
#     ax2.barh(np.arange(inferred_img.shape[0]) - .5, inferred_img[:, -1, 1],
#              height=np.ones(inferred_img.shape[0]), color='g')
#     ax2.set_ylim([-.5, inferred_img.shape[0] - .5])
#     ax2.xaxis.set_ticks([0., 0.5, 1.])
#     ax2.tick_params(left='off', right='off', labelleft='off', labelright='off')
#     fig.savefig('windowed_traj{:.1f}.png'.format(fraction))

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
#     plt.savefig('uncovering{}window{}samples.png'.format(win_size, n_sampl))

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
# plt.savefig('samples.png')

# # filters
# rand_ind = np.random.randint(testrbm.n_hidden, size=25)
# tiled_filters = tile_raster_images(X=testrbm.w.T[rand_ind, :testrbm.n_inputs],
#                                    img_shape=img_shape,
#                                    tile_shape=(5, 5),
#                                    tile_spacing=(1, 1),
#                                    scale_rows_to_unit_interval=False,
#                                    output_pixel_vals=False)
# plt.figure()
# plt.imshow(tiled_filters, interpolation='Nearest', cmap='gray')
# plt.colorbar()
# plt.savefig('filters.png')

# # weight histogram
# plt.figure()
# plt.hist(testrbm.w[:testrbm.n_inputs].flatten(), 50)
# plt.savefig('weight_histo.png')

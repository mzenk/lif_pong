# animation for pong
from __future__ import division
import numpy as np
import cPickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import time
from util import get_windowed_image_index


img_shape = (36, 48)
# Load Pong data
data_name = 'pong_var_start{}x{}'.format(*img_shape)
with np.load('datasets/' + data_name + '.npz') as d:
    train_set, _, test_set = d[d.keys()[0]]
# Load rbm
rbm_name = data_name + '_crbm.pkl'
rbm_name = 'pong_cdbm.pkl'
with open('saved_rbms/' + rbm_name, 'rb') as f:
    testrbm = cPickle.load(f)

# first for one example
np.random.seed(125575)
example_id = np.random.randint(test_set[0].shape[0])
example = test_set[0][example_id]
correct_label = test_set[1][example_id]
n_labels = test_set[1].max() + 1

# uncover the image pixel-by-pixel => t = n_pxls_x;
# the paddle speed is limited to n_pxls_y / n_pxls_x / 2
# fractions = np.arange(0, 1., 1./img_shape[1])
# if an index range is given
fractions = np.arange(img_shape[1] + 1)

frames = []
paddle_position = img_shape[0] / 2
paddle_length = img_shape[0] / n_labels

max_speed = img_shape[1] / img_shape[0] / 2

prediction_time = []
fig = plt.figure()
for frac in fractions:
    # first get the prediction
    clamped_ind = get_windowed_image_index(img_shape, frac, window_size=100)
    clamped_input = example[clamped_ind]

    # start = time.time()
    # prediction = np.average(testrbm.sample_with_clamped_units(
    #     100, clamped_ind, clamped_input)[10:], axis=0)
    # trajectory = prediction[:-testrbm.n_labels]
    # labels = prediction[-testrbm.n_labels:]
    # # track time prediction takes
    # prediction_time.append(time.time() - start)

    # inferred_img = np.zeros(np.prod(img_shape))
    # # use rgb to make the clamped part distinguishable from the unclamped part
    # inferred_img[clamped_ind] = clamped_input
    # inferred_img = np.tile(inferred_img, (3, 1)).T
    # inferred_img[np.setdiff1d(np.arange(testrbm.n_inputs),
    #                           clamped_ind), 0] = trajectory

    # DBM
    clamped = [None] * (1 + testrbm.n_layers)
    clamped[0] = clamped_ind
    clamped_val = [None] * (1 + testrbm.n_layers)
    clamped_val[0] = clamped_input
    samples = testrbm.draw_samples(100, clamped=clamped,
                                   clamped_val=clamped_val, layer_ind='all')
    inferred_img = np.average(samples[10:, :testrbm.n_visible], axis=0)
    labels = np.average(samples[10:, -n_labels:], axis=0)
    inferred_img = np.tile(inferred_img, (3, 1)).T
    if not clamped_ind.size == np.prod(img_shape):
        inferred_img[np.setdiff1d(np.arange(testrbm.n_visible),
                                  clamped_ind), 1:] = 0

    inferred_img = inferred_img.reshape((img_shape[0], img_shape[1], 3))

    # timestep=1; paddle center should be aligned with label index, hence +.5
    target = np.average(np.arange(n_labels), weights=labels)
    paddle_position += max_speed * \
        min(2*(target - paddle_position / paddle_length + .5), 1)

    # print paddle_position/n_labels, np.argmax(labels)
    paddle_pixels = np.zeros((img_shape[0], 3))
    if paddle_position <= paddle_length/2:
        paddle_pixels[:paddle_length, 1] = 1
        paddle_position = 0
    elif paddle_position + paddle_length/2 >= img_shape[0] - 1:
        paddle_pixels[-paddle_length:, 1] = 1
        paddle_position = img_shape[0] - 1
    else:
        paddle_pixels[int(paddle_position) - paddle_length//2:
                      int(paddle_position) + np.round(paddle_length/2), 1] = 1

    pixels = np.concatenate((inferred_img, np.expand_dims(paddle_pixels, 1)),
                            axis=1)
    # frames.append((plt.imshow(pixels, interpolation='Nearest', cmap='gray'),))
    width = .7
    ax1 = fig.add_axes([.05, .2, width, width*3/4])
    ax2 = fig.add_axes([width - .02, .2, .2, width*3/4])
    ax2.set_ylim([-.5, pixels.shape[0] - .5])
    ax2.xaxis.set_ticks([0., 0.5, 1.])
    ax2.tick_params(left='off', right='off', labelleft='off', labelright='off')
    # barh does not work because apparently BarContainer has no 'set_visible'
    f1 = ax1.imshow(pixels, interpolation='Nearest', cmap='gray', origin='lower')
    f2 = ax2.plot(pixels[:, -2, 0], np.arange(pixels.shape[0]) - .5, 'ro')[0]
    frames.append((f1, f2))

# print('Max. prediction time: {}s'.format(max(prediction_time)))

# Set up formatting for the movie files --- whatever this is
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

traj_anim = animation.ArtistAnimation(fig, frames, interval=200,
                                      repeat_delay=3000, blit=True)
traj_anim.save('animation_' + data_name + '.mp4')
# plt.show()
# plt.close()

# animation for pong
from __future__ import division
import numpy as np
import cPickle
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import time
from util import get_windowed_image_index


def get_frames(rbm, image, winsize):
    global img_shape
    # uncover the image pixel-by-pixel => t = n_pxls_x;
    # the paddle speed is limited to n_pxls_y / n_pxls_x / 2
    # fractions = np.arange(0, 1., 1./img_shape[1])
    # if an index range is given
    fractions = np.arange(img_shape[1] + 1)

    frames = []
    paddle_pos = img_shape[0] / 2
    paddle_length = img_shape[0] / rbm.n_labels

    max_speed = img_shape[1] / img_shape[0] / 2

    v_init = np.random.randint(2, size=rbm.n_visible).astype(float)
    burnIn = 20
    n_samples = 100
    fig = plt.figure()
    for frac in fractions:
        # first get the prediction
        clamped = get_windowed_image_index(img_shape, frac,
                                           window_size=winsize)
        clamped_input = image[clamped]

        # RBM
        unclampedinp = np.setdiff1d(np.arange(rbm.n_inputs), clamped)
        samples = \
            rbm.sample_with_clamped_units(n_samples + burnIn, clamped,
                                          clamped_input, v_init=v_init,
                                          binary=True)[burnIn:]
        prediction = np.average(samples, axis=0)

        # carry previous state over
        unclampedvis = np.setdiff1d(np.arange(rbm.n_visible), clamped)
        v_init[unclampedvis] = samples[-1]

        trajectory = prediction[:-rbm.n_labels]
        labels = prediction[-rbm.n_labels:]

        inferred_img = np.zeros(np.prod(img_shape))
        # use rgb to make clamped part distinguishable from unclamped part
        inferred_img[clamped] = clamped_input
        inferred_img = np.tile(inferred_img, (3, 1)).T
        inferred_img[unclampedinp, 0] = trajectory
        # ---

        # # DBM
        # clamped = [None] * (1 + rbm.n_layers)
        # clamped[0] = clamped
        # clamped_val = [None] * (1 + rbm.n_layers)
        # clamped_val[0] = clamped_input
        # samples = rbm.draw_samples(n_samples + burnIn, clamped=clamped,
        #                            clamped_val=clamped_val, layer_ind='all')
        # inferred_img = np.average(samples[burnIn:, :rbm.n_visible], axis=0)
        # labels = np.average(samples[burnIn:, -rbm.n_labels:], axis=0)
        # inferred_img = np.tile(inferred_img, (3, 1)).T
        # if not clamped.size == np.prod(img_shape):
        #     inferred_img[np.setdiff1d(np.arange(rbm.n_visible),
        #                               clamped), 1:] = 0
        # # ---

        inferred_img = inferred_img.reshape((img_shape[0], img_shape[1], 3))

        # timestep=1; paddle center should be aligned with label index => +.5
        target = np.average(np.arange(rbm.n_labels), weights=labels)
        paddle_pos += max_speed * \
            min(2*(target - paddle_pos / paddle_length + .5), 1)

        # update paddle
        paddle_pxls = np.zeros((img_shape[0], 3))
        if paddle_pos <= paddle_length/2:
            paddle_pxls[:paddle_length, 1] = 1
            paddle_pos = 0
        elif paddle_pos + paddle_length/2 >= img_shape[0] - 1:
            paddle_pxls[-paddle_length:, 1] = 1
            paddle_pos = img_shape[0] - 1
        else:
            paddle_pxls[int(paddle_pos) - paddle_length//2:
                        int(paddle_pos) + np.round(paddle_length/2), 1] = 1

        pixels = np.hstack((inferred_img, np.expand_dims(paddle_pxls, 1)))

        # visualize labels as well
        labelsrep = np.repeat(labels, img_shape[0] // rbm.n_labels)
        labelsrgb = np.tile(np.expand_dims(labelsrep, 1), (1, 3))
        labelsrgb[:, 1:] = 0
        pixels = np.hstack((pixels, np.expand_dims(labelsrgb, 1)))

        # plotting
        width = .7
        ax1 = fig.add_axes([.05, .2, width, width*3/4])
        ax2 = fig.add_axes([width - .02, .2, .2, width*3/4])
        ax2.set_ylim([-.5, pixels.shape[0] - .5])
        ax2.xaxis.set_ticks([0., 0.5, 1.])
        ax2.tick_params(left='off', right='off',
                        labelleft='off', labelright='off')
        # barh doesnt work because apparently BarContainer has no 'set_visible'
        f1 = ax1.imshow(pixels, interpolation='Nearest', cmap='gray',
                        origin='lower')
        f2 = ax2.plot(inferred_img[:, -1, 0], np.arange(img_shape[0]) - .5,
                      'ro')[0]
        frames.append((f1, f2))

    # print('Max. prediction time: {}s'.format(max(prediction_time)))
    return fig, frames


img_shape = (36, 48)
# Load Pong data
data_name = 'gauss_var_start{}x{}'.format(*img_shape)
with np.load('datasets/' + data_name + '.npz') as d:
    train_set, _, test_set = d[d.keys()[0]]
# Load rbm
rbm_name = data_name + '_crbm.pkl'
# rbm_name = 'pong_cdbm.pkl'
with open('saved_rbms/' + rbm_name, 'rb') as f:
    rbm = cPickle.load(f)

# pick random examples and infer trajectories
np.random.seed(125575)
example_id = np.random.choice(test_set[0].shape[0], size=1, replace=False)
for i, example in enumerate(test_set[0][example_id]):
    fig, frames = get_frames(rbm, example, 100)

    # Set up formatting for the movie files --- whatever this is
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, bitrate=1800)

    traj_anim = animation.ArtistAnimation(fig, frames, interval=200,
                                          repeat_delay=3000, blit=True)
    traj_anim.save('figures/animation_' + data_name + str(i) + '.mp4')
    # plt.show()
    # plt.close()

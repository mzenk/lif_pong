# animation for data from gibbs simulation -> adapted from animate_lif.py
from __future__ import division
import numpy as np
import cPickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from rbm import RBM, CRBM
from util import get_data_path

sampling_interval = 100.


class Pong_updater(object):
    def __init__(self, image, points, img_shape, rbm, winsize=None,
                 clamp_duration=100., paddle_len=0, max_step=0.):
        self.image = image
        self.points = points
        self.img_shape = img_shape
        self.rbm = rbm
        self.winsize = winsize
        self.clamp_duration = clamp_duration
        self.clamp_position = 0.
        self.paddle_pos = img_shape[0]/2
        if paddle_len == 0:
            self.paddle_len = img_shape[0] / rbm.n_labels
        if max_step == 0:
            # maximum distance traveled in one update step. For now: double the
            # horizontal ball speed
            v_horiz_ball = 1. / self.clamp_duration        # 1 pxl each 100ms
            self.max_step = 2*v_horiz_ball * sampling_interval  # 10ms sampling

    def update_paddle(self, prediction):
        # timestep=1; paddle center should be aligned with label index => +.5
        if np.all(prediction == 0):
            prediction = np.ones_like(prediction)
        target = np.average(np.arange(rbm.n_labels), weights=prediction)
        self.paddle_pos += self.max_step * \
            min(2*(target - self.paddle_pos / self.paddle_len + .5), 1)

        # update paddle
        paddle_pxls = np.zeros((img_shape[0], 3))
        if self.paddle_pos <= self.paddle_len/2:
            paddle_pxls[:self.paddle_len, 1] = 1
            self.paddle_pos = 0
        elif self.paddle_pos + self.paddle_len/2 >= img_shape[0] - 1:
            paddle_pxls[-self.paddle_len:, 1] = 1
            self.paddle_pos = img_shape[0] - 1
        else:
            paddle_pxls[
                int(self.paddle_pos) - self.paddle_len//2:
                int(self.paddle_pos + np.round(self.paddle_len/2)), 1] = 1
        return paddle_pxls

    def __call__(self, vis_state):
            pixels = vis_state[:np.prod(self.img_shape)].astype(float)
            pixels = pixels.reshape(img_shape)
            self.points.set_data(pixels[:, -1],
                                 np.arange(self.img_shape[0]) - .5)

            # to rgb
            pixels = np.tile(np.expand_dims(pixels, 2), (1, 1, 3))
            labels = vis_state[np.prod(self.img_shape):]

            if self.winsize is not None:
                # overlay clamping window (update interval is 10ms)
                self.clamp_position += sampling_interval / self.clamp_duration
                self.clamp_position = self.clamp_position % \
                    (self.img_shape[1] + 1)
                clamped_window = np.zeros(img_shape + (3,))
                end = int(self.clamp_position)
                start = max(0, (end - self.winsize))
                clamped_window[:, start:end, 2] = .2
                pixels = np.minimum(np.ones_like(pixels),
                                    pixels + clamped_window)

            # add paddle
            paddle_pxls = self.update_paddle(labels)
            pixels = np.hstack((pixels, np.expand_dims(paddle_pxls, 1)))

            # visualize labels as well
            labelsrep = np.repeat(labels, img_shape[0] // rbm.n_labels)
            labelsrgb = np.tile(np.expand_dims(labelsrep, 1), (1, 3))
            labelsrgb[:, :2] = 0
            pixels = np.hstack((pixels, np.expand_dims(labelsrgb, 1)))

            self.image.set_data(pixels)
            return self.image,  self.points


if __name__ == "__main__":
    img_shape = (36, 48)
    n_pixels = np.prod(img_shape)
    # Load Pong data
    data_name = 'pong_var_start{}x{}'.format(*img_shape)
    with np.load('../datasets/' + data_name + '.npz') as d:
        train_set, _, test_set = d[d.keys()[0]]
    # Load rbm
    rbm_name = data_name + '_crbm.pkl'
    with open('saved_rbms/' + rbm_name, 'rb') as f:
        rbm = cPickle.load(f)
    # file with sampled states
    sample_file = 'pong_win48_all_chunk000'
    print(sample_file)
    with np.load(get_data_path('gibbs_sampling') + sample_file + '.npz') as d:
        vis_samples = d['vis']

    if len(vis_samples.shape) == 2:
        vis_samples = np.expand_dims(vis_samples, 0)
        # hid_samples = np.expand_dims(hid_samples, 1)
    print('Number of instances in file: ' + str(vis_samples.shape[0]))
    # # marginal visible probabilities can be calculated from hidden states
    # vis_probs = rbm.sample_v_given_h(hid_samples)[0] -> only if binary

    winsize = img_shape[1]

    # set sup figure
    fig = plt.figure()
    width = .7
    ax1 = fig.add_axes([.05, .2, width, width*3/4])
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax2 = fig.add_axes([width - .02, .2, .2, width*3/4])
    ax2.set_ylim([-.5, img_shape[0] - .5])
    ax2.set_xlim([0, 1.1])
    ax2.xaxis.set_ticks([0., 0.5, 1.])
    ax2.tick_params(left='off', right='off',
                    labelleft='off', labelright='off')
    ax2.set_xlabel('P(last column)')

    # if a homogeneous picture is used here it does not work -> wtf
    image = ax1.imshow(np.random.rand(*img_shape), interpolation='Nearest',
                       cmap='gray', origin='lower', animated=True)
    # barh doesnt work because apparently BarContainer has no 'set_visible'
    points, = ax2.plot([], [], 'bo')
    # ex. for adding dynamic text; needs to be inside bounding box to be updated
    # lab_text = ax.text(0.95, 0.01, '', va='bottom', ha='right',
    #                    transform=ax.transAxes, color='green')

    # Produce animations of several examples
    for i in range(len(vis_samples[:2])):
        print('Processing instance {} of {}...'
              ''.format(i + 1, vis_samples.shape[0]))
        upd = Pong_updater(image, points, img_shape=img_shape, rbm=rbm,
                           winsize=winsize, clamp_duration=120.)
        ani = animation.FuncAnimation(
            fig, upd, frames=vis_samples[i], interval=100., blit=True,
            repeat_delay=2000)
        ani.save('figures/animation_{}_{}.mp4'.format(sample_file, i))
        # plt.show()

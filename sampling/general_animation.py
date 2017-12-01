from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# since this script is still called from my office pc, expand path variable
import sys
sys.path.insert(0, '../')
from lif_pong.utils.data_mgmt import get_data_path, make_figure_folder
from lif_pong.utils import average_helper
from pong_agent import Pong_agent


class Pong_updater(object):
    def __init__(self, image, points, img_shape=(36, 48), win_size=None,
                 clamp_duration=1, paddle_len=0, max_step=0.):
        self.image = image
        self.points = points
        self.img_shape = img_shape
        self.n_labels = 12  # remove hard-coded version later
        self.show_labels = False

        # for overlay
        self.win_size = win_size
        self.clamp_pos = -1
        self.clamp_duration = int(clamp_duration)
        # no. of frames during which clamp is constant

        # for agent
        if max_step == 0:
            max_step = 1./clamp_duration
        if paddle_len == 0:
            self.agent = None
        else:
            self.agent = Pong_agent(img_shape[0], paddle_len, max_step)

    def update_paddle(self, prediction):
        self.agent.update_pos(prediction)
        paddle_pxls = np.zeros((self.img_shape[0], 3))
        paddle_len = self.agent.paddle_len
        pos = self.agent.pos

        if pos <= paddle_len/2:
            paddle_pxls[:paddle_len, 1] = 1
        elif pos + paddle_len/2 >= self.img_shape[0] - 1:
            paddle_pxls[-paddle_len:, 1] = 1
        else:
            paddle_pxls[int(pos) - paddle_len//2:
                        int(pos + np.round(paddle_len/2)), 1] = 1
        return paddle_pxls

    def __call__(self, state):
            img_shape = self.img_shape
            t = state[0]
            vis_state = state[1]
            if self.agent is not None:
                prediction = vis_state[:np.prod(img_shape)].reshape(
                    img_shape)[:, -1]
                prediction = np.average(np.arange(len(prediction)),
                                        weights=prediction)

            pixels = vis_state[:np.prod(img_shape)].astype(float)
            pixels = pixels.reshape(img_shape)
            self.points.set_data(pixels[:, -1],
                                 np.arange(img_shape[0]) - .5)

            # to rgb
            pixels = np.tile(np.expand_dims(pixels, 2), (1, 1, 3))
            labels = vis_state[np.prod(img_shape):]

            # overlay clamping window (update every 'clamp_duration' steps)
            if self.win_size is not None:
                clamped_window = np.zeros(img_shape + (3,))
                end = int(self.clamp_pos)
                start = max(0, (end - self.win_size))
                clamped_window[:, start:end, 2] = .2
                pixels = np.clip(pixels + clamped_window, 0., 1.)
                if t % self.clamp_duration == 0:
                    self.clamp_pos += 1
                    self.clamp_pos = self.clamp_pos % (img_shape[1] + 1)

            # add paddle
            if self.agent is not None:
                paddle_pxls = self.update_paddle(prediction)
                pixels = np.hstack((pixels, np.expand_dims(paddle_pxls, 1)))

            # visualize labels as well
            if self.show_labels:
                labelsrep = np.repeat(labels, img_shape[0] // self.n_labels)
                labelsrgb = np.tile(np.expand_dims(labelsrep, 1), (1, 3))
                pixels = np.hstack((pixels, np.expand_dims(labelsrgb, 1)))

            self.image.set_data(pixels)
            return self.image, self.points


def make_animation(fig_name, img_shape, win_size, vis_samples, paddle_len=0,
                   predictions=None, clamp_interval=1, anim_interval=100.):
    # set up figure
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
    image = ax1.imshow(np.zeros(img_shape), vmin=0., vmax=1.,
                       interpolation='Nearest', cmap='gray', origin='lower',
                       animated=True)
    # barh doesnt work because apparently BarContainer has no 'set_visible'
    points, = ax2.plot([], [], 'bo')
    # ex. for adding dynamic text; must be inside bounding box to be updated
    # lab_text = ax.text(0.95, 0.01, '', va='bottom', ha='right',
    #                    transform=ax.transAxes, color='green')

    # Produce animation
    if predictions is not None:
        frames = zip(range(len(vis_samples)), vis_samples, predictions)
    else:
        frames = zip(range(len(vis_samples)), vis_samples)
    upd = Pong_updater(image, points, img_shape=img_shape,
                       win_size=win_size, paddle_len=paddle_len,
                       clamp_duration=clamp_interval)
    ani = animation.FuncAnimation(fig, upd, frames=frames, blit=True,
                                  interval=anim_interval, repeat_delay=2000)
    ani.save(fig_name + '.mp4')

if __name__ == '__main__':
    img_shape = (36, 48)
    n_pixels = np.prod(img_shape)
    pot_str = 'pong'

    # file with sampled states
    sample_file = pot_str + '_win48_avg_chunk000'
    print(sample_file)
    with np.load(get_data_path('gibbs_sampling') + sample_file + '.npz') as d:
        vis_samples = d['vis']
        win_size = d['win_size']
        sample_idx = d['data_idx']
        if len(vis_samples.shape) == 2:
            vis_samples = np.expand_dims(vis_samples, 0)
        print('Number of instances in file: ' + str(vis_samples.shape[0]))

    # # Get predictions -----> this does not work yet; instead I computed the prediction in the call method
    # pred_file = pot_str + '_win{}_prediction_incomplete'.format(win_size)
    # with np.load(get_data_path('gibbs_sampling') + pred_file + '.npz') as d:
    #     last_col = d['last_col']
    #     pred_idx = d['data_idx']

    # # filter the data to keep only those instances which are present in both
    # # samples and predictions
    # last_col = last_col[np.argsort(pred_idx)]
    # vis_samples = vis_samples[np.argsort(sample_idx)]
    # sample_idx = np.sort(sample_idx)
    # pred_idx = np.sort(pred_idx)

    # data_idx = np.intersect1d(sample_idx, pred_idx)
    # last_col = last_col[np.where(np.in1d(pred_idx, data_idx))]
    # vis_samples = vis_samples[np.where(np.in1d(sample_idx, data_idx))]

    # # get predicted positions from last columns' mean activity
    # predictions = np.zeros(last_col.shape[:-1])
    # for i in range(len(last_col)):
    #     predictions[i] = average_helper(img_shape[0], last_col[i])
    # predictions = predictions.T

    fig_name = 'test'
    n = 3
    for i in range(n):
        print('Making animation {} of {}'.format(i + 1, n))
        make_animation(fig_name + str(i), img_shape, win_size, vis_samples[i],
                       paddle_len=3)

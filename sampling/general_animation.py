from __future__ import division
from __future__ import print_function
import numpy as np
import sys
import os
import yaml
from scipy.ndimage import convolve1d
from lif_pong.utils.data_mgmt import make_figure_folder, load_images
from lif_pong.utils import average_helper
from pong_agent import Pong_agent
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = u'/home/hd/hd_hd/hd_kq433/ffmpeg-3.4.1-64bit-static/ffmpeg'


class Pong_updater(object):
    def __init__(self, image, points, img_shape=(40, 48), win_size=None,
                 clamp_duration=1, paddle_len=0, max_step=0.):
        self.image = image
        self.points = points
        self.img_shape = img_shape
        self.n_labels = 10  # remove hard-coded version later

        # for overlay
        self.win_size = win_size
        # matplotlib seems to call the update function twice with the same parameters
        # before starting a simulation. The -3 compensates this bug.
        self.clamp_pos = -3
        # no. of frames during which clamp is constant
        self.clamp_duration = int(clamp_duration)

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
                        int(pos + np.round(paddle_len/2))] = 1

            # C1 255,127,14
            paddle_pxls[:, 0] *= 255/255
            paddle_pxls[:, 1] *= 127/255
            paddle_pxls[:, 2] *= 14/255

            # # C2 44,160,44
            # paddle_pxls[:, 0] *= 44/255
            # paddle_pxls[:, 1] *= 160/255
            # paddle_pxls[:, 2] *= 44/255
        return paddle_pxls

    def __call__(self, state):
            img_shape = self.img_shape
            t = state[0]
            vis_state = state[1][:np.prod(img_shape)]
            pixels = vis_state.reshape(img_shape)
            if self.agent is not None:
                prediction = pixels[:, -1]
                if np.all(prediction == 0):
                    prediction = np.ones_like(prediction)
                prediction = np.average(np.arange(len(prediction)),
                                        weights=prediction)

            # to rgb
            rgb_pixels = np.tile(np.expand_dims(pixels, 2), (1, 1, 3))

            # overlay clamping window (update every 'clamp_duration' steps)
            if self.win_size is not None:
                clamped_window = np.zeros(img_shape + (3,))
                end = int(self.clamp_pos)
                start = max(0, (end - self.win_size))
                clamped_window[:, start:end, 0] = 31/255
                clamped_window[:, start:end, 1] = 119/255
                clamped_window[:, start:end, 2] = 180/255
                clamped_window *= .25
                rgb_pixels = np.clip(rgb_pixels + clamped_window, 0., 1.)
                if t % self.clamp_duration == 0:
                    self.clamp_pos += 1
                    self.clamp_pos = self.clamp_pos % (img_shape[1] + 1)

            # add paddle
            if self.agent is not None:
                paddle_pxls = self.update_paddle(prediction)
                rgb_pixels = np.hstack((rgb_pixels,
                                        np.expand_dims(paddle_pxls, 1)))

            self.image.set_data(rgb_pixels)
            self.points.set_data(pixels[:, -1], np.arange(img_shape[0]) - .5)

            # # for testing layout etc. or plotting a single frame
            # if t == 20*10:
            #     target = 32.77
            #     fig = plt.figure(figsize=(10, 7))
            #     left, width = .05, .7
            #     bottom, height = .1, .8
            #     pad = 0.02
            #     left_p = left + width + pad
            #     width_p = .17
            #     img_rect = [left, bottom, width, height]
            #     pred_rect = [left_p, bottom, width_p, height]

            #     ax_img = fig.add_axes(img_rect)
            #     ax_pred = fig.add_axes(pred_rect)   # , adjustable='box', aspect=img_shape[0]/img_shape[1]
            #     ax_img.xaxis.set_visible(False)
            #     ax_img.yaxis.set_visible(False)

            #     ax_pred.set_xlim([0., 1.1])
            #     ax_pred.set_ylim([-.5, img_shape[0] - .5])
            #     ax_pred.xaxis.set_ticks([0., 0.5, 1.])
            #     ax_pred.tick_params(left='off', right='off', top='on',
            #                         labelleft='off', labelright='off')
            #     ax_pred.set_xlabel('P (last column)')

            #     image = ax_img.imshow(np.zeros(img_shape), vmin=0., vmax=1.,
            #                        interpolation='Nearest', cmap='gray', origin='lower')
            #     points, = ax_pred.plot([], [], 'ko')

            #     image.set_data(rgb_pixels)
            #     points.set_data(pixels[:, -1], np.arange(img_shape[0]))
            #     ax_pred.plot([0., 1.1], [target, target], '-', color='C2', linewidth=2)
            #     fig.savefig('frame.pdf')
            #     sys.exit()


            return self.image, self.points


def make_animation(fig_name, img_shape, win_size, vis_samples, paddle_len=0,
                   clamp_interval=1, anim_interval=10., target=None):
    # set up figure
    # fig = plt.figure()
    # width = .7
    # ax1 = fig.add_axes([.05, .2, width, width*3/4])
    # ax1.xaxis.set_visible(False)
    # ax1.yaxis.set_visible(False)
    # ax2 = fig.add_axes([width - .02, .2, .2, width*3/4])
    # ax2.set_ylim([-.5, img_shape[0] - .5])
    # ax2.set_xlim([0, 1.1])
    # ax2.xaxis.set_ticks([0., 0.5, 1.])
    # ax2.tick_params(left='off', right='off',
    #                 labelleft='off', labelright='off')
    # ax2.set_xlabel('P(last column)')

    # define axes locations
    fig = plt.figure(figsize=(10, 7))
    left, width = .05, .7
    bottom, height = .1, .8
    pad = 0.02
    left_p = left + width + pad
    width_p = .17
    img_rect = [left, bottom, width, height]
    pred_rect = [left_p, bottom, width_p, height]

    ax_img = fig.add_axes(img_rect)
    ax_pred = fig.add_axes(pred_rect)   # , adjustable='box', aspect=img_shape[0]/img_shape[1]
    ax_img.xaxis.set_visible(False)
    ax_img.yaxis.set_visible(False)

    ax_pred.set_xlim([0., 1.1])
    ax_pred.set_ylim([-.5, img_shape[0] - .5])
    ax_pred.xaxis.set_ticks([0., 0.5, 1.])
    ax_pred.tick_params(left='off', right='off', top='on',
                        labelleft='off', labelright='off')
    ax_pred.set_xlabel('P (last column)')

    image = ax_img.imshow(np.zeros(img_shape), vmin=0., vmax=1.,
                       interpolation='Nearest', cmap='gray', origin='lower')
    points, = ax_pred.plot([], [], 'ko')
    # ex. for adding dynamic text; must be inside bounding box to be updated
    # lab_text = ax.text(0.95, 0.01, '', va='bottom', ha='right',
    #                    transform=ax.transAxes, color='green')
    if target is not None:
        ax_pred.plot([0, 1.1], [target, target], '-', color='C2', linewidth=2)

    # Produce animation
    frames = zip(range(len(vis_samples)), vis_samples)
    upd = Pong_updater(image, points, img_shape=img_shape,
                       win_size=win_size, paddle_len=paddle_len,
                       clamp_duration=clamp_interval)
    ani = animation.FuncAnimation(fig, upd, frames=frames, blit=True,
                                  interval=anim_interval, repeat_delay=2000)
    ani.save(fig_name + '.mp4', writer='ffmpeg')


def main(config_dict):
    data_path = config_dict['sample_data']
    img_shape = tuple(config_dict['img_shape'])
    clamp_interval = config_dict['n_samples']
    win_size = config_dict['win_size']
    n_imgs = config_dict['n_imgs']
    if 'data_name' in config_dict.keys() and 'start_idx' in config_dict.keys():
        # for target visualization
        _, _, test_set = load_images(config_dict['data_name'])
        start_idx = config_dict['start_idx']
        targets = test_set[0].reshape((-1,) + img_shape)[..., -1]
        targets = average_helper(img_shape[0], targets[start_idx:start_idx + n_imgs])
    else:
        targets = [None]*n_imgs
    save_name = os.path.join(make_figure_folder(), config_dict['save_name'])

    with np.load(data_path) as d:
        samples = d['samples'][:n_imgs]
        vis_samples = samples[..., :np.prod(img_shape) + img_shape[0]//3].astype(float)
    # maybe average like in lif_inspect_samples
    kernel = np.ones(clamp_interval)/clamp_interval
    vis_samples = convolve1d(vis_samples, kernel, axis=1)

    for i in range(len(vis_samples)):
        print('Making animation {} of {}'.format(i + 1, len(vis_samples)))
        make_animation(save_name + '_{}'.format(i), img_shape, win_size, vis_samples[i],
                       paddle_len=4, clamp_interval=clamp_interval, target=targets[i])



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Wrong number of arguments. Please provide a yaml that specifies'
              ' the configuration.')
        sys.exit()

    # load list of identifiers used for selecting data
    with open(sys.argv[1]) as f:
        config = yaml.load(f)

    main(config)

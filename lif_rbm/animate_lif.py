# animation for data from lif simulation
from __future__ import division
import numpy as np
import cPickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
sys.path.insert(0, '../gibbs_rbm')
from rbm import RBM, CRBM

sampling_interval = 10.


class Pong_updater(object):
    def __init__(self, image, points, img_shape, rbm, winsize,
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
            # overlay clamping window (update interval is 10ms)
            self.clamp_position += sampling_interval / self.clamp_duration
            self.clamp_position = self.clamp_position % (self.img_shape[1] + 1)
            clamped_window = np.zeros(img_shape + (3,))
            end = int(self.clamp_position)
            start = max(0, (end - self.winsize))
            clamped_window[:, start:end, 2] = .2
            pixels = np.minimum(np.ones_like(pixels), pixels + clamped_window)

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


# def updatefig(frame, rbm, img_shape, winsize):
#     global image, points
#     # idx = i % sample_imgs.shape[0]
#     # if clamped_label:
#     #     lab_text.set_text('Active label: {}'.format(active_lab[idx]))
#     # im.set_data(sample_imgs[idx])
#     # im.set_data(hid_samples.reshape(hid_samples.shape[0], 20, 20)[idx])
#     image.set_data(frame)
#     points.set_data(np.ones(50)*.5, np.linspace(0, 35))

#     return image,  points


img_shape = (36, 48)
n_pixels = np.prod(img_shape)
# Load Pong data
data_name = 'pong_var_start{}x{}'.format(*img_shape)
with np.load('../datasets/' + data_name + '.npz') as d:
    train_set, _, test_set = d[d.keys()[0]]
# Load rbm
rbm_name = data_name + '_crbm.pkl'
with open('../gibbs_rbm/saved_rbms/' + rbm_name, 'rb') as f:
    rbm = cPickle.load(f)
# file with sampled states from sbs simulation
sample_file = 'pong_window_samples'
with np.load('Data/' + sample_file + '.npz') as d:
    samples = d[d.files[0]]

vis_samples = samples[:, :rbm.n_visible]
pixel_samples = samples[:, :n_pixels]
hid_samples = samples[:, rbm.n_visible:]
# marginal visible probabilities can be calculated from hidden states
vis_probs = rbm.sample_v_given_h(hid_samples)[0]

winsize = img_shape[1]//2

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

# Produce animations of several random examples
# modify later to fetch data from files generated by lif_sampling.py
np.random.seed(125575)
example_id = np.random.choice(test_set[0].shape[0], size=1, replace=False)
for i, example in enumerate(test_set[0][example_id]):
    # fig, frames = get_frames(rbm, example, winsize=100)
    upd = Pong_updater(image, points, img_shape=img_shape, rbm=rbm,
                       winsize=winsize)
    ani = animation.FuncAnimation(fig, upd, frames=vis_samples,
                                  # fargs=(rbm, img_shape, winsize),
                                  interval=50., blit=True, repeat_delay=2000)
    ani.save('Figures/animation_' + data_name + str(i) + '.mp4')
    # plt.show()

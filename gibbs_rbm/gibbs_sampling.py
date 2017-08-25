from __future__ import division
from __future__ import print_function
import numpy as np
import cPickle
from util import get_windowed_image_index
from scipy.ndimage import convolve1d
# from skimage.measure import block_reduce


# shouldn't be used with more than a few imgs (due to memory limitations)
def run_simulation(rbm, n_steps, imgs, v_init=None,
                   burnin=500, clamp_fct=None):
    if clamp_fct is None:
        return rbm.draw_samples(burnin + n_steps, v_init=v_init)[burnin:]

    if len(imgs.shape) == 1:
        # special case for one image
        imgs = np.expand_dims(imgs, 0)
    vis_samples = np.zeros((n_steps, imgs.shape[0], rbm.n_visible)).squeeze()
    hid_samples = np.zeros((n_steps, imgs.shape[0], rbm.n_hidden)).squeeze()

    # burnin
    _, clamped_ind = clamp_fct(0)
    temp = rbm.draw_samples(burnin, v_init=v_init,
                            clamped=clamped_ind,
                            clamped_val=imgs[:, clamped_ind])
    v_init = temp[-1, ..., :rbm.n_visible]

    t = 0
    while t < n_steps:
        # get clamped_ind and next callback time
        delta_t, clamped_ind = clamp_fct(t)
        n_samples = delta_t if t + delta_t <= n_steps else n_steps - t

        # there's a bug in this method if called with more than one img
        # temp = rbm.draw_samples(n_samples, v_init=v_init,
        #                         clamped=clamped_ind,
        #                         clamped_val=imgs[:, clamped_ind])
        #
        # vis_samples[t:t + n_samples] = temp[..., :rbm.n_visible]
        # hid_samples[t:t + n_samples] = temp[..., rbm.n_visible:]

        unclamped_vis = rbm.sample_with_clamped_units(
            n_samples, clamped_ind, imgs[:, clamped_ind])
        unclamped_ind = np.setdiff1d(np.arange(rbm.n_visible), clamped_ind)
        vis_samples[t:t + n_samples, :, clamped_ind] = imgs[:, clamped_ind]
        vis_samples[t:t + n_samples, :, unclamped_ind] = unclamped_vis
        hid_samples = 0

        # if the gibbs chain should be continued
        v_init = vis_samples[-1]
        t += delta_t
    return vis_samples, hid_samples


# Custom clamping methods---adapted from lif_sampling.py
class Clamp_anything(object):
    # refresh times must be a list
    def __init__(self, refresh_times, clamped_idx):
        if len(refresh_times) == 1 and len(refresh_times) != len(clamped_idx):
            self.clamped_idx = np.expand_dims(clamped_idx, 0)
        else:
            self.clamped_idx = clamped_idx
        self.refresh_times = refresh_times

    def __call__(self, t):
        try:
            i = np.where(np.isclose(self.refresh_times, t))[0][0]
        except IndexError:
            print('No matching clamping time stamp; this should not happen.')
            return float('inf'), [], []

        if i < len(self.refresh_times) - 1:
            dt = self.refresh_times[i + 1] - t
        else:
            dt = float('inf')
        return dt, self.clamped_idx[i]


class Clamp_window(object):
    def __init__(self, img_shape, interval, win_size):
        self.interval = interval
        self.img_shape = img_shape
        self.win_size = win_size

    def __call__(self, t):
        end = min(int(t / self.interval), self.img_shape[1])
        clamped_idx = get_windowed_image_index(
            self.img_shape, end, self.win_size)
        return self.interval, clamped_idx


def average_pool(arr, width, stride=None, mode='default'):
    # array needs to have shape (n_imgs, n_t, n_pxls)
    if stride is None:
        stride = width
    kernel = np.ones(width) / width
    filtered = convolve1d(arr, kernel, mode='constant', axis=1)
    # if no boundary effects should be visible
    if mode == 'valid':
        return filtered[:, width//2:-(width//2):stride]
    # startpoint is chosen such that the remainder of the division is
    # distributed evenly between left and right boundary
    start = ((arr.shape[1] - 1) % stride) // 2
    return filtered[:, start::stride]

# pong pattern completion
# Load Pong data and rbm
img_shape = (36, 48)
n_pxls = np.prod(img_shape)
data_name = 'pong_var_start{}x{}'.format(*img_shape)
with np.load('../datasets/' + data_name + '.npz') as d:
    train_set, _, test_set = d[d.keys()[0]]
with open('saved_rbms/' + data_name + '_crbm.pkl', 'rb') as f:
    rbm = cPickle.load(f)
np.random.seed(5116838)

imgs = test_set[0][:1000]
n_samples = 100
burnin = 20
duration = img_shape[1] * (n_samples + burnin)
# design network input here
win_size = 48
fname = 'pong_win{}'.format(win_size)
clamp = Clamp_window(img_shape, n_samples + burnin, win_size)
# clamp = Clamp_anything([0.], get_windowed_image_index(
#             img_shape, (4 + img_shape[1])//2, img_shape[1]))

# due to memory requirements not all instances can be put into an array
n_chunks = int(np.ceil(imgs.shape[0] / 100))
n_each, remainder = imgs.shape[0] // n_chunks, imgs.shape[0] % n_chunks
chunk_sizes = np.array([0] + [n_each] * n_chunks)
chunk_sizes[1:(remainder + 1)] += 1
chunk_ind = np.cumsum(chunk_sizes)
for j, chunk in enumerate(np.array_split(imgs, n_chunks)):
    chunk_init = \
        np.random.randint(2, size=(len(chunk), rbm.n_visible)).squeeze()
    vis_samples, _ = \
        run_simulation(rbm, duration, chunk, burnin=100, clamp_fct=clamp)
    avg_prediction = average_pool(np.swapaxes(vis_samples, 0, 1), 100)
    # save data
    np.savez_compressed('data/{}_chunk{}'.format(fname, j), vis=avg_prediction)

# n_imgs = 2
# duration = 1234
# test_data = test_set[0]
# example_ids = np.random.choice(test_data.shape[0], size=n_imgs, replace=False)
# # test_data = test_data[example_ids]
# test_data = test_data[:n_imgs]
# # # binary values:
# # test_data = np.round(test_data)

# # design network input here
# # clamp = Clamp_window(img_shape, 100, 48)
# clamp = Clamp_anything([0.], get_windowed_image_index(
#             img_shape, (4 + img_shape[1])//2, img_shape[1]))

# vis_samples, _ = \
#     run_simulation(rbm, duration, test_data, burnin=100, clamp_fct=clamp)

# # save data
# np.savez_compressed('data/test', vis=vis_samples)

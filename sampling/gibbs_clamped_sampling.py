from __future__ import division
from __future__ import print_function
import numpy as np
from lif_pong.utils import get_windowed_image_index

# shouldn't be used with more than a few imgs (due to memory limitations)
def run_simulation(rbm, n_steps, imgs, v_init=None, burnin=500, binary=False,
                   clamp_fct=None, continuous=True, bin_imgs=False):
    if clamp_fct is None:
        return rbm.draw_samples(burnin + n_steps, v_init=v_init)[burnin:]
    if bin_imgs:
        imgs = (imgs > .5)*1.
        assert np.all(np.logical_or(imgs == 1, imgs == 0))

    if len(imgs.shape) == 1:
        # special case for one image
        imgs = np.expand_dims(imgs, 0)
    vis_samples = np.zeros((n_steps, imgs.shape[0], rbm.n_visible))
    hid_samples = np.zeros((n_steps, imgs.shape[0], rbm.n_hidden))

    # burnin
    _, clamped_ind = clamp_fct(0)
    if burnin > 0:
        temp = rbm.draw_samples(burnin, v_init=v_init, clamped=clamped_ind,
                                clamped_val=imgs[:, clamped_ind])
        v_init = temp[-1, ..., :rbm.n_visible]

    t = 0
    while t < n_steps:
        # get clamped_ind and next callback time
        delta_t, clamped_ind = clamp_fct(t)
        n_samples = delta_t if t + delta_t <= n_steps else n_steps - t

        unclamped_vis, hid = rbm.sample_with_clamped_units(
            n_samples, clamped_ind, imgs[:, clamped_ind], v_init=v_init,
            binary=binary)
        if imgs.shape[0] == 1:
            unclamped_vis = np.expand_dims(unclamped_vis, 1)
        unclamped_ind = np.setdiff1d(np.arange(rbm.n_visible), clamped_ind)
        vis_samples[t:t + n_samples, :, clamped_ind] = imgs[:, clamped_ind]
        vis_samples[t:t + n_samples, :, unclamped_ind] = unclamped_vis
        hid_samples[t:t + n_samples] = hid

        # # probably slower
        # temp = rbm.draw_samples(n_samples, v_init=v_init,
        #                         clamped=clamped_ind,
        #                         clamped_val=imgs[:, clamped_ind],
        #                         binary=binary)

        # vis_samples[t:t + n_samples] = temp[..., :rbm.n_visible]
        # # hid_samples[t:t + n_samples] = temp[..., rbm.n_visible:]

        # if the gibbs chain should be continued
        if continuous:
            v_init = vis_samples[t + n_samples - 1]
        else:
            v_init = None
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
    def __init__(self, img_shape, interval, win_size=None):
        self.interval = interval
        self.img_shape = img_shape
        if win_size is None:
            win_size = img_shape[1]
        self.win_size = win_size

    def __call__(self, t):
        end = min(int(t / self.interval), self.img_shape[1])
        clamped_idx = get_windowed_image_index(
            self.img_shape, end, self.win_size)
        return self.interval, clamped_idx
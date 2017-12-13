# script for drawing gibbs samples from an RBM with dynamic clamping
from __future__ import division
from __future__ import print_function
import numpy as np
import yaml
import sys
import lif_pong.training.rbm as rbm_pkg
from lif_pong.utils.data_mgmt import make_data_folder, load_images, get_rbm_dict
from lif_pong.utils import get_windowed_image_index


# shouldn't be used with more than a few imgs (due to memory limitations)
def run_simulation(rbm, n_steps, imgs, v_init=None, burnin=500,
                   clamp_fct=None):
    if clamp_fct is None:
        return rbm.draw_samples(burnin + n_steps, v_init=v_init)[burnin:]

    if len(imgs.shape) == 1:
        # special case for one image
        imgs = np.expand_dims(imgs, 0)
    vis_samples = np.zeros((n_steps, imgs.shape[0], rbm.n_visible))
    # hid_samples = np.zeros((n_steps, imgs.shape[0], rbm.n_hidden))
    hid_samples = 0

    # burnin
    _, clamped_ind = clamp_fct(0)
    temp = rbm.draw_samples(burnin, v_init=v_init, clamped=clamped_ind,
                            clamped_val=imgs[:, clamped_ind])
    v_init = temp[-1, ..., :rbm.n_visible]

    t = 0
    while t < n_steps:
        # get clamped_ind and next callback time
        delta_t, clamped_ind = clamp_fct(t)
        n_samples = delta_t if t + delta_t <= n_steps else n_steps - t

        unclamped_vis = rbm.sample_with_clamped_units(
            n_samples, clamped_ind, imgs[:, clamped_ind], v_init=v_init)
        if imgs.shape[0] == 1:
            unclamped_vis = np.expand_dims(unclamped_vis, 1)
        unclamped_ind = np.setdiff1d(np.arange(rbm.n_visible), clamped_ind)
        vis_samples[t:t + n_samples, :, clamped_ind] = imgs[:, clamped_ind]
        vis_samples[t:t + n_samples, :, unclamped_ind] = unclamped_vis

        # # probably slower
        # temp = rbm.draw_samples(n_samples, v_init=v_init,
        #                         clamped=clamped_ind,
        #                         clamped_val=imgs[:, clamped_ind])

        # vis_samples[t:t + n_samples] = temp[..., :rbm.n_visible]
        # # hid_samples[t:t + n_samples] = temp[..., rbm.n_visible:]

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


def main(general_dict, identifiers):
    # pass arguments from dictionaries to simulation
    n_samples = general_dict['n_samples']
    img_shape = tuple(general_dict['img_shape'])
    # load data
    _, _, test_set = load_images(general_dict['data_name'])
    rbm = rbm_pkg.load(get_rbm_dict(general_dict['rbm_name']))
    rbm.set_seed(general_dict['seed'])
    start = general_dict['start_idx']
    end = start + general_dict['chunksize']
    winsize = general_dict['winsize']

    duration = (img_shape[1] + 1) * n_samples
    clamp = Clamp_window(img_shape, n_samples, winsize)

    print('Running gibbs simulation for instances {} to {}'.format(start, end))
    vis_samples, _ = run_simulation(
        rbm, duration, test_set[0][start:end], burnin=general_dict['burn_in'],
        clamp_fct=clamp)

    # compared to the lif-methods, the method returns an array with
    # shape [n_steps, n_imgs, n_vis]. Hence, swap axes.
    np.savez_compressed('samples', samples=np.swapaxes(vis_samples, 0, 1))

    # # leave out average pool? (done by plot_predictions)
    # vis_samples = average_pool(np.swapaxes(vis_samples, 0, 1), n_samples,
    #                            stride=n_samples)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Wrong number of arguments. Please provide a yaml-config file.')
        sys.exit()
    with open(sys.argv[1]) as configfile:
        config = yaml.load(configfile)['stub']

    general_dict = config.pop('general')
    identifiers = config.pop('identifier')

    main(general_dict, identifiers)

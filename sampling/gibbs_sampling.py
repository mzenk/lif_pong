# script for drawing gibbs samples from an RBM with dynamic clamping
from __future__ import division
from __future__ import print_function
import numpy as np
import cPickle
import sys
import lif_pong.training.rbm as rbm_pkg
from lif_pong.utils.data_mgmt import make_data_folder, load_images, get_rbm_dict
from lif_pong.utils import get_windowed_image_index, average_pool


# shouldn't be used with more than a few imgs (due to memory limitations)
def run_simulation(rbm, n_steps, imgs, v_init=None, burnin=500, binary=False,
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
            n_samples, clamped_ind, imgs[:, clamped_ind], v_init=v_init,
            binary=binary)
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
        v_init = vis_samples[t + n_samples - 1]
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


# pong pattern completion
if len(sys.argv) < 5:
    print('Please specify the arguments:'
          ' pong/gauss, start_idx, chunk_size, win_size, [name_modifier]')
    sys.exit()
pot_str = sys.argv[1]
start = int(sys.argv[2])
chunk_size = int(sys.argv[3])
win_size = int(sys.argv[4])
if len(sys.argv) == 6:
    modifier = '_' + str(sys.argv[5])
else:
    modifier = ''

save_name = pot_str + '_win{}{}_avg_chunk{:03d}'.format(win_size, modifier,
                                                        start // chunk_size)

img_shape = (36, 48)
n_pxls = np.prod(img_shape)
np.random.seed(5116838)
# settings for sampling/clamping
n_samples = 10
between_burnin = 0
# no burnin once actual simulation has started
duration = (img_shape[1] + 1) * (n_samples + between_burnin)
clamp = Clamp_window(img_shape, n_samples + between_burnin, win_size)
# clamp = Clamp_anything([0.], get_windowed_image_index(
#             img_shape, (4 + img_shape[1])//2, img_shape[1]))

# Load Pong data and rbm
data_name = pot_str + '_var_start{}x{}'.format(*img_shape)
rbm_name = data_name + '_crbm'
if pot_str == 'knick':
    rbm_name = 'pong_var_start{}x{}_crbm'.format(*img_shape)
_, _, test_set = load_images(data_name)
rbm = rbm_pkg.load(get_rbm_dict(rbm_name))

end = min(start + chunk_size, len(test_set[0]))
idx = np.arange(start, end)
chunk = test_set[0][idx]
targets = test_set[1][idx]

print('Running gibbs simulation for instances {} to {}'.format(start, end))
vis_samples, _ = \
    run_simulation(rbm, duration, chunk, burnin=100, clamp_fct=clamp)
vis_samples = average_pool(np.swapaxes(vis_samples, 0, 1), n_samples,
                           stride=n_samples + between_burnin,
                           offset=between_burnin)

# save averaged(!) samples
np.savez_compressed(make_data_folder() + save_name,
                    vis=vis_samples, win_size=win_size, data_idx=idx)


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

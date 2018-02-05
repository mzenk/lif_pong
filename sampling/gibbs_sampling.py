# script for drawing gibbs samples from an RBM with dynamic clamping
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import sys
import lif_pong.training.rbm as rbm_pkg
from lif_pong.utils.data_mgmt import make_data_folder, load_images, get_rbm_dict
from gibbs_clamped_sampling import run_simulation, Clamp_window, Clamp_anything


# pong pattern completion
if len(sys.argv) < 5:
    print('Please specify the arguments:'
          ' data_name, rbm_name, winsize, start_idx, chunk_size, [save_name]')
    sys.exit()
data_name = sys.argv[1]
rbm_name = sys.argv[2]
winsize = int(sys.argv[3])
start_idx = int(sys.argv[4])
chunk_size = int(sys.argv[5])
if len(sys.argv) == 7:
    save_name = sys.argv[6]
else:
    save_name = '{}_win{}_{}'.format(data_name, winsize, start_idx)

img_shape = (36, 48)
n_pxls = np.prod(img_shape)
np.random.seed(5116838)
# settings for sampling/clamping
n_samples = 20
between_burnin = 0
# no burnin once actual simulation has start_idxed
duration = (img_shape[1] + 1) * (n_samples + between_burnin)
clamp = Clamp_window(img_shape, n_samples + between_burnin, winsize)
# duration = n_samples
# clamp = Clamp_anything([0.], get_windowed_image_index(
#             img_shape, int(.2*img_shape[1])))

# Load Pong data and rbm
_, _, test_set = load_images(data_name)
rbm = rbm_pkg.load(get_rbm_dict(rbm_name))

end = min(start_idx + chunk_size, len(test_set[0]))
idx = np.arange(start_idx, end)
chunk = test_set[0][idx]

print('Running gibbs simulation for instances {} to {}'.format(start_idx, end))
vis_samples, _ = run_simulation(rbm, duration, chunk,
                                burnin=100, clamp_fct=clamp, binary=True)
vis_samples = np.swapaxes(vis_samples, 0, 1)

# save samples
np.savez_compressed(os.path.join(make_data_folder(), save_name),
                    samples=vis_samples, data_idx=idx)

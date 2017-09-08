from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import lif_clamped_sampling as lifsampl
from utils.data_mgmt import make_data_folder, load_images, load_rbm
from rbm import RBM, CRBM

# clamped sampling: Pong
if len(sys.argv) != 5:
    print('Please specify the arguments:'
          ' pong/gauss, start_idx, chunk_size, win_size')
    sys.exit()

# oarameters that can be changed via command line
pot_str = sys.argv[1]
start = int(sys.argv[2])
chunk_size = int(sys.argv[3])
win_size = int(sys.argv[4])
save_file = pot_str + \
    '_win{}_all_chunk{:03d}'.format(win_size, start // chunk_size)

# simulation parameters
sim_dt = .1
sampling_interval = 10.  # samples are taken every tau_refrac [ms]
n_samples = 10
burn_in = 3 * sampling_interval
seed = 7741092

mixing_tso_params = {
    "U": .01,
    "tau_rec": 280.,
    "tau_fac": 0.
}

# load stuff
img_shape = (36, 48)
n_pixels = np.prod(img_shape)
data_name = pot_str + '_var_start{}x{}'.format(*img_shape)
_, _, test_set = load_images(data_name)
rbm = load_rbm(data_name + '_crbm')
w_rbm = rbm.w
b = np.concatenate((rbm.vbias, rbm.hbias))
nv, nh = rbm.n_visible, rbm.n_hidden

# Bring weights and biases into right form
w = np.concatenate((np.concatenate((np.zeros((nv, nv)), w_rbm), axis=1),
                   np.concatenate((w_rbm.T, np.zeros((nh, nh))), axis=1)),
                   axis=0)

calib_file = 'dodo_calib.json'
bm = lifsampl.initialise_network('calibrations/' + calib_file, w, b)

# clamp sliding window
clamp_duration = n_samples * sampling_interval + burn_in
clamp_fct = lifsampl.Clamp_window(clamp_duration, np.zeros(img_shape),
                                  win_size)
duration = clamp_duration * (img_shape[1] + 1)

# ====== actual simulation ======
# setup simulation with seed
sim_setup_kwargs = {
    'rng_seeds_seed': seed
}
lifsampl.setup_simulation(sim_dt, sim_setup_kwargs)
# connect pyNN neurons
lifsampl.make_network_connections(bm, duration, burn_in_time=burn_in,
                                  tso_params=mixing_tso_params)

# run simulations for each image in the chunk
# store samples as bools to save disk space
end = min(start + chunk_size, len(test_set[0]))
samples = np.zeros((end - start, int(duration/sampling_interval), nv + nh)
                   ).astype(bool)
for i, test_img in enumerate(test_set[0][start:end]):
    clamp_fct.clamp_img = test_img.reshape(img_shape)
    samples[i] = lifsampl.simulate_network(
        bm, duration, dt=sim_dt, burn_in_time=burn_in, clamp_fct=clamp_fct)
lifsampl.end_simulation()

np.savez_compressed(make_data_folder() + save_file,
                    samples=samples,
                    data_idx=np.arange(start, end),
                    win_size=win_size,
                    samples_per_frame=int(clamp_duration/sampling_interval)
                    )

# # ======== testing ========
# # Load rbm and data
# # minimal rbm example for debugging
# nv = 4
# nh = 2
# dim = nv + nh
# w_rbm = .5*np.random.randn(nv, nh)
# b = np.zeros(dim)
# w = np.concatenate((np.concatenate((np.zeros((nv, nv)), w_rbm), axis=1),
#                     np.concatenate((w_rbm.T, np.zeros((nh, nh))), axis=1)),
#                    axis=0)
# save_file = 'toyrbm_samples'

# test_img = np.random.rand(nv)
# # fixed clamped image part
# clamped_mask = np.ones(nv)
# clamped_mask = clamped_mask.flatten()
# clamped_idx = np.nonzero(clamped_mask == 1)[0]
# refresh_times = np.array([0])
# clamped_val = np.random.rand(nv)
# clamp_fct = Clamp_anything(refresh_times, clamped_idx, clamped_val)

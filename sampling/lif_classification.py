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
          ' pong/gauss, start_idx, chunk_size, n_samples')
    sys.exit()

pot_str = sys.argv[1]
start = int(sys.argv[2])
chunk_size = int(sys.argv[3])
n_samples = int(sys.argv[4])
save_file = pot_str + '_classif_{}samples'.format(n_samples)

# simulation parameters
sim_dt = .1
sampling_interval = 10.  # samples are taken every tau_refrac [ms]
burn_in = 10
seed = 7741092

calib_file = 'dodo_calib.json'
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
nv, nh, nl = rbm.n_visible, rbm.n_hidden, rbm.n_labels

# Bring weights and biases into right form
w = np.concatenate((np.concatenate((np.zeros((nv, nv)), w_rbm), axis=1),
                   np.concatenate((w_rbm.T, np.zeros((nh, nh))), axis=1)),
                   axis=0)

bm = lifsampl.initialise_network('calibrations/' + calib_file, w, b)

# clamp all but labels
clamped_mask = np.ones(img_shape)
clamped_mask = clamped_mask.flatten()
clamped_idx = np.nonzero(clamped_mask == 1)[0]
refresh_times = np.array([0])
clamped_val = np.random.rand(len(clamped_idx))
clamp_fct = lifsampl.Clamp_anything(refresh_times, clamped_idx, clamped_val)
duration = n_samples * sampling_interval
burn_in_time = burn_in * sampling_interval

# ====== actual simulation ======
# setup simulation with seed
sim_setup_kwargs = {
    'rng_seeds_seed': seed
}
lifsampl.setup_simulation(sim_dt, sim_setup_kwargs)
# connect pyNN neurons
lifsampl.make_network_connections(bm, duration, burn_in_time=burn_in,
                                  tso_params=None)

# run simulations for each image in the chunk
# store samples as bools to save disk space
end = min(start + chunk_size, len(test_set[0]))
samples = np.zeros((end - start, n_samples, nv + nh)).astype(bool)

test_data = test_set[0][start:end]
test_targets = np.argmax(test_set[1][start:end], axis=1)
for i, test_img in enumerate(test_data):
    clamp_fct.set_clamped_val(test_img[clamped_idx])
    samples[i] = lifsampl.simulate_network(
        bm, duration, dt=sim_dt, burn_in_time=burn_in_time,
        clamp_fct=clamp_fct)
lifsampl.end_simulation()

np.savez_compressed(make_data_folder() + save_file,
                    samples=samples,
                    data_idx=np.arange(start, end),
                    n_samples=n_samples
                    )

# compute classification rate
labels = np.argmax(samples.sum(axis=1), axis=1)
print('Correct predictions: {}'.format((labels == test_targets).mean()))

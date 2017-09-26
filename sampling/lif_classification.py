from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import lif_clamped_sampling as lifsampl
from utils.data_mgmt import make_data_folder, load_images, load_rbm
from rbm import RBM, CRBM

# clamped sampling: Pong
if len(sys.argv) != 4:
    print('Please specify the arguments: '
          'pong/gauss/mnist, start_idx, chunk_size')
    sys.exit()

pot_str = sys.argv[1]
start = int(sys.argv[2])
chunk_size = int(sys.argv[3])

# simulation parameters
sim_dt = .1
sampling_interval = 10.  # samples are taken every tau_refrac [ms]
burn_in = 10
n_samples = 20
seed = 7741092
save_file = pot_str + '_classif_{}samples'.format(n_samples)

calib_file = 'dodo_calib.json'
mixing_tso_params = {
    "U": .01,
    "tau_rec": 280.,
    "tau_fac": 0.
}

# load stuff
if pot_str == 'mnist':
    import gzip
    img_shape = (28, 28)
    with gzip.open('../shared_data/datasets/mnist.pkl.gz', 'rb') as f:
        _, _, test_set = np.load(f)
    rbm = load_rbm('mnist_disc_rbm')
else:
    img_shape = (36, 48)
    data_name = pot_str + '_var_start{}x{}'.format(*img_shape)
    _, _, test_set = load_images(data_name)
    rbm = load_rbm(data_name + '_crbm')
n_pixels = np.prod(img_shape)
nv, nh, nl = rbm.n_visible, rbm.n_hidden, rbm.n_labels

# Bring weights and biases into right form
w, b = rbm.bm_params()
bm = lifsampl.initialise_network('calibrations/' + calib_file, w, b,
                                 tso_params=mixing_tso_params)

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

# run simulations for each image in the chunk
# store samples as bools to save disk space
end = min(start + chunk_size, len(test_set[0]))
samples = np.zeros((end - start, n_samples, nv + nh)).astype(bool)

test_data = test_set[0][start:end]
if pot_str == 'mnist':
    test_targets = test_set[1][start:end]
else:
    test_targets = np.argmax(test_set[1][start:end], axis=1)

for i, test_img in enumerate(test_data):
    clamp_fct.set_clamped_val(test_img[clamped_idx])
    bm.spike_data = lifsampl.gather_network_spikes_clamped(
        bm, duration, dt=sim_dt, burn_in_time=burn_in_time,
        sim_setup_kwargs=sim_setup_kwargs, clamp_fct=clamp_fct)
    samples[i] = bm.get_sample_states(sampling_interval)

# I would like to use the reset mechanism so that I don't have to setup the
# network over and over again but somehow it doesn't work (clamping is not changed)
# lifsampl.setup_simulation(sim_dt, sim_setup_kwargs)
# # connect pyNN neurons
# lifsampl.make_network_connections(bm, duration, burn_in_time=burn_in)
# lifsampl.simulate_network(bm, duration, dt=sim_dt, reset=(i != 0),
#                               burn_in_time=burn_in_time, clamp_fct=clamp_fct)
#     samples[i] = bm.get_sample_states(sampling_interval)
# lifsampl.end_simulation()

np.savez_compressed(make_data_folder() + save_file,
                    samples=samples,
                    data_idx=np.arange(start, end),
                    n_samples=n_samples
                    )

# compute classification rate
labels = np.argmax(samples.sum(axis=1), axis=1)
print('Correct predictions: {}'.format((labels == test_targets).mean()))

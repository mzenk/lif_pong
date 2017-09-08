# LIF-sampling experiments
from __future__ import division
from __future__ import print_function
import numpy as np
import lif_clamped_sampling as lifsamp
from utils.data_mgmt import make_data_folder, load_images, load_rbm
from rbm import RBM, CRBM

n_samples = 1000
burnin = 50
sampling_interval = 10.  # == tau_refrac [ms]

calib_file = 'calibrations/dodo_calib.json'
mixing_tso_params = {
    "U": .01,
    "tau_rec": 280.,
    "tau_fac": 0.
}

# ------ Load data -----
# # Pong
# img_shape = (36, 48)
# n_pixels = np.prod(img_shape)
# data_name = 'pong_var_start{}x{}'.format(*img_shape)
# _, _, test_set = load_images(data_name)
# rbm = load_rbm(data_name + '_crbm')
# save_file = data_name.split('_')[0] + '_samples'
# -- OR --
# MNIST
import gzip
img_shape = (28, 28)
n_pixels = np.prod(img_shape)
with gzip.open('../shared_data/datasets/mnist.pkl.gz', 'rb') as f:
    _, _, test_set = np.load(f)
rbm = load_rbm('mnist_disc_rbm')
save_file = 'mnist_samples'

w_rbm = rbm.w
b = np.concatenate((rbm.vbias, rbm.hbias))
nv, nh = rbm.n_visible, rbm.n_hidden

# ------

# Bring weights and biases into right form
w = np.concatenate((np.concatenate((np.zeros((nv, nv)), w_rbm), axis=1),
                   np.concatenate((w_rbm.T, np.zeros((nh, nh))), axis=1)),
                   axis=0)

bm = lifsamp.initialise_network(calib_file, w, b)

duration = n_samples * sampling_interval
burnin_time = burnin * sampling_interval
samples = np.zeros((1, n_samples, nv + nh)).astype(bool)

samples[0] = lifsamp.sample_network(
    bm, duration, dt=.1, burn_in=burnin_time, tso_params='default',
    seed=7741092)

np.savez_compressed(make_data_folder() + save_file,
                    samples=samples,
                    data_idx=np.arange(1),
                    n_samples=n_samples
                    )

# LIF-sampling experiments
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import lif_clamped_sampling as lifsamp
from lif_pong.utils.data_mgmt import make_data_folder, load_images, get_rbm_dict
import lif_pong.training.rbm as rbm_pkg


def dreaming_expt(rbm_name, n_samples, calib_file, burn_in_time=500.,
                  tso_params=None, seed=42, save_name=None):
    rbm = rbm_pkg.load(get_rbm_dict(rbm_name))
    w, b = rbm.bm_params()
    sampling_interval = 10.  # == tau_refrac [ms]
    duration = n_samples * sampling_interval

    # setup simulation with seed
    sim_setup_kwargs = {
        'rng_seeds_seed': seed,
        'spike_precision': 'on_grid'
    }
    samples = lifsamp.sample_network(calib_file, w, b, duration,
                                     tso_params=tso_params,
                                     burn_in_time=burn_in_time,
                                     sim_setup_kwargs=sim_setup_kwargs)
    if save_name is None:
        save_name = rbm_name
    # to view with the inspection script, samples must have shape (N, npxls)
    np.savez_compressed(os.path.join(make_data_folder(), save_name),
                        samples=np.expand_dims(samples, 0))
    return samples


if __name__ == '__main__':
    rbm_name = 'pong_var_start36x48_crbm'
    n_samples = int(1e4)
    burn_in_time = 0.
    sampling_interval = 10.  # == tau_refrac [ms]=
    calib_file = 'calibrations/mihai_calib.json'
    mixing_tso_params = {
        "U": .01,
        "tau_rec": 280.,
        "tau_fac": 0.
    }
    dreaming_expt(rbm_name, n_samples, calib_file, burn_in_time,
                  tso_params=None, save_name='mihai')

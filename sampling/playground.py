#! /usr/bin/env python
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import lif_clamped_sampling as lifsampl
from lif_pong.utils.data_mgmt import load_images, get_rbm_dict, make_data_folder
import lif_pong.training.rbm as rbm_pkg


def lif_clamp_pattern(n_samples, test_imgs, img_shape, rbm, calib_file,
                      sampling_interval=10.):
    # Bring weights and biases into right form
    w, b = rbm.bm_params()
    duration = n_samples * sampling_interval
    refresh_times = [0]
    clamp_mask = np.ones(img_shape).reshape(img_shape)
    clamp_mask[:, int(.2*img_shape[1]):] = 0
    clamp_idx = np.where(clamp_mask.flatten() == 1)[0]
    refresh_times = [0.]

    bm = lifsampl.initialise_network(calib_file, w, b)
    results = []

    for img in test_imgs:
        clamp_fct = lifsampl.Clamp_anything(refresh_times, [clamp_idx],
                                            [img[clamp_idx]])
        bm.spike_data = lifsampl.gather_network_spikes_clamped(
            bm, duration, clamp_fct=clamp_fct, burn_in_time=burn_in_time)
        if bm.spike_data is None:
            return None
        results.append(bm.get_sample_states(sampling_interval))
    return np.array(results)



if __name__ == '__main__':
    # load data
    data_name = 'pong_var_start36x48'
    rbm_name = 'pong_var_start36x48_crbm'
    calib_file = 'calibrations/wei_curr_calib.json'
    save_name = 'wei'

    n_samples = 1e3
    burn_in_time = 0.
    sampling_interval = 10.  # == tau_refrac [ms]=
    img_shape = (36, 48)
    _, _, test_set = load_images(data_name)
    test_imgs = test_set[0][:5]
    rbm = rbm_pkg.load(get_rbm_dict(rbm_name))
    samples = lif_clamp_pattern(n_samples, test_imgs, img_shape, rbm,
                                calib_file)
    np.savez_compressed(os.path.join(make_data_folder(), save_name),
                        samples=samples)

#! /usr/bin/env python
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import lif_clamped_sampling as lifsampl
from lif_pong.utils.data_mgmt import load_images, get_rbm_dict, make_data_folder
import lif_pong.training.rbm as rbm_pkg
from lif_pong.utils import average_helper
import matplotlib.pyplot as plt


def lif_clamp_pattern(n_samples, test_imgs, img_shape, rbm, calib_file,
                      sampling_interval=10., burn_in_time=100., synclamp=False):
    # Bring weights and biases into right form
    w, b = rbm.bm_params()
    duration = n_samples * sampling_interval
    refresh_times = [.75*duration, .85*duration]
    clamp_mask = np.ones(img_shape).reshape(img_shape)
    clamp_mask[:, int(.5*img_shape[1]):] = 0
    clamp_idx = np.where(clamp_mask.flatten() == 1)[0]
    clamp_dict = {
      'n_pixels': np.prod(img_shape),
      'spike_interval': 1.,
      'bias_shift': 8,
      'tso_params': {'U': 1., 'tau_rec': 10., 'tau_fac': 0.}
    }

    renewing_params = {'U': 1., 'tau_rec': 10., 'tau_fac': 0}
    bm = lifsampl.initialise_network(calib_file, w, b, tso_params=renewing_params)
    results = []

    sim_setup_kwargs = {
        # choose different seed for each simulation
        'rng_seeds_seed': 479233,
    }

    for img in test_imgs:
        sim_setup_kwargs['rng_seeds_seed'] += 1
        clamp_fct = lifsampl.Clamp_anything(refresh_times, [[], clamp_idx],
                                            [[], img[clamp_idx]])
        if not synclamp:
            bm.spike_data = lifsampl.gather_network_spikes_clamped(
                bm, duration, clamp_fct=clamp_fct, burn_in_time=burn_in_time,
                sim_setup_kwargs=sim_setup_kwargs)
        else:
            bm.spike_data = lifsampl.gather_network_spikes_clamped_sf(
                bm, duration, clamp_fct=clamp_fct, burn_in_time=burn_in_time,
                clamp_dict=clamp_dict, sim_setup_kwargs=sim_setup_kwargs)
        if bm.spike_data is None:
            return None
        results.append(bm.get_sample_states(sampling_interval))
    return np.array(results)


def lif_clamp_window(n_samples, test_imgs, img_shape, rbm, calib_file,
                     sampling_interval=10., burn_in_time=100., offset=0.,
                     synclamp=False):
    # Bring weights and biases into right form
    w, b = rbm.bm_params()
    duration = n_samples * sampling_interval
    offset_time = offset * sampling_interval
    clamp_interval = 10 * sampling_interval
    clamp_dict = {
      'n_pixels': np.prod(img_shape),
      'spike_interval': 1.,
      'bias_shift': 8,
      'tso_params': {'U': 1., 'tau_rec': 10., 'tau_fac': 0.}
    }

    renewing_params = {'U': 1., 'tau_rec': 100., 'tau_fac': 0}
    bm = lifsampl.initialise_network(calib_file, w, b, tso_params=renewing_params)
    results = []

    sim_setup_kwargs = {
        # choose different seed for each simulation
        'rng_seeds_seed': 479233,
    }

    for img in test_imgs:
        sim_setup_kwargs['rng_seeds_seed'] += 1
        clamp_fct = lifsampl.Clamp_window(
            clamp_interval, img.reshape(img_shape), offset=offset_time)
        if not synclamp:
            bm.spike_data = lifsampl.gather_network_spikes_clamped(
                bm, duration, clamp_fct=clamp_fct, burn_in_time=burn_in_time,
                sim_setup_kwargs=sim_setup_kwargs)
        else:
            bm.spike_data = lifsampl.gather_network_spikes_clamped_sf(
                bm, duration, clamp_fct=clamp_fct, burn_in_time=burn_in_time,
                clamp_dict=clamp_dict, sim_setup_kwargs=sim_setup_kwargs)
        if bm.spike_data is None:
            return None
        results.append(bm.get_sample_states(sampling_interval))
    return np.array(results)


def gibbs_classification_check(rbm, data_set, img_shape):
    n_labels = data_set[1].shape[1]
    pred_labprobs = rbm.classify(data_set[0], class_prob=True)
    pred_labels = np.argmax(pred_labprobs, axis=1)
    pred_labpos = average_helper(n_labels, pred_labprobs)
    target_labels = np.argmax(data_set[1], axis=1)
    target_labpos = average_helper(n_labels, data_set[1])
    last_col = data_set[0].reshape((-1,) + img_shape)[..., -1]
    target_pos = average_helper(img_shape[0], last_col)
    target_pxl = np.argmax(last_col, axis=1)

    # make a 2d histogram
    xedges = np.arange(img_shape[0] + 1)
    yedges = np.arange(n_labels  + 1)
    H, xedges, yedges = np.histogram2d(target_pxl, pred_labels, bins=(xedges, yedges))
    # # grid orientation of pcolor, meshgrid
    H = H.T
    # X, Y = np.meshgrid(xedges, yedges)
    # plt.pcolor(X, Y, H, cmap='gray')
    wrong_cases = 0
    labwidth = int(img_shape[0]/n_labels)
    for i, row in enumerate(H):
        wrong_cases += row.sum() - row[i*labwidth: (i+1)*labwidth].sum()

    n_reflected = np.all(last_col == 0, axis=1).sum()
    n_pos0 = H[:, 0].sum()
    print(n_reflected, n_pos0)

    print('Wrong: {:.1f}%'.format(100*wrong_cases/H.sum()))

    plt.figure(figsize=(10, 7))
    plt.imshow(H, cmap='gray', interpolation='Nearest', origin='lower')
    plt.colorbar(orientation='horizontal')
    plt.xlabel('brightest pixel of last column')
    plt.ylabel('best label')
    plt.savefig('label_histo')


if __name__ == '__main__':
    calib_file = 'calibrations/wei_curr_calib.json'
    # load data
    data_name = 'pong_lw5_40x48'
    rbm_name = 'pong_lw5_40x48_crbm'
    img_shape = (40, 48)
    _, _, test_set = load_images(data_name)
    rbm = rbm_pkg.load(get_rbm_dict(rbm_name))
    save_name = 'test'

    # # gibbs_classification_check(rbm, test_set, img_shape)
    # samples = lif_clamp_pattern(500, test_set[0][:2], img_shape, rbm, calib_file,
    #                             burn_in_time=0.)
    # samples = lif_clamp_window(750, test_set[0][:2], img_shape, rbm, calib_file,
    #                            burn_in_time=0., offset=500, synclamp=True)
    # np.savez_compressed(os.path.join(make_data_folder(), 'window_sf'), samples=samples)
    # samples = lif_clamp_window(750, test_set[0][:2], img_shape, rbm, calib_file,
    #                            burn_in_time=0., offset=500, synclamp=False)
    # np.savez_compressed(os.path.join(make_data_folder(), 'window'), samples=samples)

    # toy rbm
    nv = 7
    rbm = rbm_pkg.RBM(nv, 5)
    w, b = rbm.bm_params()
    duration = 1000.
    test_set = np.random.randint(2, size=(2, nv))
    save_name = 'toytest'

    renewing_params = {'U': 1., 'tau_rec': 10., 'tau_fac': 0}
    samples = lifsampl.sample_network(calib_file, w, b, duration,
     tso_params=renewing_params, burn_in_time=0.)

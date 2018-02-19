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
                      sampling_interval=10., burn_in_time=100.):
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
    # load data
    data_name = 'gauss_lw5_40x48'
    rbm_name = 'gauss_lw5_40x48_crbm'
    img_shape = (40, 48)
    # data_name = 'pong_var_start36x48'
    # rbm_name = 'pong_var_start36x48_crbm'
    # img_shape = (36, 48)
    calib_file = 'calibrations/wei_curr_calib.json'
    save_name = 'test'
    
    _, _, test_set = load_images(data_name)
    rbm = rbm_pkg.load(get_rbm_dict(rbm_name))

    gibbs_classification_check(rbm, test_set, img_shape)

#! /usr/bin/env python
# script for drawing gibbs samples from an RBM with dynamic clamping
from __future__ import division
from __future__ import print_function
import numpy as np
import yaml
import sys
import expt_analysis as analysis
import lif_pong.training.rbm as rbm_pkg
from lif_pong.utils.data_mgmt import load_images, get_rbm_dict
from gibbs_clamped_sampling import Clamp_window, run_simulation


def main(data_set, rbm, general_dict, analysis_dict):
    # pass arguments from dictionaries to simulation
    n_samples = general_dict['n_samples']
    img_shape = tuple(general_dict['img_shape'])
    start = general_dict['start_idx']
    end = start + general_dict['chunksize']
    winsize = general_dict['winsize']
    gather_data = general_dict['gather_data']
    # take different seed for each chunk so that they are not too similar initially
    rbm.set_seed(general_dict['seed'] + start)

    if 'binary' in general_dict.keys():
        binary = general_dict['binary']
    else:
        binary = False

    # mainly for testing
    if 'continuous' in general_dict.keys():
        continuous = general_dict['continuous']
    else:
        continuous = True

    if 'bin_imgs' in general_dict.keys():
        bin_imgs = general_dict['bin_imgs']
    else:
        bin_imgs = False

    duration = (img_shape[1] + 1) * n_samples
    clamp = Clamp_window(img_shape, n_samples, winsize)

    try:
        with np.load('samples.npz') as d:
            samples = d['samples']
    except Exception:
        if gather_data:
            print('Running gibbs simulation for instances {} to {}'
                  ''.format(start, end))
            clamped_imgs = data_set[0][start:end]
            # clamped_imgs = (data_set[0][start:end] > .5)*1.
            vis_samples, hid_samples = run_simulation(
                rbm, duration, clamped_imgs, binary=binary,
                burnin=general_dict['burn_in'], clamp_fct=clamp,
                continuous=continuous, bin_imgs=bin_imgs)

            # hidden samples are only saved in binary format due to file size
            if binary:
                samples = np.concatenate((vis_samples, hid_samples), axis=2)
            else:
                samples = vis_samples
            # compared to the lif-methods, the method returns an array with
            # shape [n_steps, n_imgs, n_vis]. Hence, swap axes.
            samples = np.swapaxes(samples, 0, 1)
            if binary:
                np.savez_compressed('samples', samples=samples > .5)
            else:
                np.savez_compressed('samples', samples=samples)
        else:
            print('Missing sample file', file=sys.stderr)
            samples = None
    # produce analysis file
    return analysis.inf_speed_analysis(samples, data_set=data_set,
                                       **analysis_dict)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Wrong number of arguments. Please provide a yaml-config file.')
        sys.exit()
    with open(sys.argv[1]) as configfile:
        config = yaml.load(configfile)

    general_dict = config.pop('general')
    _, _, test_set = load_images(general_dict['data_name'])
    rbm = rbm_pkg.load(get_rbm_dict(general_dict['rbm_name']))
    try:
        analysis_dict = config.pop('analysis')
    except KeyError:
        analysis_dict = {}

    main(test_set, rbm, general_dict, analysis_dict)

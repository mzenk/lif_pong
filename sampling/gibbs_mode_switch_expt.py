#! /usr/bin/env python
# script for drawing gibbs samples from an RBM with dynamic clamping
from __future__ import division
from __future__ import print_function
import numpy as np
import yaml
import sys
import lif_pong.training.rbm as rbm_pkg
from lif_pong.utils import get_windowed_image_index
from lif_pong.utils.data_mgmt import get_rbm_dict, load_images
from gibbs_clamped_sampling import run_simulation, Clamp_anything
from expt_analysis import mode_switch_analysis


def main(data_set, rbm, general_dict, analysis_dict):
    # pass arguments from dictionaries to simulation
    n_samples = general_dict['n_samples']
    img_shape = tuple(general_dict['img_shape'])
    start = general_dict['start_idx']
    end = start + general_dict['chunksize']
    gather_data = general_dict['gather_data']
    rbm.set_seed(general_dict['seed'] + start)

    if 'init_time' in general_dict.keys():
        init_time = general_dict['init_time']
    else:
        init_time = 10

    if 'binary' in general_dict.keys():
        binary = general_dict['binary']
    else:
        binary = False

    if 'bin_imgs' in general_dict.keys():
        bin_imgs = general_dict['bin_imgs']
    else:
        bin_imgs = False

    try:
        with np.load('samples.npz') as d:
            samples = d['samples']
    except Exception:
        if gather_data:
            # clamp whole picture for short time
            refresh_times = [0.]
            clamp_idx = [np.arange(np.prod(img_shape))]
            clamp = Clamp_anything(refresh_times, clamp_idx)
            # just take neighbouring image in the set (is randomly shuffled)
            # as initialisation
            clamped_imgs = data_set[0][(start + 1) % len(data_set[0]):
                                       (end + 1) % len(data_set[0])]
            tmp_vis, tmp_hid = run_simulation(
                rbm, init_time, clamped_imgs, binary=binary,
                burnin=general_dict['burn_in'], clamp_fct=clamp,
                bin_imgs=bin_imgs)

            # clamp part of other picture for long time
            refresh_times = [0.]
            clamp_idx = [get_windowed_image_index(
                img_shape, general_dict['fraction'], fractional=True)]
            clamp = Clamp_anything(refresh_times, clamp_idx)
            clamped_imgs = data_set[0][start:end]
            vis_samples, hid_samples = run_simulation(
                rbm, n_samples, clamped_imgs, binary=binary,
                burnin=0., clamp_fct=clamp,
                bin_imgs=bin_imgs, v_init=tmp_vis[-1])

            vis_samples = np.concatenate((tmp_vis, vis_samples), axis=0)
            hid_samples = np.concatenate((tmp_hid, hid_samples), axis=0)

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
    mode_switch_analysis()


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

#! /usr/bin/env python
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import yaml
import lif_clamped_sampling as lifsampl
import lif_chunk_analysis as analysis
from lif_pong.utils.data_mgmt import load_images, get_rbm_dict
import lif_pong.training.rbm as rbm_pkg


def lif_window_expt(win_size, test_imgs, img_shape, rbm, sbs_kwargs,
                    n_samples=20):
    # Bring weights and biases into right form
    w, b = rbm.bm_params()
    sampling_interval = sbs_kwargs.pop('sampling_interval')
    # clamp sliding window
    clamp_duration = n_samples * sampling_interval
    duration = clamp_duration * (img_shape[1] + 1)

    bm = lifsampl.initialise_network(
        sbs_kwargs.pop('calib_file'), w, b,
        tso_params=sbs_kwargs.pop('tso_params'),
        weight_scaling=sbs_kwargs.pop('weight_scaling'))
    results = []

    for img in test_imgs:
        clamp_fct = lifsampl.Clamp_window(
            clamp_duration, img.reshape(img_shape), win_size)
        bm.spike_data = lifsampl.gather_network_spikes_clamped(
            bm, duration, clamp_fct=clamp_fct, **sbs_kwargs)
        if bm.spike_data is None:
            return None
        results.append(bm.get_sample_states(sampling_interval))
    return np.array(results)


def main(general_dict, sbs_dict, analysis_dict):
    # pass arguments from dictionaries to simulation
    gather_data = general_dict['gather_data']
    n_samples = general_dict['n_samples']
    img_shape = tuple(general_dict['img_shape'])
    # load data
    _, _, test_set = load_images(general_dict['data_name'])
    rbm = rbm_pkg.load(get_rbm_dict(general_dict['rbm_name']))
    start = general_dict['start_idx']
    end = start + general_dict['chunksize']
    winsize = general_dict['winsize']

    sim_setup_kwargs = {
        'rng_seeds_seed': sbs_dict['seed'],
        'threads': general_dict['threads']
    }

    sbs_kwargs = sbs_dict
    del sbs_kwargs['seed']
    sbs_kwargs['sim_setup_kwargs'] = sim_setup_kwargs

    try:
        with np.load('samples.npz') as d:
            samples = d['samples'].astype(float)
    except Exception:
        if gather_data:
            samples = lif_window_expt(
                winsize, test_set[0][start:end], img_shape, rbm, sbs_kwargs,
                n_samples=n_samples)
            if samples is not None:
                np.savez_compressed('samples', samples=samples.astype(bool))
        else:
            print('Missing sample file', file=sys.stderr)
            samples = None
    # produce analysis file
    analysis.inf_speed_analysis(samples.astype(float), **analysis_dict)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Wrong number of arguments. Please provide a yaml-config file.')
        sys.exit()
    with open(sys.argv[1]) as configfile:
        config = yaml.load(configfile)

    general_dict = config.pop('general')
    sbs_dict = config.pop('sbs')
    try:
        analysis_dict = config.pop('analysis')
    except KeyError:
        analysis_dict = {}

    main(general_dict, sbs_dict, analysis_dict)

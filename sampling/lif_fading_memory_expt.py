#! /usr/bin/env python
from __future__ import division
from __future__ import print_function
import sys
import yaml
import numpy as np
import lif_clamped_sampling as lifsampl
import expt_analysis as analysis
from lif_pong.utils.data_mgmt import load_images, get_rbm_dict
import lif_pong.training.rbm as rbm_pkg


def lif_tso_clamping_expt(test_imgs, img_shape, rbm, sbs_kwargs,
                          clamp_dict, n_samples=20, winsize=None):
    # Bring weights and biases into right form
    w, b = rbm.bm_params()
    sampling_interval = sbs_kwargs.pop('sampling_interval')
    # clamp using TSO
    clamp_duration = n_samples * sampling_interval
    duration = clamp_duration * (img_shape[1] + 1)

    bm = lifsampl.initialise_network(
        sbs_kwargs.pop('calib_file'), w, b,
        tso_params=sbs_kwargs.pop('tso_params'))

    # add all necessary kwargs to one dictionary
    kwargs = {k: sbs_kwargs[k] for k in sbs_kwargs.keys()}
    clamp_dict['n_pixels'] = rbm.n_inputs

    results = []
    for i, img in enumerate(test_imgs):
        # choose different seed for each simulation
        sbs_kwargs['sim_setup_kwargs']['rng_seeds_seed'] += 100
        kwargs['clamp_fct'] = \
            lifsampl.Clamp_window(clamp_duration, img.reshape(img_shape),
                                  win_size=winsize)
        bm.spike_data = lifsampl.gather_network_spikes_clamped_sf(
            bm, duration, clamp_dict=clamp_dict, **kwargs)
        results.append(bm.get_sample_states(sampling_interval))
    return np.array(results)


def main(data_set, rbm, general_dict, sbs_dict, clamp_dict, analysis_dict):
    # pass arguments from dictionaries to simulation
    gather_data = general_dict['gather_data']
    n_samples = general_dict['n_samples']
    img_shape = tuple(general_dict['img_shape'])
    start = general_dict['start_idx']
    end = start + general_dict['chunksize']

    if 'winsize' in clamp_dict.keys():
        winsize = clamp_dict.pop('winsize')
    else:
        winsize = None

    sim_setup_kwargs = {
        # choose different seed for each simulation
        'rng_seeds_seed': sbs_dict['seed'] + start,
        'threads': general_dict['threads']
    }

    sbs_kwargs = sbs_dict
    del sbs_kwargs['seed']
    sbs_kwargs['sim_setup_kwargs'] = sim_setup_kwargs

    try:
        with np.load('samples.npz') as d:
            samples = d['samples']
    except Exception:
        # ensures that only experiments that didn't produce data are repeated
        if gather_data:
            samples = lif_tso_clamping_expt(
                data_set[0][start:end], img_shape, rbm, sbs_kwargs,
                clamp_dict, n_samples=n_samples, winsize=winsize)
            if samples is not None:
                np.savez_compressed('samples', samples=samples.astype(bool))
        else:
            print('Missing sample file', file=sys.stderr)
            samples = None

    # also possible: perform analysis on chunk right here
    # analysis can be replaced
    analysis.inf_speed_analysis(samples, data_set=data_set, **analysis_dict)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Wrong number of arguments. Please provide a yaml-config file.')
        sys.exit()
    with open(sys.argv[1]) as configfile:
        config = yaml.load(configfile)

    general_dict = config.pop('general')
    sbs_dict = config.pop('sbs')
    clamp_dict = config.pop('clamping')
    try:
        analysis_dict = config.pop('analysis')
    except KeyError:
        analysis_dict = {}

    # load data
    _, _, test_set = load_images(general_dict['data_name'])
    rbm = rbm_pkg.load(get_rbm_dict(general_dict['rbm_name']))

    main(test_set, rbm, general_dict, sbs_dict, clamp_dict, analysis_dict)

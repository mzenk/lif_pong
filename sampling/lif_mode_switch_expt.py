#! /usr/bin/env python
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import yaml
import lif_clamped_sampling as lifsampl
from expt_analysis import mode_switch_analysis
from lif_pong.utils.data_mgmt import load_images, get_rbm_dict
from lif_pong.utils import get_windowed_image_index
import lif_pong.training.rbm as rbm_pkg


def mode_switch_expt(n_samples, imgs, img_shape, rbm, fraction, start, end,
                     sbs_kwargs, n_init=50):
    # Bring weights and biases into right form
    w, b = rbm.bm_params()
    sampling_interval = sbs_kwargs.pop('sampling_interval')

    duration = sampling_interval * (n_samples + n_init)

    bm = lifsampl.initialise_network(
        sbs_kwargs.pop('calib_file'), w, b,
        tso_params=sbs_kwargs.pop('tso_params'),
        weight_scaling=sbs_kwargs.pop('weight_scaling'))
    results = []

    for i in range(start, end):
        # clamp first whole trajectory for a short time, then parts for a while
        refresh_times = [0., n_init*sampling_interval]
        clamp_idx = [np.arange(np.prod(img_shape)), get_windowed_image_index(
            img_shape, fraction, fractional=True)]
        clamp_val = [imgs[i, clamp_idx[0]],
                     imgs[(i + 1) % len(imgs), clamp_idx[1]]]
        clamp_fct = lifsampl.Clamp_anything(refresh_times, clamp_idx, clamp_val)
        bm.spike_data = lifsampl.gather_network_spikes_clamped(
            bm, duration, clamp_fct=clamp_fct, **sbs_kwargs)
        if bm.spike_data is None:
            return None
        results.append(bm.get_sample_states(sampling_interval))
    return np.array(results)


def main(data_set, rbm, general_dict, sbs_dict, analysis_dict):
    # pass arguments from dictionaries to simulation
    gather_data = general_dict['gather_data']
    n_samples = general_dict['n_samples']
    img_shape = tuple(general_dict['img_shape'])
    start = general_dict['start_idx']
    end = start + general_dict['chunksize']
    fraction = general_dict['fraction']

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
            samples = mode_switch_expt(
                n_samples, data_set[0], img_shape, rbm, fraction, start, end,
                sbs_kwargs, n_init=50)
            if samples is not None:
                np.savez_compressed('samples', samples=samples.astype(bool))
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
    sbs_dict = config.pop('sbs')
    try:
        analysis_dict = config.pop('analysis')
    except KeyError:
        analysis_dict = {}

    # load data
    _, _, test_set = load_images(general_dict['data_name'])
    rbm = rbm_pkg.load(get_rbm_dict(general_dict['rbm_name']))
    main(test_set, rbm, general_dict, sbs_dict, analysis_dict)

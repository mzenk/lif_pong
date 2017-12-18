#! /usr/bin/env python
from __future__ import division
from __future__ import print_function
import sys
import yaml
import numpy as np
import lif_clamped_sampling as lifsampl
import lif_chunk_analysis as analysis
from lif_pong.utils.data_mgmt import load_images, get_rbm_dict
import lif_pong.training.rbm as rbm_pkg


def lif_tso_clamping_expt(test_imgs, img_shape, rbm, sbs_kwargs,
                          clamp_kwargs, n_samples=20):
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
    for k in clamp_kwargs.keys():
        kwargs[k] = clamp_kwargs[k]
    results = []
    for img in test_imgs:
        kwargs['clamp_fct'] = \
            lifsampl.Clamp_window(clamp_duration, img.reshape(img_shape))
        bm.spike_data = lifsampl.gather_network_spikes_clamped_sf(
            bm, duration, rbm.n_inputs, **kwargs)
        results.append(bm.get_sample_states(sampling_interval))
    return np.array(results)


def test(test_imgs, img_shape, rbm, sbs_kwargs,
         clamp_kwargs, n_samples=20):
    # Bring weights and biases into right form
    w, b = rbm.bm_params()
    sampling_interval = sbs_kwargs['sampling_interval']
    # clamp using TSO
    duration = n_samples * sampling_interval
    # clamp all but labels
    clamped_mask = np.ones(img_shape)
    clamped_mask = clamped_mask.flatten()
    clamped_idx = np.nonzero(clamped_mask == 1)[0]
    refresh_times = [0.]

    bm = lifsampl.initialise_network(
        sbs_kwargs['calib_file'], w, b, tso_params=sbs_kwargs['tso_params'])

    # add all necessary kwargs to one dictionary
    kwargs = {k: sbs_kwargs[k] for k in
              ('dt', 'sim_setup_kwargs', 'burn_in_time')}
    for k in clamp_kwargs.keys():
        kwargs[k] = clamp_kwargs[k]
    results = []
    for img in test_imgs:
        kwargs['clamp_fct'] = \
            lifsampl.Clamp_anything(refresh_times, clamped_idx, img)
        # bm.spike_data = lifsampl.gather_network_spikes_clamped_bn(
        #     bm, duration, rbm.n_inputs, **kwargs)
        bm.spike_data = lifsampl.gather_network_spikes_clamped_sf(
            bm, duration, rbm.n_inputs, **kwargs)
        results.append(bm.get_sample_states(sampling_interval))
    return np.array(results)


def main(general_dict, sbs_dict, clamp_dict, identifiers):
    # pass arguments from dictionaries to simulation
    gather_data = general_dict['gather_data']
    n_samples = general_dict['n_samples']
    img_shape = tuple(general_dict['img_shape'])
    # load data
    _, _, test_set = load_images(general_dict['data_name'])
    rbm = rbm_pkg.load(get_rbm_dict(general_dict['rbm_name']))
    start = general_dict['start_idx']
    end = start + general_dict['chunksize']

    # It is possible to set the weights of the bias synapses using a
    # calibration file or all equal to the value in tso_params
    if 'calib_file' in clamp_dict.keys():
        wp_fit_params = {}
        with np.load(clamp_dict['calib_file']) as d:
            for k in d.keys():
                wp_fit_params[k] = d[k]
    else:
        wp_fit_params = None

    clamp_kwargs = {
        'clamp_tso_params': clamp_dict['tso_params'],
        'wp_fit_params': wp_fit_params
    }

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
        print('Missing sample file', file=sys.stderr)
        samples = None
        # temporary? ensures that only experiments that didn't produce data are
        # repeated
        if gather_data:
            samples = lif_tso_clamping_expt(
                test_set[0][start:end], img_shape, rbm, sbs_kwargs, clamp_kwargs,
                n_samples=n_samples)

            np.savez_compressed('samples', samples=samples.astype(bool))

    # also possible: perform analysis on chunk right here
    # analysis can be replaced
    analysis.inf_speed_analysis(identifiers, samples)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Wrong number of arguments. Please provide a yaml-config file.')
        sys.exit()
    with open(sys.argv[1]) as configfile:
        config = yaml.load(configfile)

    general_dict = config.pop('general')
    sbs_dict = config.pop('sbs')
    clamp_dict = config.pop('clamping')
    identifiers = config.pop('identifier')

    main(general_dict, sbs_dict, clamp_dict, identifiers)

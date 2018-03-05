#! /usr/bin/env python
# LIF-sampling experiments
from __future__ import division
from __future__ import print_function
import numpy as np
import sys
import yaml
import lif_clamped_sampling as lifsamp
from lif_pong.utils.data_mgmt import make_data_folder, load_images, get_rbm_dict
import lif_pong.training.rbm as rbm_pkg
from expt_analysis import burnin_analysis


def dreaming_expt(rbm_name, n_samples, sbs_kwargs):
    # in sbs dict
    sampling_interval = sbs_kwargs.pop('sampling_interval')
    calib_file = sbs_kwargs.pop('calib_file')
    w, b = rbm.bm_params()
    duration = n_samples * sampling_interval

    samples = lifsamp.sample_network(calib_file, w, b, duration, **sbs_kwargs)

    # to view with the inspection script, samples must have shape (N, npxls)
    np.savez_compressed('samples', samples=np.expand_dims(samples, 0))
    return samples


def main(rbm, general_dict, sbs_dict, analysis_dict):
    n_samples = general_dict['n_samples']
    gather_data = general_dict['gather_data']

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
            samples = dreaming_expt(rbm, n_samples, sbs_kwargs)
        else:
            print('Missing sample file', file=sys.stderr)
            samples = None

    nv = rbm.n_visible
    if 'n_labels' in rbm.__dict__.keys():
        nv -= rbm.n_labels
    result_dict = burnin_analysis(samples[..., :nv].squeeze())
    result_dict['seed'] = sim_setup_kwargs['rng_seeds_seed']
    with open('analysis', 'w') as f:
            f.write(yaml.dump(result_dict))


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

    rbm = rbm_pkg.load(get_rbm_dict(general_dict['rbm_name']))
    main(rbm, general_dict, sbs_dict, analysis_dict)

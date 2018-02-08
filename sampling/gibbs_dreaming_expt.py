#! /usr/bin/env python
# script for drawing gibbs samples from an RBM with dynamic clamping
from __future__ import division
from __future__ import print_function
import numpy as np
import yaml
import sys
import lif_pong.training.rbm as rbm_pkg
from lif_pong.utils.data_mgmt import get_rbm_dict
from expt_analysis import burnin_analysis

def main(rbm, general_dict):
    n_samples = general_dict['n_samples']
    gather_data = general_dict['gather_data']
    rbm.set_seed(general_dict['seed'])

    if 'binary' in general_dict.keys():
        binary = general_dict['binary']
    else:
        binary = False

    try:
        with np.load('samples.npz') as d:
            samples = d['samples'].astype(float)
    except Exception:
        if gather_data:
            samples = rbm.draw_samples(n_samples, binary=binary)
            np.savez_compressed('samples', samples=np.expand_dims(samples, 0))
        else:
            print('Missing sample file', file=sys.stderr)
            samples = None

    # add analysis
    nv = rbm.n_visible
    if 'n_labels' in rbm.__dict__.keys():
        nv -= rbm.n_labels
    result_dict = burnin_analysis(samples[:, :nv])
    result_dict['seed'] = general_dict['seed']
    with open('analysis', 'w') as f:
            f.write(yaml.dump(result_dict))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Wrong number of arguments. Please provide a yaml-config file.')
        sys.exit()
    with open(sys.argv[1]) as configfile:
        config = yaml.load(configfile)

    general_dict = config.pop('general')
    rbm = rbm_pkg.load(get_rbm_dict(general_dict['rbm_name']))

    main(rbm, general_dict)

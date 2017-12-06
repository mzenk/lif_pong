#! /usr/bin/env python
# short decription
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import sys
import yaml
from lif_pong.utils import average_pool
import pong_agent

assert os.path.exists('analysis')

if len(sys.argv) != 4:
    print('Wrong number of arguments.')
    sys.exit()

config_file = sys.argv[1]
u_idx = int(sys.argv[2])
tau_rec_idx = int(sys.argv[3])

with open(config_file) as config:
    experiment_dict = yaml.load(config)

stub_dict = experiment_dict.pop('stub')
replacements = experiment_dict.pop('replacements')
general_dict = stub_dict.pop('general')
start_ids = replacements.pop('start_idx')
U = replacements['U'][u_idx]
tau_rec = replacements['tau_rec'][tau_rec_idx]
n_samples = general_dict['n_samples']
img_shape = tuple(general_dict['img_shape'])
n_labels = img_shape[0]//3
n_pxls = np.prod(img_shape)

pred_file = 'analysis/{}_{}_prediction'.format(U, tau_rec)
agent_file = 'analysis/{}_{}_agent'.format(U, tau_rec)

# make prediction files from samples
lab = 0
last_col = 0
counter = 0
folder_list = os.listdir('.')
folder_list.remove('analysis')
for folder in folder_list:
    with open(folder + '/sim.yaml') as d:
        clamp_dict = yaml.load(d)['clamping']['tso_params']
    # check if simulation has correct parameters
    if clamp_dict['U'] == U and clamp_dict['tau_rec'] == tau_rec:
        print('Processing chunk {} of {}'.format(counter, len(start_ids)))
        try:
            with np.load(folder + '/samples.npz') as d:
                chunk_samples = d['samples'].astype(float)
                chunk_vis = chunk_samples[..., :n_pxls + n_labels]
        except IOError:
            print('File not found: ' + folder + '/samples.npz', file=sys.stderr)
            continue

        # average pool on each chunk, then compute prediction
        chunk_vis = average_pool(chunk_vis, n_samples, n_samples)

        tmp_col = chunk_vis[..., :-n_labels].reshape(
                    chunk_vis.shape[:-1] + img_shape)[..., -1]
        tmp_lab = chunk_vis[..., -n_labels:]

        lab = tmp_lab if counter == 0 else np.vstack((lab, tmp_lab))
        last_col = tmp_col if counter == 0 else np.vstack((last_col, tmp_col))
        counter += 1

if counter > 0:
    print('Saving prediction data of {} chunks'.format(counter))
    # save data (averaged samples for label units and last column)
    np.savez_compressed(pred_file, label=lab, last_col=last_col)

    # compute agent performance
    result_dict = pong_agent.compute_performance(img_shape, general_dict['data_name'],
                                                 pred_file + '.npz')
    np.savez_compressed(agent_file, U=U, tau_rec=tau_rec, **result_dict)

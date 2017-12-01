#! /usr/bin/env python
# short decription
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import sys
import yaml
from utils import average_pool
import pong_agent

assert os.path.exists('analysis')

if len(sys.argv) != 4:
    print('Wrong number of arguments.')
    sys.exit()

config_file = sys.argv[1]
u_idx = sys.argv[2]
tau_rec_idx = sys.argv[3]

with open(config_file) as config:
    experiment_dict = yaml.load(config)

stub_dict = experiment_dict.pop('stub')
replacements = experiment_dict.pop('replacements')

start_ids = replacements['start_idx']
U = replacements['U'][u_idx]
tau_rec = replacements['tau_rec'][tau_rec_idx]
n_samples = stub_dict['general']['n_samples']
img_shape = stub_dict['general']['img_shape']
n_labels = img_shape[0]//3
n_pxls = np.prod(img_shape)

pred_file = 'analysis/{}_{}_prediction'.format(U, tau_rec)
agent_file = 'analysis/{}_{}_agent'.format(U, tau_rec)

# make prediction files from samples
lab = 0
last_col = 0
for i, start in enumerate(start_ids):
    print('Processing chunk {} of {}'.format(i, len(start_ids)))
    curr_folder = '{}_{}_{}/'.format(U, tau_rec, start)
    with np.load(curr_folder + 'samples.npz') as d:
        chunk_samples = d['samples'].astype(float)
        chunk_vis = chunk_samples[..., :n_pxls + n_labels]

    # average pool on each chunk, then compute prediction
    chunk_vis = average_pool(chunk_vis, n_samples, n_samples)

    tmp_col = chunk_vis[..., :-n_labels].reshape(
                chunk_vis.shape[:-1] + img_shape)[..., -1]
    tmp_lab = chunk_vis[..., -n_labels:]

    lab = tmp_lab if i == 0 else np.vstack((lab, tmp_lab))
    last_col = tmp_col if i == 0 else np.vstack((last_col, tmp_col))

# save data (averaged samples for label units and last column)
np.savez_compressed(pred_file, label=lab, last_col=last_col)

# compute agent performance
result_dict = pong_agent.compute_performance(img_shape, stub_dict['data_name'],
                                             pred_file + '.npz')
np.savez_compressed(agent_file, **result_dict)

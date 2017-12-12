#! /usr/bin/env python
from __future__ import division
from __future__ import print_function
import numpy as np
import yaml
import sys
from lif_pong.utils import average_pool
import pong_agent


# identifier params will be save in the analysis file; defaults to fm-choice
def inf_speed_analysis(identifier_params, samples=None):
    with open('sim.yaml') as config:
        simdict = yaml.load(config)

    general_dict = simdict.pop('general')
    n_samples = general_dict['n_samples']
    start = general_dict['start_idx']
    chunksize = general_dict['chunksize']
    img_shape = tuple(general_dict['img_shape'])
    n_labels = img_shape[0]//3
    n_pxls = np.prod(img_shape)

    if samples is None:
        anadict = {'n_instances': chunksize,
                   'inf_success': float('nan'), 'inf_std': float('nan')}
    else:
        # make prediction files from samples
        chunk_vis = samples[..., :n_pxls + n_labels]
        chunk_vis = average_pool(chunk_vis, n_samples, n_samples)
        chunk_idxs = np.arange(start, start + len(chunk_vis))

        last_col = chunk_vis[..., :-n_labels].reshape(
            chunk_vis.shape[:-1] + img_shape)[..., -1]
        lab = chunk_vis[..., -n_labels:]

        print('Saving prediction data of {} instances'.format(len(last_col)))
        # save data (averaged samples for label units and last column)
        np.savez_compressed('prediction', label=lab, last_col=last_col,
                            data_idx=chunk_idxs)

        # compute agent performance
        result_dict = pong_agent.compute_performance(
            img_shape, general_dict['data_name'], chunk_idxs, last_col)
        print('Saving agent performance data...')
        np.savez_compressed('agent_performance', **result_dict)

        # save summarized analysis data for this chunk
        inf_success = float(result_dict['successes'][-1])
        inf_std = float(result_dict['successes_std'][-1])
        anadict = {'n_instances': result_dict['n_instances'],
                   'inf_success': inf_success, 'inf_std': inf_std}

    # add clamping-tso parameters for identification
    anadict['start_idx'] = start
    for k in identifier_params.keys():
        anadict[k] = identifier_params[k]
    with open('analysis', 'w') as f:
        f.write(yaml.dump(anadict))
    return anadict

if __name__ == '__main__':
    # if this is called as an independent analysis script load samples first
    try:
        with np.load('samples.npz') as d:
            samples = d['samples'].astype(float)
    except Exception as e:
        print('Missing sample file', file=sys.stderr)
        samples = None
    inf_speed_analysis(samples)
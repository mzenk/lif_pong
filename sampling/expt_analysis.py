#! /usr/bin/env python
from __future__ import division
from __future__ import print_function
import numpy as np
import yaml
import sys
import scipy.ndimage as ndimage
from lif_pong.utils import average_pool
from lif_pong.utils.data_mgmt import load_images
import pong_agent


# identifier params will be save in the analysis file
def inf_speed_analysis(samples=None, identifier_params=None, clamp_pos=-2,
                       data_set=None):
    with open('sim.yaml') as config:
        simdict = yaml.load(config)

    general_dict = simdict.pop('general')
    n_samples = general_dict['n_samples']
    start = general_dict['start_idx']
    img_shape = tuple(general_dict['img_shape'])
    kink_dict = None
    data_tuple = load_images(general_dict['data_name'], for_analysis=True)
    if len(data_tuple) == 4:
            kink_dict = data_tuple[3]
    if data_set is None:
        # take test set as default
        data_set = data_tuple[2]
    data_tuple = None
    data = data_set[0]
    n_labels = data_set[1].shape[1]
    n_pxls = np.prod(img_shape)

    if identifier_params is None:
        identifier_params = simdict.pop('identifier')

    # try the analysis and if something fails leave a analysis file with nan's
    # the program sould still crash after that so that it is not successfull
    try:
        if samples is None:
            raise IOError('No samples file found.')
        else:
            # make prediction files from samples
            chunk_vis = samples[..., :n_pxls + n_labels]
            chunk_pred = average_pool(chunk_vis, n_samples, n_samples)
            chunk_idxs = np.arange(start, start + len(chunk_pred))

            last_col = chunk_pred[..., :n_pxls].reshape(
                chunk_pred.shape[:-1] + img_shape)[..., -1]
            lab = chunk_pred[..., n_pxls:]

            print('Saving prediction data of {} instances'.format(len(last_col)))
            # save data (averaged samples for label units and last column)
            np.savez_compressed('prediction', label=lab, last_col=last_col,
                                data_idx=chunk_idxs)

            # make groundtruth array
            targets = data[chunk_idxs].reshape((-1,) + img_shape)[..., -1]
            if kink_dict is not None:
                prekink_test_targets = kink_dict['nokink_lastcol'][-len(data):]
                nsteps_pre = int(kink_dict['pos']*img_shape[1])
                targets = np.concatenate(
                    (np.tile(np.expand_dims(prekink_test_targets[chunk_idxs], 1), (1, nsteps_pre, 1)),
                    np.tile(np.expand_dims(targets, 1), (1, last_col.shape[1] - nsteps_pre, 1))),
                    axis=1)

            # compute agent performance
            result_dict = pong_agent.compute_performance(
                last_col, targets, chunk_idxs, max_speedup=np.inf)
            print('Saving agent performance data...')
            np.savez_compressed('agent_performance', **result_dict)

            # save summarized analysis data for this chunk
            # 'successes' contains numbers of instances for which an agent
            # with infinite speed is within paddle_width of the correct pos.
            # index of list represents number of clamped pixel columns
            inf_success = result_dict['successes']
            pred_error = result_dict['distances']

            # better quantity (which doesn't involve agent speed): integrated
            # prediction error. save the mean and squared resid. of this quantity
            # (median is not suited for chunk-wise application because not linear)
            # here it is assumed that the prediction positions are equidistant
            # and only relevant up to clamp_pos
            cum_prederr = pred_error[:, :clamp_pos].sum(axis=1) / img_shape[1]
            cum_prederr_sum = np.sum(cum_prederr, axis=0)
            cum_prederr_sqsum = np.sum(cum_prederr**2, axis=0)
            anadict = {'n_instances': result_dict['n_instances'],
                       'inf_success': float(inf_success[clamp_pos]),
                       'cum_prederr_sum': float(cum_prederr_sum),
                       'cum_prederr_sqsum': float(cum_prederr_sqsum)}

            if 'wrong_idx' in result_dict.keys():
                with open('wrong_cases', 'w') as f:
                    f.write(yaml.dump(result_dict['wrong_idx'].tolist()))
    except Exception as e:
        anadict = {'n_instances': float('nan'),
                   'inf_success': float('nan'),
                   'cum_prederr_sum': float('nan'),
                   'cum_prederr_sqsum': float('nan')}
        raise e
    finally:
        # add clamping-tso parameters for identification
        anadict['start_idx'] = start
        for k in identifier_params.keys():
            anadict[k] = identifier_params[k]
        with open('analysis', 'w') as f:
            f.write(yaml.dump(anadict))

    return anadict


# analysis for dreaming experiments
def burnin_analysis(vis_samples):
    # vis_samples.shape: (n_samples, n_visible)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # the first sample is always zero and the second one the initialised state;
    # both shall be neglected for analysis
    vis_samples[:2] = 0

    # assumption: network reaches stationary activity level state after 3/4 of
    # simulation
    mean_activity = vis_samples.mean(axis=1)
    smoothed_signal = ndimage.gaussian_filter1d(mean_activity, 5)
    stat_activity = mean_activity[-int(.75*len(mean_activity)):].mean()

    # good estimator of burnin: #samples after which activity level is at
    # X % (90, 95?) of final level
    thresh = .9*stat_activity
    above_thresh = np.argmax(smoothed_signal > thresh)

    # # use of sobel is discouraged because not always just one rising flank
    # sobel_filtered = ndimage.sobel(smoothed_signal)
    # edge_loc = int(np.argmax(sobel_filtered))

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    ax[0].hlines(stat_activity, 0, len(mean_activity), label='stationary')
    ax[1].hlines(thresh, 0, len(smoothed_signal), colors='C1', label='threshold')
    ax[1].hlines(stat_activity, 0, len(smoothed_signal), label='stationary')

    ax[0].plot(mean_activity, '.', label='signal')
    ax[1].plot(smoothed_signal, '.', label='smoothed')

    ax[0].set_ylim(bottom=0.)
    ax[1].set_ylim(bottom=0.)
    ax[0].legend()
    ax[1].legend()
    fig.savefig('activity.png')
    return {'stat_activity': float(stat_activity),
            'thresh_crossing': int(above_thresh)}


# under construction
def mode_switch_analysis(vis_samples, target_data, n_init):
    # vis_samples.shape: (n_instances, n_samples, n_visible)
    # target_data.shape: (n_instances, n_visible)
    l2_diff = np.sqrt(np.sum(
        (vis_samples - np.expand_dims(target_data, 1))**2, axis=2))

    smoothed_signal = ndimage.gaussian_filter1d(l2_diff, 5, axis=1)
    sobel_filtered = ndimage.sobel1d(smoothed_signal, axis=1)
    edge_loc = np.argmin(sobel_filtered, axis=1)
    # for threshold assume that after 3/4 of expt, difference is stationary
    stat_diff = l2_diff[-int(.75*l2_diff.shape[1]):].mean(axis=1)
    thresh = 1.1*stat_diff
    thresh_crossing = np.argmax(smoothed_signal < np.expand_dims(thresh, 1), axis=1)

    return {'n_instances': len(vis_samples),
            'edge_loc': edge_loc.tolist(),
            'stat_diff': stat_diff.tolist(),
            'thresh_crossing': thresh_crossing.tolist()}


if __name__ == '__main__':
    # if this is called as an independent analysis script load samples first
    try:
        with np.load('samples.npz') as d:
            samples = d['samples']
    except Exception as e:
        print('Missing sample file', file=sys.stderr)
        samples = None
    inf_speed_analysis(samples)

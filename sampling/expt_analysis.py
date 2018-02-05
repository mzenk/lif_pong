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
    if data_set is None:
        # take test set as default
        _, _, data_set = load_images(general_dict['data_name'])
    n_labels = img_shape[0]//3
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

            last_col = chunk_pred[..., :-n_labels].reshape(
                chunk_pred.shape[:-1] + img_shape)[..., -1]
            lab = chunk_pred[..., -n_labels:]

            print('Saving prediction data of {} instances'.format(len(last_col)))
            # save data (averaged samples for label units and last column)
            np.savez_compressed('prediction', label=lab, last_col=last_col,
                                data_idx=chunk_idxs)

            # compute agent performance
            result_dict = pong_agent.compute_performance(
                img_shape, data_set, chunk_idxs, last_col, max_speedup=np.inf)
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

    # try getting burn-in by looking for edges in activity
    mean_activity = vis_samples.squeeze().mean(axis=1)
    smoothed_signal = ndimage.gaussian_filter1d(mean_activity, 5)
    sobel_filtered = ndimage.sobel(smoothed_signal)
    # other option: threshold crossing
    thresh = .9
    thresh_crossings = np.where(smoothed_signal > thresh*smoothed_signal.max())[0]
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    ax[0].plot(mean_activity, '.', label='signal')
    ax[1].plot(smoothed_signal, '.', label='smoothed')
    ax[1].plot(ax[1].get_xlim(), 2*[thresh*smoothed_signal.max()], 'C1:')
    ax[2].plot(sobel_filtered, '.', label='sobel')
    fig.savefig('activity.png')
    return {'edge_loc': int(np.argmax(sobel_filtered)),
            'thresh_crossing': int(thresh_crossings.min())}

def mode_switch_analysis():
    pass

if __name__ == '__main__':
    # if this is called as an independent analysis script load samples first
    try:
        with np.load('samples.npz') as d:
            samples = d['samples']
    except Exception as e:
        print('Missing sample file', file=sys.stderr)
        samples = None
    inf_speed_analysis(samples)

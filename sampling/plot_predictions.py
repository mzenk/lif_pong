from __future__ import division
from __future__ import print_function
import sys
import yaml
import os
import numpy as np
from lif_pong.utils import average_pool, average_helper
from lif_pong.utils.data_mgmt import make_figure_folder, load_images, make_data_folder
import pong_agent
from cycler import cycler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

simfolder = '/work/ws/nemo/hd_kq433-data_workspace-0/experiment/simulations'


# from stackoverflow for checking what goes wrong
def findDiff(d1, d2, path=""):
    for k in d1.keys():
        if k not in d2.keys():
            print(path, ":")
            print(k + " as key not in d2", "\n")
        else:
            if type(d1[k]) is dict:
                if path == "":
                    path = k
                else:
                    path = path + "->" + k
                findDiff(d1[k], d2[k], path)
            else:
                if d1[k] != d2[k]:
                    print(path, ":")
                    print(" - ", k, " : ", d1[k])
                    print(" + ", k, " : ", d2[k])


def merge_chunks(basefolder, identifier_dict, savename):
    # make prediction files from samples
    data_idx = None
    prediction = None
    counter = 0
    folder_list = os.listdir(basefolder)
    for folder in folder_list:
        simpath = os.path.join(basefolder, folder)
        with open(os.path.join(simpath, 'sim.yaml')) as config:
            simdict = yaml.load(config)
            chunk_start = simdict['general'].pop('start_idx')
            sim_identifiers = simdict.pop('identifier')
        # only append predictions if it has same parameters
        if identifier_dict != sim_identifiers:
            # findDiff(identifier_dict, sim_identifiers)
            continue

        general_dict = simdict.pop('general')
        n_samples = general_dict['n_samples']
        img_shape = tuple(general_dict['img_shape'])
        n_labels = img_shape[0]//3
        n_pxls = np.prod(img_shape)
        try:
            with np.load(os.path.join(simpath, 'samples.npz')) as d:
                chunk_samples = d['samples'].astype(float)
                if 'data_idx' in d.keys():
                    chunk_idx = d['data_idx']
                else:
                    chunk_idx = np.arange(
                        chunk_start, chunk_start + len(chunk_samples))
        except IOError:
            print('File not found: ' + folder + '/samples.npz',
                  file=sys.stderr)
            continue

        chunk_vis = chunk_samples[..., :n_pxls + n_labels]
        # average pool on each chunk, then compute prediction
        chunk_vis = average_pool(chunk_vis, n_samples, n_samples)

        tmp_col = chunk_vis[..., :-n_labels].reshape(
            chunk_vis.shape[:-1] + img_shape)[..., -1]

        if counter == 0:
            data_idx = chunk_idx
            prediction = tmp_col
        else:
            data_idx = np.hstack((data_idx, chunk_idx))
            prediction = np.vstack((prediction, tmp_col))
        counter += 1

    print('Merged prediction data of {} chunks'.format(counter),
          file=sys.stdout)
    if prediction is not None:
        # save prediction data (faster for re-plotting)
        print('Saved as '+savename)
        np.savez(savename, last_col=prediction, lab=prediction[..., :12],
                 data_idx=data_idx)
    else:
        raise RuntimeError('No matching data could be found. '
                           'Maybe wrong identifiers.')
    return prediction, data_idx


def get_performance_data(basefolder, identifier_dict):
    id_string = '_'.join(
        ['{}'.format(identifier_dict[k]) for k in identifier_dict.keys()])
    savename = os.path.join(make_data_folder(),
                            basefolder.split('/')[-1] + id_string)
    # try to load data
    try:
        with np.load(savename + '.npz') as f:
            prediction = f['last_col']
            data_idx = f['data_idx']
        print('Loaded prediction with shape {}.'.format(prediction.shape))
    except IOError:
        prediction, data_idx = merge_chunks(basefolder, identifier_dict,
                                            savename)

    # load stuff for pong agent from an arbitrary sim folder
    # (params should be the same for all)
    simpath = os.path.join(basefolder, os.listdir(basefolder)[0])
    with open(os.path.join(simpath, 'sim.yaml')) as config:
        general_dict = yaml.load(config)['general']

    if prediction is not None:
        # compute agent performance
        agent_result = pong_agent.compute_performance(
            tuple(general_dict['img_shape']), general_dict['data_name'],
            data_idx, prediction)

    return data_idx, prediction, agent_result


def evaluate_prediction(prediction, targets, n_pos, use_labels=False,
                        lab2pxl=3):
    if use_labels:
        raise NotImplementedError

    n_instances = prediction.shape[0]
    n_frames = prediction.shape[1]

    # transform one-hot-encoded prediction and targets to position in pxls
    target_pos = average_helper(n_pos, targets)
    predicted_pos = np.zeros((n_instances, n_frames))
    for i in range(n_instances):
        predicted_pos[i] = average_helper(n_pos, prediction[i])

    if use_labels:
        predicted_pos *= lab2pxl
        target_pos *= lab2pxl

    # compute percentiles of prediction error
    dist = np.abs(predicted_pos - target_pos.reshape((len(target_pos), 1)))
    dist_median = np.percentile(dist, 50, axis=0)
    dist_quartile_lower = np.percentile(dist, 25, axis=0)
    dist_quartile_upper = np.percentile(dist, 75, axis=0)
    return dist_median, dist_quartile_lower, dist_quartile_upper


# copied and adapted from prediction_quality.py; ugly but no time for more
def plot_prediction_error(ax, prediction, data_name, data_idx, label=None,
                          x_data=None):
    n_pxlrows = prediction.shape[2]
    n_frames = prediction.shape[1]
    if x_data is None:
        x_data = np.linspace(0, 1, n_frames)

    # load data and compute groundtruth
    _, _, test_set = load_images(data_name)
    test_data = test_set[0][data_idx]

    targets = test_data.reshape((len(test_data), n_pxlrows, -1))[..., -1]

    median, lower_quart, upper_quart = \
        evaluate_prediction(prediction, targets, n_pxlrows)

    # add prediction error curve to plot
    ax.plot(x_data, median, '.-')
    ax.fill_between(x_data, lower_quart, upper_quart, alpha=.3, label=label)


def plot_agent_performance(ax, agent_dict, label=None):
    succrate = agent_dict['successes'] / agent_dict['n_instances']
    # succrate_std = agent_dict['successes_std'] / agent_dict['n_instances']
    # std of binary rv is not very helpfu
    speeds = agent_dict['speeds']
    print('Maximum success rate: {:.3f}'.format(np.max(succrate)))
    ax.plot(speeds, succrate, label=label)


def main(identifier_list):
    # set up figures
    fig_pe, ax_pe = plt.subplots()
    ax_pe.set_ylabel('Prediction error')
    ax_pe.set_ylim([-.5, 16])
    ax_pe.set_xlabel('Ball position / field length')
    color_cycle = [plt.cm.rainbow(i)
                   for i in np.linspace(0, 1, len(identifier_list))]
    ax_pe.set_prop_cycle(cycler('color', color_cycle))

    fig_ap, ax_ap = plt.subplots()
    ax_ap.set_xlabel('Agent speed / ball speed')
    ax_ap.set_ylabel('Success rate')
    ax_ap.set_ylim([0., 1.1])
    for i, identifier_dict in enumerate(identifier_list):
        expt_name = identifier_dict.pop('experiment')
        # load yaml-config of experiment
        config_file = os.path.join(simfolder, '01_runs', expt_name)
        with open(config_file) as config:
            experiment_dict = yaml.load(config)
        # get indices of data used for simulation
        stub_dict = experiment_dict.pop('stub')
        data_name = stub_dict['general']['data_name']
        try:
            label = identifier_dict.pop('label')
        except KeyError:
            label = 'parameters {}'.format(i)
        data_idx, prediction, agent_result = get_performance_data(
            os.path.join(simfolder, expt_name), identifier_dict)
        plot_prediction_error(ax_pe, prediction, data_name, data_idx, label)
        plot_agent_performance(ax_ap, agent_result, label)

    ax_pe.legend()
    fig_pe.savefig(make_figure_folder() + 'pred_error.png')  #, transparent=True)
    ax_ap.plot(ax_ap.get_xlim(), [1, 1], 'k:')
    ax_ap.legend()
    fig_ap.savefig(make_figure_folder() + 'agent_perf.png')  #, transparent=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Wrong number of arguments. Please provide a yaml that specifies'
              ' which simulations to compare.')
        sys.exit()

    # load list of identifiers used for selecting data
    with open(sys.argv[1]) as f:
        identifier_list = yaml.load(f)

    main(identifier_list)

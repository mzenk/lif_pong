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
import matplotlib.pyplot as plt

plt.style.use('mthesis_style')

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

        try:
            with np.load(os.path.join(simpath, 'prediction.npz')) as d:
                tmp = d['last_col'].astype(float)
                if 'data_idx' in d.keys():
                    chunk_idx = d['data_idx']
                else:
                    chunk_idx = np.arange(
                        chunk_start, chunk_start + len(tmp))
        except IOError:
            print('No prediction file found in ' + folder, file=sys.stderr)
            continue

        if counter == 0:
            data_idx = chunk_idx
            prediction = tmp
        else:
            data_idx = np.hstack((data_idx, chunk_idx))
            prediction = np.vstack((prediction, tmp))
        counter += 1

    print('Merged prediction data of {} chunks'.format(counter),
          file=sys.stdout)
    if prediction is not None:
        # save prediction data (faster for re-plotting)
        print('Saved as ' + savename)
        np.savez(savename, last_col=prediction, lab=prediction[..., :12],
                 data_idx=data_idx)
    else:
        raise RuntimeError('No matching data could be found. '
                           'Maybe wrong identifiers.\n' +
                           ',\n'.join(['{}={}'.format(k, identifier_dict[k]) for k in identifier_dict.keys()]))
    return prediction, data_idx


def get_performance_data(basefolder, identifier_dict, test_data,
                         leave_uncovered=1):
    id_string = '_'.join(
        ['{}'.format(identifier_dict[k]) for k in sorted(identifier_dict.keys())])
    savename = os.path.join(make_data_folder(),
                            basefolder.split('/')[-1] + '_' + id_string)
    # try to load data
    try:
        with np.load(savename + '.npz') as f:
            prediction = f['last_col']
            data_idx = f['data_idx']
        print('Loaded prediction with shape {}.'.format(prediction.shape))
    except IOError:
        prediction, data_idx = merge_chunks(basefolder, identifier_dict,
                                            savename)

    if prediction is not None:
        targets = test_data[data_idx].reshape(
            (len(data_idx), prediction.shape[2], -1))[..., -1]
        # compute agent performance
        agent_result = pong_agent.compute_performance(
            prediction, targets, data_idx, leave_uncovered=leave_uncovered)

    return data_idx, prediction, agent_result


def compute_prediction_error(predictions, targets, n_pos, use_labels=False,
                             lab2pxl=3):
    if use_labels:
        raise NotImplementedError

    n_instances = predictions.shape[0]
    n_frames = predictions.shape[1]

    # transform one-hot-encoded prediction and targets to position in pxls
    if targets.size != predictions.size:
        target_pos = average_helper(n_pos, targets).reshape((n_instances, 1))
    else:
        target_pos = np.zeros((n_instances, n_frames))
    predicted_pos = np.zeros((n_instances, n_frames))
    for i in range(n_instances):
        predicted_pos[i] = average_helper(n_pos, predictions[i])
        if targets.size == predictions.size:
            target_pos[i] = average_helper(n_pos, targets[i])

    if use_labels:
        predicted_pos *= lab2pxl
        target_pos *= lab2pxl

    return np.abs(predicted_pos - target_pos)


# copied and adapted from prediction_quality.py; ugly but no time for more
def plot_prediction_error(ax, pred_error, label=None, x_data=None):
    n_frames = pred_error.shape[1]
    if x_data is None:
        x_data = np.linspace(0, 1, n_frames)

    median = np.percentile(pred_error, 50, axis=0)
    mean = np.mean(pred_error, axis=0)
    lower_quart = np.percentile(pred_error, 25, axis=0)
    upper_quart = np.percentile(pred_error, 75, axis=0)
    print('Integral = {:.1f}'.format(
          median[:48].sum()/48.))

    # add prediction error curve to plot
    if label == 'Static clamping':
        ax.plot(x_data[:-1], median[:-1], '.-', alpha=.7)
        ax.fill_between(x_data[:-1], median[:-1], median[:-1], alpha=.3, label=label)
    else:
        ax.plot(x_data[:-1], median[:-1], '.-', alpha=.7)
        ax.fill_between(x_data[:-1], lower_quart[:-1], upper_quart[:-1], alpha=.3, label=label)

    # ax.plot(x_data, mean, '.-', alpha=.7, label='mean ' + label)


def plot_agent_performance(ax, agent_dict, label=None):
    succrate = agent_dict['successes'] / agent_dict['n_instances']
    # succrate_std = agent_dict['successes_std'] / agent_dict['n_instances']
    # std of binary rv is not very helpfu
    speeds = agent_dict['speeds']
    print('Maximum success rate: {:.3f}'.format(np.max(succrate)))
    if label == 'baseline':
        ax.plot(speeds, succrate, 'k:')
    else:
        ax.plot(speeds, succrate, label=label)


def plot_prediction_error_dist(ax, pred_error, dist_pos=None, label=None,
                               title=None, color='C0'):
    if dist_pos is None:
        dist_pos = pred_error.shape[1] - 1
    if title is not None:
        ax.set_title(title)
    ax.hist(pred_error[:, dist_pos], bins='auto', label=label, color=color,
            histtype='stepfilled')
    # ax.set(xlabel='Prediction error')


def list_bad_examples(pred_errors, img_shape):
    median = np.percentile(pred_errors, 50, axis=0)
    # mean = np.mean(pred_error, axis=0)
    lower_quart = np.percentile(pred_errors, 25, axis=0)
    upper_quart = np.percentile(pred_errors, 75, axis=0)

    bad_examples = []
    worse_examples = []
    good_examples = []
    fine_examples = []
    for i in range(pred_errors.shape[1]):
        err = pred_errors[:, i]
        above_med = err > median[i]
        above_upquart = err > upper_quart[i]
        below_loquart = err < lower_quart[i]

        bad_examples.append(np.where(np.logical_and(above_med, 1 - above_upquart))[0].tolist())
        worse_examples.append(np.where(above_upquart)[0].tolist())
        good_examples.append(np.where(np.logical_and(1 - above_med, 1 - below_loquart))[0].tolist())
        fine_examples.append(np.where(below_loquart)[0].tolist())

    # save dictionary in yaml format
    d = {'fine': fine_examples, 'good': good_examples,
         'bad': bad_examples, 'worse': worse_examples}
    with open('mylist.yaml', 'w') as f:
        f.write(yaml.dump(d))


def main(identifier_list, list_bad=False):
    fig_pe, (ax_pe, ax_ap) = plt.subplots(1, 2)
    fig_pe.set_figheight(fig_pe.get_figwidth()/2.2)
    fig_pe.subplots_adjust(wspace=.25, right=.98, left=.1, top=.95, bottom=.133)
    ax_ap.set_xlim([-0.1, 1.8])
    ax_pe.set_xlim([-.01, 1.01])

    # fig_ap, ax_ap = plt.subplots()
    ax_pe.set_ylabel('Prediction error [pxls]')
    ax_pe.set_xlabel('Ball position / field length')
    ax_ap.set_xlabel('Agent speed / ball speed ($r$)')
    ax_ap.set_ylabel('Success rate')
    ax_ap.set_ylim([0., 1.1])

    # plot the pred.err. distributions at some predefined places

    # ---> gauss plot in MA
    n_ids = len(identifier_list)
    n_dist = 3
    rel_pos_dists = np.arange(8, 17, 4)/48.
    titlelist = ['{:d}/48 clamped'.format(int(s)) for s in rel_pos_dists*48]
    fig_pd, axarr_pd = plt.subplots(n_ids, n_dist, sharey='all', sharex='col')
    fig_pd.set_figheight(fig_pd.get_figwidth()/2.2)
    if n_ids == 1 or n_dist == 1:
        axarr_pd = np.array(axarr_pd).reshape((n_ids, n_dist))
    axarr_pd[-1, 0].set(ylabel='Frequency', xlabel='Prediction error')
    axarr_pd[-1, 0].set_ylim(top=760)
    fig_pd.subplots_adjust(wspace=0.1, bottom=.18, right=.95)
    axarr_pd[0, 1].axvline(9.5, linestyle=':', lw=1., color='k')

    # # ---> kink plot in MA
    # rel_pos_dists = np.arange(47, 48)/48.
    # fig_pd, axarr_pd = plt.subplots(3, 3, sharey='all', sharex='all')
    # fig_pd.set_figheight(fig_pd.get_figwidth())
    # axarr_pd[0, 0].set_xlim(right=5.5)
    # axarr_pd[-1, 0].set(ylabel='Frequency', xlabel='Prediction error')
    # titlelist = [r'$x_\mathrm{kink}$ = ' + s for s in ['0.6', '0.7', '0.8']]
    # # insert at # (*):
    # for j, dist_pos in enumerate(rel_pos_dists * img_shape[1]):
    #         if i//3 == 0:
    #             title = titlelist[i%3]
    #         else:
    #             title = None
    #         curr_ax = axarr_pd[i//3, i%3]
    #         plot_prediction_error_dist(curr_ax, pred_error,
    #                                    int(dist_pos), label, title=title,
    #                                    color='C{}'.format(i//3))
    #         curr_ax.axvline(2., color='k', linestyle=':')

    for i, identifier_dict in enumerate(identifier_list):
        expt_name = identifier_dict.pop('experiment')
        # load yaml-config of experiment
        config_file = os.path.join(simfolder, '01_runs', expt_name)
        with open(config_file) as config:
            experiment_dict = yaml.load(config)
        # load data
        stub_dict = experiment_dict.pop('stub')
        if 'data_name' in identifier_dict.keys():
            data_name = identifier_dict['data_name']
        else:
            data_name = stub_dict['general']['data_name']
        img_shape = tuple(stub_dict['general']['img_shape'])
        data_tuple = load_images(data_name, for_analysis=True)
        test_set = data_tuple[2]
        kink_dict = None
        if len(data_tuple) == 4:
            kink_dict = data_tuple[3]

        try:
            label = identifier_dict.pop('label')
        except KeyError:
            label = 'parameters {}'.format(i)

        # load data and compute prediction errors
        test_data = test_set[0]
        data_idx, prediction, agent_result = get_performance_data(
            os.path.join(simfolder, expt_name), identifier_dict, test_data)
            # leave_uncovered=24)
        targets = test_data[data_idx].reshape((-1,) + img_shape)[..., -1]

        if kink_dict is not None:
            prekink_test_targets = kink_dict['nokink_lastcol'][-len(test_data):]
            nsteps_pre = int(np.round(kink_dict['pos']*img_shape[1]))
            targets = np.concatenate(
                (np.tile(np.expand_dims(prekink_test_targets[data_idx], 1), (1, nsteps_pre, 1)),
                 np.tile(np.expand_dims(targets, 1), (1, prediction.shape[1] - nsteps_pre, 1))),
                axis=1)

        n_pos = prediction.shape[2]
        pred_error = compute_prediction_error(prediction, targets, n_pos)

        plot_prediction_error(ax_pe, pred_error, label)
        for j, dist_pos in enumerate(rel_pos_dists * img_shape[1]):
            if i//n_dist == 0:
                title = titlelist[j]
            else:
                title = None
            curr_ax = axarr_pd[i, j]
            plot_prediction_error_dist(curr_ax, pred_error,
                                       int(dist_pos), label, title=title)
        plot_agent_performance(ax_ap, agent_result, label)

        if list_bad:
            list_bad_examples(pred_error, img_shape)

    # # Add line for ideal agent
    # agent_dict = pong_agent.compute_ideal_performance(img_shape, test_set, data_idx)
    # succrate = agent_dict['successes'] / agent_dict['n_instances']
    # speeds = agent_dict['speeds']
    # ax_ap.plot(speeds, succrate, 'k:', label='Omniscient agent')
    ax_ap.plot([-0.1, 2.], [1,1], 'k:')

    ax_ap.legend()
    fig_pd.tight_layout()
    # ax_pe.legend()
    if axarr_pd.shape[0] > 1:
        [axarr_pd[i, -1].legend() for i in range(len(axarr_pd))]
    # fig_pe.tight_layout()
    fig_pe.savefig(make_figure_folder() + 'pred_error.pdf')  #, transparent=True)
    fig_pd.savefig(make_figure_folder() + 'pred_error_dist.pdf')  #, transparent=True)
    # fig_ap.savefig(make_figure_folder() + 'agent_perf.pdf')  #, transparent=True)


if __name__ == '__main__':
    if len(sys.argv) > 3:
        print('Wrong number of arguments. Please provide a yaml that specifies'
              ' which simulations to compare.')
        sys.exit()

    # load list of identifiers used for selecting data
    with open(sys.argv[1]) as f:
        identifier_list = yaml.load(f)

    if len(sys.argv) == 3 and sys.argv[2] == 'bad':
        list_bad = True
    else:
        list_bad = False

    main(identifier_list, list_bad)

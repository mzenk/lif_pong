from __future__ import division
from __future__ import print_function
import numpy as np
from utils.data_mgmt import get_data_path, make_figure_folder, load_images
from utils import average_helper
from cycler import cycler
import os
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt


mpl.rcParams['font.size'] = 14
# mpl.rcParams['text.usetex'] = True


def load_prediction_data(file_name, use_labels=False):
    with np.load(file_name) as d:
        avg_lab = d['label']
        avg_vis = d['last_col']
        if 'data_idx' in d.keys():
            data_idx = d['data_idx']
        else:
            data_idx = np.arange(len(avg_vis))
        if 'winpos' in d.keys():
            winpos = d['winpos']
        else:
            winpos = np.arange(avg_vis.shape[1])

    lab2pxl = img_shape[0] / n_labels
    # compare vis_prediction not to label but to actual pixels
    data_name = os.path.basename(file_name).split('_')[0] + \
        '_var_start{}x{}'.format(*img_shape)
    _, _, test_set = load_images(data_name)
    if use_labels:
        targets = average_helper(n_labels, test_set[1]) * lab2pxl
    else:
        last_col = test_set[0].reshape((-1,) + img_shape)[..., -1]
        targets = average_helper(img_shape[0], last_col)
    targets = targets[data_idx]

    lab_prediction = np.zeros(avg_lab.shape[:-1])
    vis_prediction = np.zeros(avg_vis.shape[:-1])
    for i in range(len(avg_lab)):
        lab_prediction[i] = average_helper(n_labels, avg_lab[i])
        vis_prediction[i] = average_helper(img_shape[0], avg_vis[i])
    prediction = lab_prediction * lab2pxl if use_labels else vis_prediction

    # compute percentiles of prediction error
    dist = np.abs(prediction - targets.reshape((len(targets), 1)))
    # dist = np.abs(vis_prediction - targets.reshape((len(targets), 1)))
    dist_median = np.percentile(dist, 50, axis=0)
    dist_quartile_lower = np.percentile(dist, 25, axis=0)
    dist_quartile_upper = np.percentile(dist, 75, axis=0)
    fractions = winpos / img_shape[1]
    return fractions, dist_median, dist_quartile_lower, dist_quartile_upper


def load_agent_data(file_name):
    with np.load(get_data_path('pong_agent') + file_name + '.npz') as d:
        success = d['successes']
        dist = d['distances']
        speeds = d['speeds']
    print('Asymptotic value (full history): {}'.format(success.max()))
    return success, dist, speeds


def plot_prediction_error():
    # compare window sizes
    # data_path = get_data_path('gibbs_sampling')
    # winsize = [48, 12]
    # data_prefixes = [data_path + 'knick_win{}'.format(w) for w in winsize]
    # data_prefixes.insert(0, data_path + 'pong_win48')
    # labels = ['window size: {}'.format(win) for win in winsize]
    # labels.insert(0, 'No knick')
    # fig_name = 'gibbs_knick_prediction_error'

    # Other comparisons are possible, just make a list of which prediction
    # files to use
    data_path = get_data_path('lif_clamp_window')
    data_prefixes = [data_path + s for s in ['pong_win48', 'pong_win48_post']]
    labels = ['pre', 'post']
    fig_name = 'lif_compare_post_prederror'

    fnames = [s + '_prediction.npz' for s in data_prefixes]
    fractions, medians, upper_quartiles, lower_quartiles = [], [], [], []
    for fn in fnames:
        frac, median, lower_quart, upper_quart = load_prediction_data(fn)
        fractions.append(frac)
        medians.append(median)
        lower_quartiles.append(lower_quart)
        upper_quartiles.append(upper_quart)

    # plot prediction error
    fig, ax = plt.subplots()
    ax.set_ylabel('Prediction error d')
    ax.set_ylim([-.5, 16])
    ax.set_xlabel('Ball position / field length')
    color_cycle = [plt.cm.rainbow(i) for i in np.linspace(0, 1, len(medians))]
    ax.set_prop_cycle(cycler('color', color_cycle))

    for i, med in enumerate(medians):
        ax.plot(fractions[i], med, '.-')
        ax.fill_between(fractions[i], lower_quartiles[i], upper_quartiles[i],
                        alpha=.3, label=labels[i])

    plt.legend()
    plt.tight_layout()
    plt.savefig(make_figure_folder() + fig_name + '.pdf', transparent=True)
    plt.close(fig)


def plot_agent_performance():
    # plot agent performance
    data_prefixes = ['gibbs_knick_win48', 'gibbs_knick_win12',
                     'gibbs_pong_win48']
    labels = ['Knick (48)', 'Knick (12)', 'Pong (48)']
    file_names = [s + '_agent_performance' for s in data_prefixes]
    figname = 'gibbs_knick_agent_performance'

    fig, ax = plt.subplots()
    ax.set_xlabel('Agent speed / ball speed')
    ax.set_ylabel('Success rate')
    ax.set_ylim([0., 1.])

    successes, distances, speeds = [], [], []
    for fn in file_names:
        suc, dis, spe = load_agent_data(fn)
        successes.append(suc)
        distances.append(dis)
        speeds.append(spe)

    for i, fn in enumerate(file_names):
        ax.plot(speeds[i], successes[i], label=labels[i])

    plt.legend()
    plt.savefig(make_figure_folder() + figname + '.pdf')


# main
img_shape = (36, 48)
n_labels = img_shape[0]//3
plot_prediction_error()
# plot_agent_performance()

# === other plots, not functional ===
# image dissimilarity
# plt.errorbar(fractions, img_diff, fmt='ro', yerr=img_diff_std)
# plt.ylabel('L2 image dissimilarity')
# plt.xlabel(xlabel)
# plt.tight_layout()
# plt.savefig('figures/' + fig_name + '.pdf', transparent=True)

# # show distance distribution
# fig, ax = plt.subplots()
# distances[10]
# n, bins, patches = ax.hist(distances[10], 50, facecolor='green', alpha=0.75)
# plt.tight_layout()
# plt.savefig('figures/histo.png')

# # histogram of errors
# fig, ax = plt.subplots()
# n_bins = 30
# bins = np.linspace(0, 12, n_bins + 1)
# d_histograms = np.zeros((dist.shape[0], n_bins))
# for i, d in enumerate(dist):
#     d_histograms[i], _ = np.histogram(d, bins=bins)
# plt.imshow(d_histograms.T)
# plt.ylabel('Distance to target')
# plt.xlabel('time')
# plt.savefig('figures/test_histo.png')

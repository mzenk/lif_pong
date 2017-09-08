from __future__ import division
from __future__ import print_function
import numpy as np
from utils.data_mgmt import get_data_path, make_figure_folder, load_images
from utils import average_helper
from cycler import cycler
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt


mpl.rcParams['font.size'] = 14
# mpl.rcParams['text.usetex'] = True

img_shape = (36, 48)
n_labels = 12


def load_data(sampling_method, file_name, use_labels=False):
    with np.load(get_data_path(sampling_method + '_sampling') +
                 file_name + '.npz') as d:
        avg_lab = d['label']
        avg_vis = d['last_col']
        if 'data_idx' in d.keys():
            data_idx = d['data_idx']
        else:
            data_idx = np.arange(len(avg_vis))

    lab2pxl = img_shape[0] / n_labels
    # compare vis_prediction not to label but to actual pixels
    data_name = file_name.split('_')[0] + '_var_start{}x{}'.format(*img_shape)
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
    fractions = np.linspace(0, 1, avg_lab.shape[1])
    return fractions, dist_median, dist_quartile_lower, dist_quartile_upper


samp_meth = 'lif'
pot_str = 'pong'

win_sizes = [48]

fractions, medians, upper_quartiles, lower_quartiles = [], [], [], []
for win in win_sizes:
    fname = pot_str + '_win{}_prediction'.format(win)
    frac, median, lower_quart, upper_quart = load_data(samp_meth, fname)
    fractions.append(frac)
    medians.append(median)
    lower_quartiles.append(lower_quart)
    upper_quartiles.append(upper_quart)

# plot prediction error
fig_name = '{}_{}_prediction_error'.format(samp_meth, pot_str)
fig, ax = plt.subplots()
ax.set_ylabel('Prediction error d')
ax.set_ylim([-.5, 16])
ax.set_xlabel('Ball position / field length')
color_cycle = [plt.cm.rainbow(i) for i in np.linspace(0, 1, len(win_sizes))]
ax.set_prop_cycle(cycler('color', color_cycle))

for i, win in enumerate(win_sizes):
    ax.plot(fractions[i], medians[i], '.-')
    ax.fill_between(fractions[i], lower_quartiles[i], upper_quartiles[i],
                    alpha=.3, label='window: {}'.format(win_sizes[i]))

plt.legend()
plt.tight_layout()
plt.savefig(make_figure_folder() + fig_name + '.pdf', transparent=True)
plt.close(fig)

# # plot agent performance
# fig, ax = plt.subplots()
# ax.set_xlabel('Agent speed / ball speed')
# ax.set_ylabel('Success rate')
# ax.set_ylim([0., 1.])

# data_file = pot_str + '_win{}_agent_performance'.format(win_size1)
# with np.load(get_data_path('pong_eval_agent') + data_file + '.npz') as d:
#     success1 = d['successes']
#     dist1 = d['distances']
#     speeds1 = d['speeds']
#     print('Asymptotic value (full history): {}'.format(success1.max()))
# ax.plot(speeds1, success1, '-r', label='window: {}'.format(win_size1))

# data_file = pot_str + '_win{}_agent_performance'.format(win_size2)
# with np.load(get_data_path('pong_eval_agent') + data_file + '.npz') as d:
#     success2 = d['successes']
#     dist2 = d['distances']
#     speeds2 = d['speeds']
#     print('Asymptotic value (full history): {}'.format(success2.max()))
# ax.plot(speeds2, success2, '-b', label='window: {}'.format(win_size2))

# data_file = 'baseline_agent_performance'
# with np.load(get_data_path('pong_eval_agent') + data_file + '.npz') as d:
#     success_base = d['successes']
#     dist_base = d['distances']
#     speeds_base = d['speeds']
#     print('Asymptotic value (comparison): {}'.format(success_base.max()))
# ax.plot(speeds_base, success_base, '-k', label='baseline')

# plt.legend()
# plt.savefig(make_figure_folder() + pot_str + '_agent_performance.pdf')

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

from __future__ import division
from __future__ import print_function
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['font.size'] = 14
# mpl.rcParams['text.usetex'] = True


def load_data(file_name):
    with np.load('data/' + file_name + '.npz') as d:
        predictions = d['pred']  # predictions is on the label scale
        pred_lab = d['lab']
        distances = d['dist']
        speed = d['speed']
        img_diff = d['idiff']

    # calculate statistical quantities
    # dist_median = np.mean(distances, axis=1)
    # dist_std = np.std(distances, axis=1)
    dist_median = np.percentile(distances, 50, axis=1)
    dist_quartile_lower = np.percentile(distances, 25, axis=1)
    dist_quartile_upper = np.percentile(distances, 75, axis=1)
    fractions = np.linspace(0, 1, len(distances))
    return fractions, dist_median, dist_quartile_lower, dist_quartile_upper

img_shape = (36, 48)
win_size1 = 48
win_size2 = 8
n_sampl = 100
pot_str = 'gaus'
# dname1 = pot_str + '_uncover{}w{}s'.format(win_size1, n_sampl)
# dname2 = pot_str + '_uncover{}w{}s'.format(win_size2, n_sampl)
dname1 = pot_str + '_pred{}w'.format(win_size1)
dname2 = pot_str + '_pred{}w'.format(win_size2)
fractions1, dist_median1, dist_quartile_lower1, dist_quartile_upper1 = \
    load_data(dname1)
fractions2, dist_median2, dist_quartile_lower2, dist_quartile_upper2 = \
    load_data(dname2)


# plotting
fig_name = pot_str + '_uncover'
fig, ax1 = plt.subplots()
ax1.plot(fractions1, dist_median1, 'b.-')
ax1.fill_between(fractions1, dist_quartile_lower1, dist_quartile_upper1,
                 alpha=.5, facecolor='blue', label='window: full')
# ax1.plot(fractions2, dist_median2, 'r.-')
# ax1.fill_between(fractions2, dist_quartile_lower2, dist_quartile_upper2,
#                  alpha=.33, facecolor='red', label='window: 8/48')
ax1.set_ylabel('Prediction error d')
ax1.set_ylim([-.5, 6.5])
ax1.set_xlabel('Ball position / field length')
plt.legend()

# not used any more
# ax2 = ax1.twinx()
# # ax2.errorbar(fractions[1:] - .01, speed[:-1], fmt='bo', yerr=speed_std[:-1])
# ax2.plot(fractions2, speed_mean2, 'bo')
# ax2.fill_between(fractions2, speed_mean2 - speed_std2, speed_mean2 + speed_std2,
#                  alpha=.33, facecolor='blue')
# ax2.set_ylabel('Min. agent speed / ball speed', color='b')
# ax2.set_ylim([-0.1, 1.1])
# ax2.set_xlabel('Ball position')
# ax2.tick_params('y', colors='b')

# ax2 = ax1.twinx()
# ax2.plot(fractions, correct_predictions, 'bo')
# ax2.set_ylabel('Correct predictions', color='b')
# ax2.set_ylim([-.05, .95])
# ax2.tick_params('y', colors='b')
# plt.title('#samples: {}, window size: {}'.format(n_sampl, win_size))

# image dissimilarity
# plt.errorbar(fractions, img_diff, fmt='ro', yerr=img_diff_std)
# plt.ylabel('L2 image dissimilarity')
# plt.xlabel(xlabel)
plt.tight_layout()
plt.savefig('figures/' + fig_name + '.pdf', transparent=True)

# # show distance distribution
# fig, ax = plt.subplots()
# distances[10]
# n, bins, patches = ax.hist(distances[10], 50, facecolor='green', alpha=0.75)
# plt.tight_layout()
# plt.savefig('figures/histo.png')

# # agent performance
# fig, ax = plt.subplots()
# ax.set_xlabel('Agent speed / ball speed')
# ax.set_ylabel('Success rate')

# pot_str = 'pong'
# data_file = pot_str + 'w48agent_performance'
# with np.load('data/' + data_file + '.npz') as d:
#     success1 = d['successes']
#     dist1 = d['distances']
#     speeds1 = d['speeds']
# ax.plot(speeds1, success1, '-r', label='full history')

# data_file = pot_str + 'w8agent_performance'
# with np.load('data/' + data_file + '.npz') as d:
#     success2 = d['successes']
#     print(success2.max())
#     dist2 = d['distances']
#     speeds2 = d['speeds']
# ax.plot(speeds2, success2, '-g', label='window = 8')

# plt.legend()
# plt.savefig('figures/' + pot_str + '_agent_performance.png')

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

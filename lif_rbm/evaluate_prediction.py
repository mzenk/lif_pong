# script that should produce something like uncover plots later
from __future__ import division
import numpy as np
import cPickle
import sys
from scipy.ndimage import convolve1d
sys.path.insert(0, '../gibbs_rbm')
from rbm import RBM, CRBM
import matplotlib as mpl
mpl.use( "Agg" )
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sampling_interval = 10.   # ms


# different from method in generate_data.py: pooled along first axis
def pool_vector(vector, width, stride, mode='default'):
    vector = vector.astype(float)
    kernel = np.ones(width) / width
    filtered = convolve1d(vector, kernel, mode='constant', axis=0)
    # if no boundary effects should be visible
    if mode == 'valid':
        return filtered[width//2:-(width//2):stride]
    # startpoint is chosen such that the remainder of the division is
    # distributed evenly between left and right boundary
    start = ((vector.shape[1] - 1) % stride) // 2
    return filtered[start::stride]

# Load rbm and data
img_shape = (36, 48)
n_pixels = np.prod(img_shape)
n_pxls = np.prod(img_shape)
data_name = 'pong_var_start{}x{}'.format(*img_shape)
with np.load('../datasets/' + data_name + '.npz') as d:
    train_set, _, test_set = d[d.keys()[0]]
with open('../gibbs_rbm/saved_rbms/' + data_name + '_crbm.pkl', 'rb') as f:
    crbm = cPickle.load(f)

win_size = 48
sample_file = 'pong01_window{}_samples'.format(win_size)
n_imgs = 505
clamp_duration = 100.   # ms
clamped_fraction = np.arange(img_shape[1])/img_shape[1]
pool_width = int(clamp_duration / sampling_interval)

# predictions have shape (n_imgs, n_timesteps)
correct_predictions = np.zeros((n_imgs, img_shape[1]))
distances = np.zeros((n_imgs, img_shape[1]))
for i in range(n_imgs):
    try:
        with np.load('Data/' + sample_file + '{:03d}.npz'.format(i)) as d:
            samples = d[d.files[0]]
    except IOError:
        # deal with missing data
        correct_predictions[i] = np.nan
        distances[i] = np.nan

    target = np.average(np.arange(crbm.n_labels), weights=test_set[1][i])
    target_label = np.argmax(test_set[1][i])

    vis_samples = samples[:, :crbm.n_inputs]
    lab_samples = samples[:, crbm.n_inputs:crbm.n_visible]
    hid_samples = samples[:, crbm.n_visible:]

    average_label = pool_vector(lab_samples, pool_width, pool_width,
                                mode='valid')

    # compute distance and classification error
    pred_label = np.argmax(average_label, axis=1)
    average_label[np.all(average_label == 0, axis=1)] = np.ones(crbm.n_labels)
    pred_pos = np.average(
        np.tile(np.arange(crbm.n_labels), (len(average_label), 1)),
        weights=average_label, axis=1)
    pred_label = np.argmax(average_label, axis=1)

    correct_predictions[i] = 1*(pred_label == target_label)
    distances[i] = np.abs(pred_pos - target)

# missing data is nan
correct_mean = np.nanmean(correct_predictions, axis=0)
dist_mean = np.nanmean(distances, axis=0)
dist_std = np.nanstd(distances, axis=0)
# plotting
plt.figure()
# plt.subplot(121)
plt.errorbar(clamped_fraction, dist_mean, fmt='ro', yerr=dist_std)
plt.ylabel('Distance to correct label')
# plt.ylim([0, 3])
plt.xlabel('Clamped fraction')
plt.twinx()
plt.plot(clamped_fraction, correct_mean, 'bo')
plt.ylabel('Correct predictions')
# plt.ylim([0, 1])
plt.gca().spines['right'].set_color('blue')
plt.gca().spines['left'].set_color('red')
plt.title('Clamp duration: {}, window size: {}'.format(clamp_duration,
                                                       win_size))

# plt.subplot(122)
# plt.errorbar(clamped_fraction, img_diff, fmt='ro', yerr=img_diff_std)
# plt.ylabel('L2 image dissimilarity')
# plt.xlabel(xlabel)
# plt.tight_layout()
plt.savefig('Figures/uncover01_window{}.png'.format(win_size))

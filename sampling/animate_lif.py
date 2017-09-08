# animate lif examples
import numpy as np
import sys
from general_animation import make_animation
sys.path.insert(0, '../')
from utils.data_mgmt import get_data_path, make_figure_folder

img_shape = (36, 48)
n_pixels = np.prod(img_shape)
n_labels = 12
pot_str = 'pong'

# file with sampled states
sample_file = pot_str + '_win48_all_chunk000'
with np.load(get_data_path('lif_sampling') + sample_file + '.npz') as d:
    vis_samples = d['samples'][:, :np.prod(img_shape)]
    hid_samples = d['samples'][:, np.prod(img_shape) + n_labels:]
    # could compute activations of visible units here
    win_size = d['win_size']
    clamp_interval = d['samples_per_frame']
    sample_idx = d['data_idx']
    if len(vis_samples.shape) == 2:
        vis_samples = np.expand_dims(vis_samples, 0)
    print('Number of instances in file: ' + str(vis_samples.shape[0]))

# # Get predictions -----> this does not work yet; instead I computed the prediction in the call method
# pred_file = pot_str + '_win{}_prediction_incomplete'.format(win_size)
# with np.load(get_data_path('gibbs_sampling') + pred_file + '.npz') as d:
#     last_col = d['last_col']
#     pred_idx = d['data_idx']

# # filter the data to keep only those instances which are present in both
# # samples and predictions
# last_col = last_col[np.argsort(pred_idx)]
# vis_samples = vis_samples[np.argsort(sample_idx)]
# sample_idx = np.sort(sample_idx)
# pred_idx = np.sort(pred_idx)

# data_idx = np.intersect1d(sample_idx, pred_idx)
# last_col = last_col[np.where(np.in1d(pred_idx, data_idx))]
# vis_samples = vis_samples[np.where(np.in1d(sample_idx, data_idx))]

# # get predicted positions from last columns' mean activity
# predictions = np.zeros(last_col.shape[:-1])
# for i in range(len(last_col)):
#     predictions[i] = average_helper(img_shape[0], last_col[i])
# predictions = predictions.T

fig_name = 'test'
n = 1
for i in range(n):
    print('Making animation {} of {}'.format(i + 1, n))
    make_animation(make_figure_folder() + fig_name + str(i),
                   img_shape, win_size, vis_samples[i], paddle_len=0,
                   clamp_interval=clamp_interval, anim_interval=100./clamp_interval)

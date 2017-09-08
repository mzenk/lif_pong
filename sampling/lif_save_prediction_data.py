# helper script for lif sampling (parallelize chunks and call this afterwards)
# analogous to gibbs except that I store all binary samples for LIF, thus have
# to average here first
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import sys
from utils.data_mgmt import get_data_path
from utils import average_pool

if len(sys.argv) != 3:
    print('Please specify the arguments: pong/gauss, win_size')
    sys.exit()

pot_str = sys.argv[1]
win_size = int(sys.argv[2])
n_labels = 12
img_shape = (36, 48)
n_pxls = np.prod(img_shape)
data_name = pot_str + '_win{}_all_chunk'.format(win_size)

lab, last_col, data_idx = 0, 0, 0
data_path = get_data_path('lif_sampling')
save_name = data_path + '_'.join(data_name.split('_')[:2]) + '_prediction'

for i in np.arange(100):
    path = data_path + data_name + '{:03d}.npz'.format(i)
    if not os.path.exists(path):
        continue

    print('Processing chunk ' + str(i))
    with np.load(path) as d:
        assert win_size == d['win_size']
        chunk_samples = d['samples'].astype(float)
        chunk_vis = chunk_samples[..., :n_pxls + n_labels]
        chunk_idx = d['data_idx']
        n_samples = d['samples_per_frame']

    # first average pool on each chunk
    chunk_vis = average_pool(chunk_vis, n_samples, n_samples)

    # then compute prediction
    tmp_col = chunk_vis[..., :-n_labels].reshape(
                chunk_vis.shape[:-1] + img_shape)[..., -1]
    tmp_lab = chunk_vis[..., -n_labels:]

    lab = tmp_lab if i == 0 else np.vstack((lab, tmp_lab))
    last_col = tmp_col if i == 0 else np.vstack((last_col, tmp_col))
    data_idx = chunk_idx if i == 0 else np.concatenate((data_idx, chunk_idx))

# save data (averaged samples for label units and last column)
np.savez_compressed(save_name, label=lab, last_col=last_col, data_idx=data_idx)

# helper script for lif sampling (parallelize chunks and call this afterwards)
# analogous to gibbs except that I store all binary samples for LIF, thus have
# to average here first
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import sys
from lif_pong.utils.data_mgmt import get_data_path
from lif_pong.utils import average_pool

if len(sys.argv) < 3:
    print('Required arguments: data_name, '
          '(data generation) script name, [n_files]')
    sys.exit()

data_name = sys.argv[1]
script_name = sys.argv[2]

img_shape = (36, 48)
n_labels = img_shape[0]//3
n_pxls = np.prod(img_shape)

lab, last_col, data_idx = 0, 0, 0
data_path = get_data_path(script_name)
data_list = [f for f in os.listdir(data_path)
             if data_name + '_chunk' in f]
save_name = data_path + data_name + '_prediction'
if len(sys.argv) == 4:
    n_data = int(sys.argv[3])
    if n_data > len(data_list):
        print('Warning: Less data files exist than specified.')
    data_list = data_list[:n_data]
    save_name = data_path + data_name + '_N{}_prediction'.format(n_data)

for i, f in enumerate(data_list):
    path = data_path + f
    if not os.path.exists(path):
        print('!!!\nFile does not exit: {}\n!!!'.format(path))
        continue

    print('Processing chunk ' + str(i))
    with np.load(path) as d:
        chunk_samples = d['samples'].astype(float)
        chunk_vis = chunk_samples[..., :n_pxls + n_labels]
        chunk_idx = d['data_idx']
        n_samples = d['samples_per_frame']
        win_size = d['win_size']
        winpos = np.arange(win_size + 1)
        if 'winpos' in d.keys():
            winpos = d['winpos']

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
np.savez_compressed(save_name, label=lab, last_col=last_col, data_idx=data_idx,
                    winpos=winpos)

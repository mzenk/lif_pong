# helper script for gibbs sampling (parallelize chunks and call this afterwards)
from __future__ import division
from __future__ import print_function
import numpy as np
from lif_pong.utils.data_mgmt import get_data_path
import os
import sys

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
save_name = data_path + data_name + '_prediction'
data_list = [f for f in os.listdir(data_path)
             if data_name + '_avg_chunk' in f]
if len(sys.argv) == 4:
    n_data = int(sys.argv[3])
    save_name = data_path + data_name + '_N{}_prediction'.format(n_data)

n_instances = 0
for i, f in enumerate(data_list):
    path = data_path + f
    if not os.path.exists(path):
        print('!!!\nFile does not exit: {}\n!!!'.format(path))
        continue

    print('Processing chunk ' + str(i))
    with np.load(path) as d:
        win_size = d['win_size']
        chunk_vis = d['vis']
        chunk_idx = d['data_idx']
    n_instances += len(chunk_idx)
    tmp_col = chunk_vis[..., :-n_labels].reshape(
                chunk_vis.shape[:-1] + img_shape)[..., -1]
    tmp_lab = chunk_vis[..., -n_labels:]

    lab = tmp_lab if i == 0 else np.vstack((lab, tmp_lab))
    last_col = tmp_col if i == 0 else np.vstack((last_col, tmp_col))
    data_idx = chunk_idx if i == 0 else np.concatenate((data_idx, chunk_idx))

    # no need to load more than necessary
    if n_instances > n_data:
        break

# restrict number of images in prediction file
if n_data != -1:
    lab = lab[:n_data]
    last_col = last_col[:n_data]
    data_idx = data_idx[:n_data]

# save data (averaged samples for label units and last column)
np.savez_compressed(save_name, label=lab, last_col=last_col, data_idx=data_idx)
print('Saved prediction data {} ({} instances)'.format(save_name, len(lab)))

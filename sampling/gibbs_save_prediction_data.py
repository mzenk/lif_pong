# helper script for gibbs sampling (parallelize chunks and call this afterwards)
from __future__ import division
from __future__ import print_function
import numpy as np
from lif_pong.utils.data_mgmt import get_data_path
import os
import sys

if len(sys.argv) < 4:
    print('Required arguments: pong/gauss, win_size, '
          'script (data generation) name [name_modifier]')
    sys.exit()

pot_str = sys.argv[1]
win_size = int(sys.argv[2])
script_name = sys.argv[3]
if len(sys.argv) == 5:
    modifier = '_' + str(sys.argv[4])
else:
    modifier = ''

img_shape = (36, 48)
n_labels = img_shape[0]//3
data_name = pot_str + '_win{}{}_avg_chunk'.format(win_size, modifier)

lab, last_col, data_idx = 0, 0, 0
data_path = get_data_path(script_name)
save_name = data_path + '_'.join(data_name.split('_')[:2]) + modifier + \
    '_prediction'

n_instances = 0
for i in np.arange(150):
    path = data_path + data_name + '{:03d}.npz'.format(i)
    if not os.path.exists(path):
        continue
    print('Processing chunk ' + str(i))
    with np.load(path) as d:
        assert win_size == d['win_size']
        chunk_vis = d['vis']
        chunk_idx = d['data_idx']
    n_instances += len(chunk_idx)
    tmp_col = chunk_vis[..., :-n_labels].reshape(
                chunk_vis.shape[:-1] + img_shape)[..., -1]
    tmp_lab = chunk_vis[..., -n_labels:]

    lab = tmp_lab if i == 0 else np.vstack((lab, tmp_lab))
    last_col = tmp_col if i == 0 else np.vstack((last_col, tmp_col))
    data_idx = chunk_idx if i == 0 else np.concatenate((data_idx, chunk_idx))

# save data (averaged samples for label units and last column)
np.savez_compressed(save_name, label=lab, last_col=last_col, data_idx=data_idx)
print('Saved prediction data {} ({} instances)'.format(save_name, n_instances))

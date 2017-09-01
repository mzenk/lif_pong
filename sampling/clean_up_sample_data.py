# clean-up disk after saving gibbs_sampling data
from __future__ import division
from __future__ import print_function
import numpy as np
from util import get_data_path
import os
import sys

if len(sys.argv) != 3:
    print('Please specify the arguments: pong/gauss, win_size')
    sys.exit()

pot_str = sys.argv[1]
win_size = int(sys.argv[2])
n_labels = 12
img_shape = (36, 48)
data_name = pot_str + '_win{}_avg_chunk'.format(win_size)
data_path = get_data_path('gibbs_sampling')

for i in np.arange(100):
    path = data_path + data_name + '{:03d}.npz'.format(i)
    # delete most of the files
    if os.path.exists(path) and i % 10 != 0:
        os.remove(path)
        print('Removed ' + path)

from __future__ import division
from __future__ import print_function
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

def get_progress(logfile):
    train_classrate = []
    valid_classrate = []
    nepoch = []
    with open(logfile) as trainlog:
        for line in trainlog:
            parts = line.split(':')
            if len(parts) < 2:
                continue
            if 'training set' in parts[-2]:
                train_classrate.append(float(parts[-1]))
            if 'validation set' in parts[-2]:
                valid_classrate.append(float(parts[-1]))
            if 'Epoch' in parts[-1]:
                nepoch.append(int(parts[-1].split(' ')[-1]))
    if len(nepoch) < len(train_classrate):
        nepoch.insert(0, 0)
    return np.array(nepoch), np.array(train_classrate), np.array(valid_classrate)

plt.rcParams['font.size'] = 16
fig, ax = plt.subplots()
basefolder = os.path.expanduser('~/workspace/experiment/simulations')
paths = ['TrainThickPongFine/15_0.0001_0.05/train.log', 'TrainLW5GaussFine02/10_0.001_0.05/train.log']
labels = ['Pong', 'Hill']
for i, path in enumerate(paths):
    nepoch, train_cr, valid_cr = get_progress(os.path.join(basefolder,path))
    ax.plot(nepoch[1:], train_cr[1:], 'o', label=labels[i])
    # ax.plot(nepoch, 1 - valid_cr, '.', label=labels[i])

ax.legend(loc=4)
ax.set_ylim(top=1.)
ax.set_xlabel('No. epochs')
ax.set_ylabel('Classification rate')
fig.savefig('test.pdf')

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


fig, ax = plt.subplots()
paths = ['pong_train.log', 'gauss_train.log']
labels = ['Pong', 'Hill']
for i, path in enumerate(paths):
    nepoch, train_cr, valid_cr = get_progress(path)
    ax.plot(nepoch[1:], 1 - train_cr[1:], 'o', label=labels[i])
    # ax.plot(nepoch, 1 - valid_cr, '.', label=labels[i])

ax.legend()
ax.set_ylim(bottom=0.)
ax.set_xlabel('No. epochs')
ax.set_ylabel('Classification error rate')
fig.savefig('test.pdf')

import os
import cPickle
import sys
from rbm import RBM, CRBM

path = os.path.expanduser('~') + '/Projects/Pong/shared_data/saved_rbms/'
with open(path + 'pong_var_start36x48_crbm' + '.pkl', 'rb') as f:
    rbm = cPickle.load(f)

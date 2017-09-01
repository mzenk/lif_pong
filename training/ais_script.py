# script for several ais runs
import numpy as np
import rbm_ais
import dbm_ais
from rbm import RBM, CRBM
from dbm import DBM, CDBM
import cPickle, gzip
from util import to_1_of_c


# fnames = ['mnist_disc500_rbm', 'mnist_disc500noreg_rbm',
#           'mnist_disc500mom_rbm', 'mnist_disc500momstep_rbm']
fnames = ['mnist_gen_rbm']

# Load MNIST
f = gzip.open('datasets/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

for name in fnames:
    with open('saved_rbms/' + name + '.pkl', 'rb') as f:
            rbm = cPickle.load(f)
    print('Processing ' + name + '...')
    test_data = test_set[0]
    test_targets = test_set[1]

    if type(rbm) is RBM:
        train_data = train_set[0]
        rbm_ais.run_ais(rbm, test_data, train_data=train_data, n_runs=300)
    if type(rbm) is CRBM:
        test_data = np.hstack((test_data, to_1_of_c(test_targets, 10)))
        train_data = np.hstack((train_set[0], to_1_of_c(train_set[1], 10)))
        rbm_ais.run_ais(rbm, test_data, train_data=train_data)
    if type(rbm) is DBM:
        dbm_ais.run_ais(rbm, test_data)
    if type(rbm) is CDBM:
        dbm_ais.run_ais(rbm, test_data, targets=test_targets)

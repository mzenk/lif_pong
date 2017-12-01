from __future__ import division
from __future__ import print_function
import numpy as np
import cPickle
import gzip
from lif_pong.utils import to_1_of_c
import sys
import time
from rbm import RBM, CRBM
from dbm import DBM, CDBM
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Load MNIST
with gzip.open('../shared_data/datasets/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = np.load(f)
n_pixels = train_set[0].shape[1]

training_params = {
    'n_epochs': 20,
    'batch_size': 20,
    'lrate': .05,
    'cd_steps': 1,
    'persistent': True,
    'momentum': 0.5,
    'weight_cost': .001,
    'cast': True,
}

mf_params = {
    'n_epochs': 20,
    'batch_size': 100,
    'lrate': .001,
    'cd_steps': 5,
    'momentum': 0.5,
    'weight_cost': .0002
}

# initialize visible biases as in Hinton's guide
pj = np.average(train_set[0], axis=0)
pj[pj == 0] = 1e-5
pj[pj == 1] = 1 - 1e-5
bias_init = np.log(pj / (1 - pj))

# seed = 68467324
if sys.argv[1] == 'gen':
    # ----- test generative RBM -----

    print('Training generative RBM on MNIST...')
    # whole MNIST
    n_hidden = 500
    my_rbm = RBM(n_pixels, n_hidden, vbias=bias_init)

    start = time.time()
    my_rbm.train(train_set[0], valid_set=None,
                 filename='./log/mnist_gen_log.txt', **training_params)

    print('Total training time: {:.1f} min'.format((time.time() - start)/60))
    print('log-PL of training set: '
          '{}'.format(my_rbm.compute_logpl(train_set[0])))

    # Save crbm for later inspection
    my_rbm.save('../shared_data/saved_rbms/mnist_gen_rbm.pkl')

if sys.argv[1] == 'dis':
    # ----- test ClassRBM -----
    # whole MNIST
    train_input = np.hstack((train_set[0], to_1_of_c(train_set[1], 10)))
    valid_input = np.hstack((valid_set[0], to_1_of_c(valid_set[1], 10)))

    n_hidden = 500
    crbm = CRBM(n_inputs=n_pixels, n_hidden=n_hidden, n_labels=10,
                input_bias=bias_init)
    print('Training Classifying RBM on MNIST...')
    start = time.time()
    crbm.train(train_input, valid_set=None,
               filename='./log/mnist_crbm_log.txt', **training_params)
    print('Total training time: {:.1f} min'.format((time.time() - start)/60))

    prediction = crbm.classify(test_set[0])
    test_performance = np.average(prediction == test_set[1])
    print("Correct classifications on test set: " + str(test_performance))

    # Save crbm for later inspection
    crbm.save('../shared_data/saved_rbms/mnist_disc_rbm.pkl')

if sys.argv[1] == 'deep':
    # ----- test (C)DBM -----

    layers = [n_pixels, 400, 600]
    fn = 'mnist_cdbm'
    print('Training DBM {} on MNIST...'.format(layers))

    my_dbm = CDBM(layers, labels=10, vbias_init=bias_init)
    # my_dbm = DBM(layers, vbias_init=bias_init)

    start = time.time()
    my_dbm.train(train_set[0], to_1_of_c(train_set[1], 10), valid_set=None,
                 filename='log/' + fn + '_log.txt', **training_params)
    # my_dbm.train(train_set[0], valid_set=None,
    #              filename='log/' + fn + '_log.txt', **training_params)

    print('Total training time: {:.1f} min'.format((time.time() - start)/60))

    # Save DBM for later inspection
    with open('saved_rbms/' + fn + '.pkl', 'wb') as output:
        cPickle.dump(my_dbm, output, cPickle.HIGHEST_PROTOCOL)

if sys.argv[1] == 'mf':
    fn = 'mnist_cdbm'
    with open('saved_rbms/' + fn + '.pkl', 'rb') as f:
        my_dbm = cPickle.load(f)

    print('Training DBM {} on MNIST with MF...'.format(my_dbm.hidden_layers))

    start = time.time()
    my_dbm.train_mf(train_set[0], to_1_of_c(train_set[1], 10),
                    valid_set=valid_set, filename='log/' + fn + '_mf_log.txt',
                    **mf_params)

    print('Total training time: {:.1f} min'.format((time.time() - start)/60))
    # Save DBM for later inspection
    with open('saved_rbms/' + fn + '_mf.pkl', 'wb') as output:
        cPickle.dump(my_dbm, output, cPickle.HIGHEST_PROTOCOL)

    # Save monitoring quantities as diagram
    # plt.figure()
    # plt.plot(t, pl, '.')
    # plt.xlabel('update steps')
    # plt.ylabel('log PL')
    # plt.savefig('./figures/logpl.png')

    # plt.figure()
    # plt.plot(t, df, '.')
    # plt.xlabel('update steps')
    # plt.ylabel('F_valid - F_train')
    # plt.savefig('./figures/df.png')

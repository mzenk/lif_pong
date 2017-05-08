from __future__ import division
from __future__ import print_function
import numpy as np
import cPickle
import gzip
from util import to_1_of_c, tile_raster_images
import sys
import time
from rbm import RBM, CRBM
from dbm import DBM
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Load MNIST
f = gzip.open('datasets/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
n_pixels = train_set[0].shape[1]

training_params = {
    'n_epochs': 4,
    'batch_size': 10,
    'lrate': .01,
    'cd_steps': 1,
    'persistent': True,
    'momentum': 0.,
    'cast': False,
}

np.random.seed(68467324)
if sys.argv[1] == 'gen':
    # ----- test generative RBM -----

    print('Training generative RBM on MNIST...')
    # whole MNIST
    pj = np.average(train_set[0], axis=0)
    pj[pj == 0] = 1e-5
    pj[pj == 1] = 1 - 1e-5
    bias_init = np.log(pj / (1 - pj))
    my_rbm = RBM(n_pixels, 300, vbias=bias_init)

    start = time.time()
    my_rbm.train(train_set[0], valid_set=valid_set[0],
                 filename='mnist_gen_log.txt', **training_params)

    print('Total training time: {:.1f} min'.format((time.time() - start)/60))
    print('log-PL of training set: '
          '{}'.format(my_rbm.compute_logpl(train_set[0])))

    # Save crbm for later inspection
    with open('saved_rbms/mnist_gen_rbm.pkl', 'wb') as output:
        cPickle.dump(my_rbm, output, cPickle.HIGHEST_PROTOCOL)

if sys.argv[1] == 'dis':
    # ----- test ClassRBM -----

    # whole MNIST
    train_input = np.concatenate((train_set[0], to_1_of_c(train_set[1], 10)),
                                 axis=1)
    valid_input = np.concatenate((valid_set[0][:1000],
                                 to_1_of_c(valid_set[1][:1000], 10)), axis=1)

    crbm = CRBM(n_inputs=n_pixels, n_hidden=300,
                n_labels=10)
    print('Training Classifying RBM on MNIST...')
    start = time.time()
    crbm.train(train_input, valid_set=valid_input,
               **training_params)
    print('Total training time: {:.1f} min'.format((time.time() - start)/60))

    prediction = crbm.classify(test_set[0])
    test_performance = np.average(prediction == test_set[1])
    print("Correct classifications on test set: " + str(test_performance))

    # Save crbm for later inspection
    with open('saved_rbms/mnist_disc_rbm.pkl', 'wb') as output:
        cPickle.dump(crbm, output, cPickle.HIGHEST_PROTOCOL)

if sys.argv[1] == 'deep':
    # ----- test DBN -----

    layers = [n_pixels, 100, 200]
    print('Training DBM {} on MNIST...'.format(layers))
    pj = np.average(train_set[0], axis=0)
    pj[pj == 0] = 1e-5
    pj[pj == 1] = 1 - 1e-5
    bias_init = np.log(pj / (1 - pj))
    my_dbm = DBM(layers, vbias_init=bias_init)

    start = time.time()
    my_dbm.train(train_set[0], valid_set=None,
                 filename='mnist_dbm_log.txt', **training_params)

    print('Total training time: {:.1f} min'.format((time.time() - start)/60))

    # Save DBM for later inspection
    with open('saved_rbms/mnist_dbm.pkl', 'wb') as output:
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

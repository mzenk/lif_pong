from __future__ import division
from __future__ import print_function
import numpy as np
from rbm import RBM, CRBM
from dbm import DBM, CDBM
import time
import cPickle
from util import to_1_of_c, tile_raster_images
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


# Load Pong data
img_shape = (36, 48)
save = True
fname = 'pong_var_start{}x{}'.format(*img_shape)
# fname = 'label1_fixed'
with np.load('datasets/' + fname + '.npz') as d:
    train_set, valid_set, test_set = d[d.keys()[0]]

n_pxls = train_set[0].shape[1]
n_labels = train_set[1].max() + 1
assert np.prod(img_shape) == n_pxls

training_params = {
    'n_epochs': 15,
    'batch_size': 20,
    'lrate': .05,
    'cd_steps': 1,
    'persistent': True,
    'cast': False,
    'momentum': 0.5
}

if sys.argv[1] == 'gen':
    print('Training generative RBM on Pong...')
    # discard labels
    train_set = train_set[0]
    valid_set = valid_set[0]

    # initialize biases like in Hinton's guide
    pj = np.average(train_set, axis=0)
    pj[pj == 0] = 1e-5
    pj[pj == 1] = 1 - 1e-5
    bias_init = np.log(pj / (1 - pj))
    my_rbm = RBM(train_set.shape[1], n_hidden=200, vbias=bias_init)

    start = time.time()
    my_rbm.train(train_set, valid_set=valid_set,
                 filename='log/pong_gen_log.txt', **training_params)

    print('Total training time: {:.1f} min'.format((time.time() - start)/60))

    if save:
        # Save crbm for later inspection
        with open('saved_rbms/' + fname + '_rbm.pkl', 'wb') as output:
            cPickle.dump(my_rbm, output, cPickle.HIGHEST_PROTOCOL)

if sys.argv[1] == 'dis':
    print('Training discriminative RBM on Pong on  {}'
          ' instances...'.format(train_set[0].shape[0]))

    train_wlabel = np.concatenate((train_set[0],
                                  to_1_of_c(train_set[1], n_labels)), axis=1)
    valid_set = np.concatenate((valid_set[0],
                                to_1_of_c(valid_set[1], n_labels)), axis=1)

    # initialize biases like in Hinton's guide
    pj = np.average(train_wlabel, axis=0)
    pj[pj == 0] = 1e-5
    pj[pj == 1] = 1 - 1e-5
    bias_init = np.log(pj / (1 - pj))
    # for r in [.001, .01, .1]:
    #     training_params['lrate'] = r
    #     print(r)
    my_rbm = CRBM(n_inputs=n_pxls,
                  n_hidden=500,
                  n_labels=n_labels,
                  vbias=bias_init)

    start = time.time()
    my_rbm.train(train_wlabel, valid_set=valid_set,
                 filename='log/pong_dis_log.txt', **training_params)

    print('Training took {:.1f} min'.format((time.time() - start)/60))

    if save:
        # Save crbm for later inspection, the training parameters should be
        # recorded elsewhere!
        with open('saved_rbms/' + fname + '_crbm.pkl', 'wb') as output:
            cPickle.dump(my_rbm, output, cPickle.HIGHEST_PROTOCOL)

if sys.argv[1] == 'deep':
    # ----- test DBN/DBM -----
    layers = [n_pxls, 200, 300]
    fn = 'pong_cdbm'
    print('Training DBM {} on Pong...'.format(layers))

    # initialize biases like in Hinton's guide
    pj = np.average(train_set[0], axis=0)
    pj[pj == 0] = 1e-5
    pj[pj == 1] = 1 - 1e-5
    bias_init = np.log(pj / (1 - pj))

    # for r in [.001, .01, .1]:
    #     training_params['lrate'] = r
    #     print(r)
    my_dbm = CDBM(layers, labels=n_labels, vbias_init=bias_init)
    # my_dbm = DBM(layers, vbias_init=bias_init)

    start = time.time()
    my_dbm.train(train_set[0], to_1_of_c(train_set[1], n_labels),
                 filename='log/' + fn + '_log.txt', **training_params)

    print('Total training time: {:.1f} min'.format((time.time() - start)/60))

    # Save DBM for later inspection
    with open('saved_rbms/' + fn + '.pkl', 'wb') as output:
        cPickle.dump(my_dbm, output, cPickle.HIGHEST_PROTOCOL)

if sys.argv[1] == 'mf':
    with open('saved_rbms/pong_cdbm.pkl', 'rb') as f:
        my_dbm = cPickle.load(f)
    fn = 'pong_mf'
    mf_params = {
        'n_epochs': 20,
        'batch_size': 10,
        'lrate': .001,
        'cd_steps': 5,
        'momentum': 0.5,
    }

    print('Training DBM {} on Pong with MF...'.format(my_dbm.hidden_layers))

    start = time.time()
    my_dbm.train_mf(train_set[0], to_1_of_c(train_set[1], n_labels),
                    valid_set=valid_set, filename='log/' + fn + '_log.txt',
                    **mf_params)

    print('Total training time: {:.1f} min'.format((time.time() - start)/60))
    # Save DBM for later inspection
    with open('saved_rbms/' + fn + '.pkl', 'wb') as output:
        cPickle.dump(my_dbm, output, cPickle.HIGHEST_PROTOCOL)

    # # visualize labels
    # dunno = np.concatenate((train_set[0].reshape((train_set[0].shape[0], img_shape[0], 3)),
    #                         np.repeat(-to_1_of_c(train_set[1], n_labels), 3, axis=1).reshape(train_set[0].shape[0],
    #                              img_shape[0], 1)), axis=2).reshape(train_set[0].shape[0], img_shape[0]*4)
    # samples = tile_raster_images(dunno[:10],
    #                              img_shape=(img_shape[0], 4),
    #                              tile_shape=(1, 10),
    #                              tile_spacing=(1, 3),
    #                              scale_rows_to_unit_interval=False,
    #                              output_pixel_vals=False)

    # plt.figure()
    # plt.imshow(samples, interpolation='Nearest', cmap='gray', origin='lower')
    # plt.title('Examples for trajectory endings + labels')
    # plt.colorbar()
    # plt.savefig('./figures/label_problem.png')

from __future__ import division
from __future__ import print_function
import numpy as np
from bm import Rbm, ClassRbm
import time
import cPickle
from util import to_1_of_c, tile_raster_images
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


# Load Pong data
img_shape = (36, 1)
fname = 'pong_fixed_start{}x{}'.format(*img_shape)
fname = 'last_columns'
with np.load('datasets/' + fname + '.npz') as d:
    train_set, valid_set, test_set = d[d.keys()[0]]

assert np.prod(img_shape) == train_set[0].shape[1]

# # inspect data
# samples = tile_raster_images(train_set[0][:16],
#                              img_shape=img_shape,
#                              tile_shape=(4, 4),
#                              tile_spacing=(1, 1),
#                              scale_rows_to_unit_interval=True,
#                              output_pixel_vals=False)

# plt.figure()
# plt.imshow(samples, interpolation='Nearest', cmap='gray', origin='lower')
# plt.savefig(fname + 'samples.png')

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
    my_rbm = Rbm(train_set.shape[1],
                 n_hidden=200,
                 n_epochs=2,
                 batch_size=10,
                 rate=.06,
                 bv=bias_init)

    start = time.time()

    t, df, pl = my_rbm.train(train_set, cd_steps=5, persistent=True,
                             valid_set=valid_set)

    print('Total training time: {:.1f} min'.format((time.time() - start)/60))

    # Save monitoring quantities as diagram
    plt.figure()
    plt.plot(t, pl, '.')
    plt.xlabel('update steps')
    plt.ylabel('log PL')
    plt.savefig('logpl.png')

    plt.figure()
    plt.plot(t, df, '.')
    plt.xlabel('update steps')
    plt.ylabel('F_valid - F_train')
    plt.savefig('df.png')

    # Save crbm for later inspection
    with open('saved_rbms/' + fname + '_rbm.pkl', 'wb') as output:
        cPickle.dump(my_rbm, output, cPickle.HIGHEST_PROTOCOL)

if sys.argv[1] == 'dis':
    print('Training discriminative RBM on Pong on  {}'
          ' instances...'.format(train_set[0].shape[0]))

    n_labels = train_set[1].max() + 1
    train_wlabel = np.concatenate((train_set[0],
                                  to_1_of_c(train_set[1], n_labels)), axis=1)
    valid_set = np.concatenate((valid_set[0],
                                to_1_of_c(valid_set[1], n_labels)), axis=1)

    my_rbm = ClassRbm(n_inputs=train_set[0].shape[1],
                      n_hidden=100,
                      n_labels=n_labels,
                      n_epochs=30,
                      batch_size=10,
                      rate=.8)

    start = time.time()
    t, df, pl = my_rbm.train(train_wlabel, cd_steps=5, persistent=True,
                             cast=False, valid_set=valid_set)

    print('Total training time: {:.1f} min'.format((time.time() - start)/60))

    # Save monitoring quantities as diagram
    plt.figure()
    plt.plot(t, pl, '.')
    plt.xlabel('update steps')
    plt.ylabel('log PL')
    plt.savefig('logpl.png')

    plt.figure()
    plt.plot(t, df, '.')
    plt.xlabel('update steps')
    plt.ylabel('F_valid - F_train')
    plt.savefig('df.png')

    # Save crbm for later inspection, the training parameters
    # (rate, batch_size) are also accessible
    with open('saved_rbms/' + fname + '_crbm.pkl', 'wb') as output:
        cPickle.dump(my_rbm, output, cPickle.HIGHEST_PROTOCOL)

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
    # plt.savefig('label_problem.png')

from __future__ import division
from __future__ import print_function
import numpy as np
import cPickle
import gzip
from util import boltzmann, bin_to_dec, dec_to_bin, compute_dkl, to_1_of_c
import sys
import time
from bm import Rbm, ClassRbm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


np.random.seed(68467324)
if sys.argv[1] == 'basics':
    # ------ Test the sampling -------

    # rbm
    nv = 3
    nh = 2
    dim = nv + nh
    w_small = np.random.rand(nv, nh)
    test = Rbm(nv, nh, w_small, np.zeros(nv), np.zeros(nh))
    w = np.concatenate((np.concatenate((np.zeros((nv, nv)), w_small), axis=1),
                       np.concatenate((w_small.T, np.zeros((nh, nh))), axis=1)),
                       axis=0)

    b = np.zeros(dim)
    # evaluate target distribution
    target = np.zeros(2**dim)
    nbits = np.floor(np.log2(2**dim - 1)) + 1
    for i in range(2**dim):
        target[i] = boltzmann(dec_to_bin(i, nbits), w, b)
    p_target = target/np.sum(target)

    # run bm and compare histograms
    n_samples = int(1e5)
    samples = test.draw_samples_ast(n_samples)[1000:, :]

    decimals = bin_to_dec(samples)

    # plt.figure()
    # h = plt.hist(decimals, bins=np.arange(0, 2**dim + 1, 1), normed=True,
    #              alpha=0.5, label='sampled', color='g', align='mid',
    #              rwidth=.5, log=True)
    # plt.bar(np.arange(0, 2**dim, 1), p_target, width=.5,
    #         alpha=0.5, label='target', color='b')
    # plt.legend(loc='upper left')
    # plt.savefig("histo.png")
    # # save
    # np.savetxt('sampling.dat', np.vstack((p_target, h[0])).T,
    #            header='target_values sample_counts')

    # calculate dkl
    ns = np.logspace(1, np.log10(samples.shape[0]), 100)
    dkl = np.zeros_like(ns)
    for i, n in enumerate(ns):
        dkl[i] = compute_dkl(samples[:int(n), :], p_target)

    plt.figure()
    plt.loglog(ns, dkl, label='ast')
    samples = test.draw_samples(n_samples, n_chains=1)[1000:, :]
    decimals = bin_to_dec(samples)
    ns = np.logspace(1, np.log10(samples.shape[0]), 100)
    dkl = np.zeros_like(ns)
    for i, n in enumerate(ns):
        dkl[i] = compute_dkl(samples[:int(n), :], p_target)
    plt.loglog(ns, dkl, 'g', label='gibbs')
    plt.legend()
    plt.savefig('dkl.png')

    # ----- test training by learning a known (simple) distribution -----
    # nv = 3
    # nh = 2
    # w_small = 2*(np.random.beta(1.5, 1.5, (nv,nh)) - .5)
    # myrbm = Rbm(nv, nh)

    # # generate samples from true distribution and train bm
    # train = Rbm(nv, nh, w_small, np.zeros(nv), np.zeros(nh))
    # train_samples = train.draw_samples(int(1e4))
    # v_train_samples = train_samples[1000:,:nv]

    # valid = train.draw_samples(2000)[1500:,:nv]
    # myrbm.train(v_train_samples, valid_set=valid, cast=True)

    # # run bm and compare histograms
    # samples = myrbm.draw_samples(int(1e5))

    # decimals = bin_to_dec(samples[:, :nv])
    # decimals_train = bin_to_dec(v_train_samples)
    # plt.figure()
    # h1 = plt.hist(decimals, bins=np.arange(0, 2**nv + 1, 1), normed=True,
    #               alpha=0.5, label='sampled', color='g', align='mid',
    #               rwidth=0.5, log=True)
    # h2 = plt.hist(decimals_train, bins=np.arange(0, 2**nv + 1, 1), normed=True,
    #               alpha=0.5, label='sampled', color='b', align='mid',
    #               rwidth=0.5, log=True)
    # plt.legend(loc='upper left')
    # plt.savefig('test.png')
    # plt.close()

    # p_target = np.histogram(decimals_train, bins=np.arange(0, 2**nv + 1, 1),
    #                         normed=True)[0]

    # print "DKL = " + str(compute_dkl(samples[:,:nv], p_target))


if sys.argv[1] == 'gen':
    # ----- test generative RBM -----

    # Load MNIST
    f = gzip.open('datasets/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    print('Training generative RBM on MNIST...')
    # whole MNIST
    pj = np.average(train_set[0], axis=0)
    pj[pj == 0] = 1e-5
    pj[pj == 1] = 1 - 1e-5
    bias_init = np.log(pj / (1 - pj))
    my_rbm = Rbm(train_set[0].shape[1], 300,
                 n_epochs=5,
                 batch_size=10,
                 rate=.08,
                 bv=bias_init)

    start = time.time()
    t, df, pl = my_rbm.train(train_set[0], cast=False, persistent=True,
                             valid_set=valid_set[0])

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
    with open('mnist_gen_rbm.pkl', 'wb') as output:
        cPickle.dump(my_rbm, output, cPickle.HIGHEST_PROTOCOL)

if sys.argv[1] == 'dis':
    # ----- test ClassRBM -----

    # Load MNIST
    f = gzip.open('datasets/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    # whole MNIST
    train_input = np.concatenate((train_set[0], to_1_of_c(train_set[1], 10)),
                                 axis=1)
    valid_input = np.concatenate((valid_set[0][:1000],
                                 to_1_of_c(valid_set[1][:1000], 10)), axis=1)

    crbm = ClassRbm(n_inputs=train_set[0].shape[1],
                    n_hidden=300,
                    n_labels=10,
                    batch_size=10,
                    rate=.05,
                    n_epochs=4)
    print('Training ClassRBM...')
    start = time.time()
    t, df, pl = crbm.train(train_input, cd_steps=5, persistent=True,
                           valid_set=valid_input)
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

    prediction = crbm.classify(test_set[0])
    test_performance = np.average(prediction == test_set[1])
    print("Correct classifications: " + str(test_performance))

    # Save crbm for later inspection
    with open('saved_rbms/mnist_disc_rbm.pkl', 'wb') as output:
        cPickle.dump(crbm, output, cPickle.HIGHEST_PROTOCOL)

    # # or small data set
    # small_train_set = np.load("small_train.npy")
    # small_test_set = np.load("small_test.npy")
    # small_valid_set = np.load("small_valid.npy")

    # c = 10
    # small_input = small_train_set[train_set[1] < c]
    # small_labels = train_set[1][train_set[1] < c]
    # train_vis = np.concatenate((small_input, to_1_of_c(small_labels, c)),
    #                            axis=1)

    # small_test_input = small_test_set[test_set[1] < c, :]
    # small_test_labels = test_set[1][test_set[1] < c]

    # small_valid_input = small_valid_set[valid_set[1] < c, :]
    # small_valid_labels = valid_set[1][valid_set[1] < c]
    # combined_valid = np.concatenate((small_valid_input,
    #                                 to_1_of_c(small_valid_labels, c)), axis=1)

    # crbm = ClassRbm(small_input.shape[1], 300, c)
    # crbm.train(train_vis, n_steps=1, persistent=True, valid_set=combined_valid)

    # prediction = crbm.classify(small_test_input)
    # print("Correct classifications: " +
    #       str(np.average(prediction == small_test_labels)))

    # # Save crbm for later inspection
    # with open('saved_rbms/small_mnist_crbm.pkl', 'wb') as output:
    #     cPickle.dump(crbm, output, cPickle.HIGHEST_PROTOCOL)

    # # xor dataset
    # training_v = np.random.randint(2, size=(1000, 2))
    # training_l = np.bitwise_xor(training_v[:, 0], training_v[:, 1])
    # training_data = np.concatenate((training_v, to_1_of_c(training_l, 2)),
    #                                axis=1)

    # test_v = np.random.randint(2, size=(300, 2))
    # test_l = np.bitwise_xor(test_v[:, 0], test_v[:, 1])

    # crbm = ClassRbm(n_inputs=2,
    #                 n_hidden=5,
    #                 n_labels=2,
    #                 batch_size=10,
    #                 rate=1.,
    #                 n_epochs=1)
    # crbm.train(training_data)

    # prediction = crbm.classify(test_v)
    # print("Correct classifications: " + str(np.average(prediction == test_l)))

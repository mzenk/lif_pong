from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import cPickle, gzip
import itertools
from util import to_1_of_c
from rbm import CRBM


# numerical stability
def logsum(x, axis=0):
    alpha = np.max(x, axis=axis) - np.log(sys.float_info.max)/2
    # alpha = np.max(x, axis=axis)
    return alpha + np.log(np.sum(np.exp(x - alpha), axis=axis))


def logdiff(x, axis=0):
    alpha = np.max(x, axis=axis) - np.log(sys.float_info.max)/2
    # alpha = np.max(x, axis=axis)
    return alpha + np.log(np.diff(np.exp(x - alpha), axis=axis)).squeeze()


def compute_partition_sum(rbm):
    # visible units are too many to sum over
    h_all = np.array(list(itertools.product([0, 1], repeat=rbm.n_hidden)))
    log_ph = np.zeros(0)
    n_chunks = np.ceil(h_all.shape[0] * rbm.n_visible // 1e8)
    for h_chunk in np.array_split(h_all, n_chunks):
        expW = np.exp(h_chunk.dot(rbm.w.T) + rbm.vbias)
        tmp = h_chunk.dot(rbm.hbias) + np.sum(np.log(1 + expW), axis=1)
        log_ph = np.concatenate((log_ph, tmp))
    return logsum(log_ph)


def estimate_partition_sum(rbm, n_runs, betas):
    # draw samples from the base model (uniform distr) and initialise logw
    samples = np.random.randint(2, size=(n_runs, rbm.n_visible))
    logw = 0

    # main AIS loop
    for beta in betas[1:-1]:
        # compute unnormalized probabilities p_k+1(v_k)
        expWh = np.exp(beta * (samples.dot(rbm.w) + rbm.hbias))
        logw += beta * samples.dot(rbm.vbias) + \
            np.sum(np.log(1 + expWh), axis=1)

        # apply transition operators
        samples = rbm.gibbs_vhv(samples, beta=beta)[1]

        # compute unnormalized probabilities p_k+1(v_k+1)
        expWh = np.exp(beta * (samples.dot(rbm.w) + rbm.hbias))
        logw -= beta * samples.dot(rbm.vbias) + \
            np.sum(np.log(1 + expWh), axis=1)

    # add target probability p_K(v_K)
    expWh = np.exp(samples.dot(rbm.w) + rbm.hbias)
    logw += samples.dot(rbm.vbias) + np.sum(np.log(1 + expWh), axis=1)

    r_ais = logsum(logw) - np.log(n_runs)
    # print(r_ais, np.log(np.average(np.exp(logw))))
    # numerical stability
    logw_avg = np.mean(logw)
    logstd_rais = np.log(np.std(np.exp(logw - logw_avg))) + logw_avg -\
        np.log(n_runs)/2
    logZ_base = rbm.n_visible * np.log(2)
    logZ = r_ais + logZ_base
    logZ_up = logsum([logstd_rais + np.log(3), r_ais]) + logZ_base
    logZ_down = logdiff([logstd_rais + np.log(3), r_ais]) + logZ_base
    return logZ, logstd_rais + logZ_base, logZ_up, logZ_down


def run_ais(rbm, test_data, train_data=None, n_runs=100, exact=False):
    if exact:
        # compute the true partition sum of the RBM (if possible)
        logZ_true = compute_partition_sum(rbm)
        avg_ll_true = np.mean(-rbm.free_energy(test_data)) - logZ_true
        print('True partition sum: {:.2f}'.format(logZ_true))
        print('True average loglik: {:.2f}'.format(avg_ll_true))

    # Use AIS to estimate the partition sum
    # betas = np.concatenate((np.linspace(0, .5, 500, endpoint=False),
    #                         np.linspace(.5, .9, 10000, endpoint=False),
    #                         np.linspace(.9, 1., 4000)))
    betas = np.linspace(0, 1, 20000)
    logZ_est, logstdZ, est_up, est_down = \
        estimate_partition_sum(rbm, n_runs, betas)

    # compute the estimated average log likelihood of a test set
    avg_ll_est_test = np.mean(-rbm.free_energy(test_data)) - logZ_est
    print('Est. partition sum (+- 3*std): {:.2f}, {:.2f}, {:.2f}'
          ''.format(logZ_est, est_up, est_down))
    print('Est. average loglik (test): {:.2f}'.format(avg_ll_est_test))
    if train_data is not None:
        avg_ll_est_train = np.mean(-rbm.free_energy(train_data)) - logZ_est
        print('Est. average loglik (train): {:.2f}'.format(avg_ll_est_train))

if __name__ == '__main__':
    # # load test set
    # f = gzip.open('datasets/mnist.pkl.gz', 'rb')
    # train_set, _, test_set = cPickle.load(f)
    # f.close()

    data_name = 'pong_var_start36x48'
    with np.load('datasets/' + data_name + '.npz') as d:
        train_set, _, test_set = d[d.keys()[0]]

    # load RBM
    # with open('saved_rbms/mnist_disc_rbm.pkl', 'rb') as f:
    #     rbm = cPickle.load(f)

    with open('saved_rbms/' + data_name + '_crbm.pkl', 'rb') as f:
        rbm = cPickle.load(f)

    test_data = test_set[0]
    train_data = train_set[0]

    if len(train_set[1].shape) == 1:
        train_label = to_1_of_c(train_set[1], rbm.n_labels)
        test_label = to_1_of_c(test_set[1], rbm.n_labels)
    else:
        train_label = train_set[1]
        test_label = test_set[1]

    if isinstance(rbm, CRBM):
        test_data = np.hstack((test_data, test_label))
        train_data = np.hstack((train_data, train_label))

    run_ais(rbm, test_data)

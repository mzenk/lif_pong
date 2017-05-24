from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import cPickle, gzip
from dbm import DBM, CDBM
import itertools


# to avoid entropy errors, i need a ln(0) = 0 function
def safe_ln(x):
    x = x.copy()
    x[x == 0] = 1
    return np.log(x)


# numerical stability
def logsum(x, axis=0):
    alpha = np.max(x, axis=axis) - np.log(sys.float_info.max)/2
    # alpha = np.max(x, axis=axis)
    return alpha + np.log(np.sum(np.exp(x - alpha), axis=axis))


def logdiff(x, axis=0):
    alpha = np.max(x, axis=axis) - np.log(sys.float_info.max)/2
    # alpha = np.max(x, axis=axis)
    return alpha + np.log(np.diff(np.exp(x - alpha), axis=axis)).squeeze()


# only for 2 layers!
def compute_partition_sum(dbm):
    # visible units are too many to sum over
    h1_all = np.array(list(itertools.product([0, 1],
                      repeat=dbm.hidden_layers[0])))
    log_ph = np.zeros(0)
    n_chunks = np.ceil(h1_all.shape[0] * dbm.n_visible / 1e8)
    for h_chunk in np.array_split(h1_all, n_chunks):
        log_ph = np.concatenate((log_ph, unnorm_prob(dbm, h_chunk)))
    return logsum(log_ph)


# add for testing the mf_posterior function
def compute_ll(dbm, data):
    pass


# for now only for 2 layer-dbm
def unnorm_prob(dbm, h1, beta=1):
    if len(h1.shape) == 1:
        h1 = np.expand_dims(h1, 0)
    expW1 = np.exp(beta * h1.dot(dbm.weights[0].T) + dbm.vbias)
    expW2 = np.exp(beta * h1.dot(dbm.weights[1]) + dbm.hbiases[1])

    return beta * h1.dot(dbm.hbiases[0]) + \
        np.sum(np.log(1 + expW1), axis=1) + np.sum(np.log(1 + expW2), axis=1)

# TBD: p* for arbitrary dbms:
# cf. my notes; adapt ais functions
# CDBM should be irrelevant here (?)


# for now only for 2 layer-dbm
def estimate_partition_sum(dbm, n_runs, betas):
    # draw samples from the base model (uniform distr) and initialise logw
    state = [np.random.randint(2, size=(n_runs, dbm.n_visible))]
    for n_units in dbm.hidden_layers:
        state.append(np.random.randint(2, size=(n_runs, n_units)))
    logw = 0

    # main AIS loop
    for beta in betas[1:-1]:
        # compute unnormalized probabilities p_k(v_k)
        logw += unnorm_prob(dbm, state[1], beta)

        # apply transition operators
        state, _ = dbm.gibbs_from_h(state, beta=beta)

        # compute unnormalized probabilities p_k(v_k+1)
        logw -= unnorm_prob(dbm, state[1], beta)
    # add target probability p_K(v_K)
    logw += unnorm_prob(dbm, state[1])

    r_ais = logsum(logw) - np.log(n_runs)
    # numerical stability
    logw_avg = np.mean(logw)
    logstd_rais = np.log(np.std(np.exp(logw - logw_avg))) + logw_avg -\
        np.log(n_runs)/2
    logZ_base = np.sum(dbm.hidden_layers[0]) * np.log(2)
    logZ = r_ais + logZ_base
    logZ_up = logsum([logstd_rais + np.log(3), r_ais]) + logZ_base
    logZ_down = logdiff([logstd_rais + np.log(3), r_ais]) + logZ_base
    return logZ, logstd_rais, logZ_up, logZ_down


def get_mf_posterior(dbm, data, iterations=10, targets=None):
    # compute some reusable quantities
    data_bias = data.dot(dbm.weights[0])
    if targets is not None:
        lab_bias = targets.dot(dbm.weights[-1].T)

    # initialize the posterior distributions\
    totin = data.dot(dbm.weights[0]) + dbm.hbiases[0]
    # mus = [1/(1 + np.exp(-totin))]
    mus = [np.random.rand(*totin.shape)]
    for l in range(1, dbm.n_layers):
        if l == dbm.n_layers - 2 and type(dbm) is CDBM:
            # label layer
            totin = mus[l - 1].dot(dbm.weights[l]) + dbm.hbiases[l] + lab_bias
            # mus.append(1/(1 + np.exp(-totin)))
            mus.append(np.random.rand(*totin.shape))
            break
        else:
            totin = mus[l - 1].dot(dbm.weights[l]) + dbm.hbiases[l]
            # mus.append(1/(1 + np.exp(-totin)))
            mus.append(np.random.rand(*totin.shape))
    print(compute_lower_bound(dbm, mus, data))
    # do mean field updates until convergence
    for n in range(iterations):
        diff_h = 0
        for l in range(len(mus)):
            mu_old = mus[l]
            if type(dbm) is CDBM and l == dbm.n_layers - 2:
                totin = mus[l - 1].dot(dbm.weights[l]) + lab_bias + \
                    dbm.hbiases[l]
                mus[l] = 1/(1 + np.exp(-totin))
                break
            if l == 0:
                totin = mus[l + 1].dot(dbm.weights[l + 1].T) + \
                            data_bias + dbm.hbiases[l]
            elif l == dbm.n_layers - 1:
                totin = mus[l - 1].dot(dbm.weights[l]) + dbm.hbiases[l]
            else:
                totin = mus[l - 1].dot(dbm.weights[l]) + \
                    mus[l + 1].dot(dbm.weights[l + 1].T) + dbm.hbiases[l]
            mus[l] = 1/(1 + np.exp(-totin))
            diff_h += np.mean(np.abs(mus[l] - mu_old))
        print(compute_lower_bound(dbm, mus, data))
        if diff_h < 1e-7 * data.shape[0]:
            break
    return mus


def compute_lower_bound(dbm, posteriors, test_data, logZ=0):
    wh = posteriors[0].dot(dbm.weights[0].T)
    ll_bound_est = np.sum(test_data * wh, axis=1) + test_data.dot(dbm.vbias)
    for l in range(dbm.n_layers - 1):
        mu_l = posteriors[l].copy()
        entropy = np.sum(mu_l * safe_ln(mu_l) +
                         (1 - mu_l) * safe_ln(1 - mu_l), axis=1)
        wh = posteriors[l + 1].dot(dbm.weights[l + 1].T)
        ll_bound_est += entropy + np.sum(posteriors[l] * wh, axis=1) +\
            posteriors[l].dot(dbm.hbiases[l])
    return np.mean(ll_bound_est) - logZ

if __name__ == '__main__':
    # load RBM
    # with open('saved_rbms/mnist_small_dbm.pkl', 'rb') as f:
    #     dbm = cPickle.load(f)
    layers = [4, 3, 2]
    dbm = DBM(layers, vbias_init=np.random.rand(layers[0]))

    # load test set
    # f = gzip.open('datasets/mnist.pkl.gz', 'rb')
    # _, _, test_set = cPickle.load(f)
    # f.close()
    # test_data = test_set[0][:10]
    # test_targets = test_set[1]
    test_data = np.random.randint(2, size=(12, dbm.n_visible))

    # # for debugging: compute true Z
    print('Compute the partition sum brute force...')
    logZ_true = compute_partition_sum(dbm)

    # Use AIS to estimate the partition sum
    n_runs = 100
    betas = np.concatenate((np.linspace(0, .5, 500, endpoint=False),
                            np.linspace(.5, .9, 10000, endpoint=False),
                            np.linspace(.9, 1., 4000)))
    print('Estimate the partition sum using AIS...')
    logZ_est, logstdZ, est_up, est_down = \
        estimate_partition_sum(dbm, n_runs, betas)

    # lower bound for loglik:
    print('Get a lower boundary for the log likelihood of the test data')
    posteriors = get_mf_posterior(dbm, test_data)
    ll_bound_est = compute_lower_bound(dbm, posteriors, test_data, logZ_est)

    print('True partition sum {}'.format(logZ_true))
    print('Est. partition sum (+- 3*std): {:.2f}, {:.2f}, {:.2f}'
          ''.format(logZ_est, est_up, est_down))
    print('Est. lower bound for loglik: {:.2f}'.format(ll_bound_est))

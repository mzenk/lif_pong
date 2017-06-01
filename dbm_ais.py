from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import cPickle, gzip
from dbm import DBM, CDBM
import itertools
from util import to_1_of_c


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
    if dbm.n_layers > 2:
        print('Exact evaluation of partition sum only with 2 layers')
        return 0
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


# # for 2 layer-dbm --- deprecated
# def unnorm_prob(dbm, h1, beta=1):
#     if len(h1.shape) == 1:
#         h1 = np.expand_dims(h1, 0)
#     expW1 = np.exp(beta * h1.dot(dbm.weights[0].T) + dbm.vbias)
#     expW2 = np.exp(beta * h1.dot(dbm.weights[1]) + dbm.hbiases[1])

#     return beta * h1.dot(dbm.hbiases[0]) + \
#         np.sum(np.log(1 + expW1), axis=1) + np.sum(np.log(1 + expW2), axis=1)


# for arbitrary dbms: cf. my notes
def unnorm_prob(dbm, state, beta=1):
    logp = 0
    for n, size in enumerate(dbm.hidden_layers):
        if n == 0:
            act = beta * (state[1].dot(dbm.weights[0].T) + dbm.vbias)
            layer_factor = np.sum(np.log(1 + np.exp(act)), axis=1)
        elif n % 2 == 1:
            layer_factor = beta * state[n].dot(dbm.hbiases[n - 1])
        elif n == dbm.n_layers - 1:
            act = beta * (state[-2].dot(dbm.weights[-1]) + dbm.hbiases[-1])
            layer_factor = np.sum(np.log(1 + np.exp(act)), axis=1)
        else:
            wh_upper = state[n-1].dot(dbm.weights[n-1])
            wh_lower = state[n+1].dot(dbm.weights[n].T)
            layer_factor = np.sum(np.log(1 + np.exp(beta*(wh_upper + wh_lower +
                                                    dbm.hbiases[n]))), axis=1)
        logp += layer_factor
    return logp


def estimate_partition_sum(dbm, n_runs, betas):
    # draw samples from the base model (uniform distr) and initialise logw
    state = [np.random.randint(2, size=(n_runs, dbm.n_visible))]
    for n_units in dbm.hidden_layers:
        state.append(np.random.randint(2, size=(n_runs, n_units)))
    logw = 0

    # main AIS loop
    for beta in betas[1:-1]:
        # compute unnormalized probabilities p_k(v_k)
        logw += unnorm_prob(dbm, state, beta)

        # apply transition operators
        state, _ = dbm.gibbs_from_h(state, beta=beta)

        # compute unnormalized probabilities p_k(v_k+1)
        logw -= unnorm_prob(dbm, state, beta)
    # add target probability p_K(v_K)
    logw += unnorm_prob(dbm, state)

    r_ais = logsum(logw) - np.log(n_runs)
    # numerical stability
    logw_avg = np.mean(logw)
    logstd_rais = np.log(np.std(np.exp(logw - logw_avg))) + logw_avg -\
        np.log(n_runs)/2
    logZ_base = np.sum(dbm.hidden_layers[0]) * np.log(2)
    logZ = r_ais + logZ_base
    logZ_up = logsum([logstd_rais + np.log(3), r_ais]) + logZ_base
    logZ_down = logdiff([logstd_rais + np.log(3), r_ais]) + logZ_base
    return logZ, logstd_rais + logZ_base, logZ_up, logZ_down


# when fully debugged: use the object method instead (identical)
def get_mf_posterior(dbm, data, targets=None, iterations=10):
    # compute some data-dependent quantities in advance
    data_bias = data.dot(dbm.weights[0])
    if targets is not None:
        lab_bias = targets.dot(dbm.weights[-1].T)

    # initialize the posterior distributions\
    totin = data.dot(dbm.weights[0]) + dbm.hbiases[0]
    mus = [1/(1 + np.exp(-totin))]
    for l in range(1, dbm.n_layers):
        if l == dbm.n_layers - 2 and type(dbm) is CDBM:
            # label layer
            totin = mus[l - 1].dot(dbm.weights[l]) + dbm.hbiases[l] + lab_bias
            mus.append(1/(1 + np.exp(-totin)))
            break
        else:
            totin = mus[l - 1].dot(dbm.weights[l]) + dbm.hbiases[l]
            mus.append(1/(1 + np.exp(-totin)))

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

        if diff_h < 1e-7 * data.shape[0]:
            break
    return mus


def compute_lower_bound(dbm, data, targets=None, logZ=0, posteriors=None):
    if posteriors is None:
        posteriors = get_mf_posterior(dbm, data, targets=targets)

    # Add the terms for each layer to the lower bound
    wh = posteriors[0].dot(dbm.weights[0].T)
    ll_bound_est = np.sum(data * wh, axis=1) + data.dot(dbm.vbias)

    for l in range(dbm.n_layers - 1):
        mu_l = posteriors[l].copy()
        entropy = np.sum(mu_l * safe_ln(mu_l) +
                         (1 - mu_l) * safe_ln(1 - mu_l), axis=1)
        if type(dbm) is CDBM and l == dbm.n_layers - 2:
            # label data contribution
            wl = np.sum(posteriors[-1]*targets.dot(dbm.weights[-1].T), axis=1)
            ll_bound_est += wl + targets.dot(dbm.hbiases[-1])
            break
        else:
            wh = posteriors[l + 1].dot(dbm.weights[l + 1].T)
            ll_bound_est += entropy + np.sum(posteriors[l] * wh, axis=1) +\
                posteriors[l].dot(dbm.hbiases[l])

    return np.mean(ll_bound_est) - logZ


def run_ais(dbm, data, targets=None, n_runs=100, exact=False):
    if targets is not None:
        targets = to_1_of_c(targets, dbm.hidden_layers[-1])
    print('Estimate the partition sum using AIS...')
    if exact:
        # for small problems: compute true Z
        logZ_true = compute_partition_sum(dbm)
        print('True partition sum {}'.format(logZ_true))

    # Use AIS to estimate the partition sum
    # betas = np.concatenate((np.linspace(0, .5, 500, endpoint=False),
    #                         np.linspace(.5, .9, 10000, endpoint=False),
    #                         np.linspace(.9, 1., 4000)))
    betas = np.linspace(0, 1, 20000)
    logZ_est, logstdZ, est_up, est_down = \
        estimate_partition_sum(dbm, n_runs, betas)

    # lower bound for loglik:
    ll_bound_est = compute_lower_bound(dbm, data, targets, logZ_est)
    print('Est. partition sum (+- 3*std): {:.2f}, {:.2f}, {:.2f}'
          ''.format(logZ_est, est_up, est_down))
    print('Est. lower bound for loglik: {:.2f}'.format(ll_bound_est))

if __name__ == '__main__':
    # # load DBM
    # with open('saved_rbms/mnist_dbm.pkl', 'rb') as f:
    #     dbm = cPickle.load(f)

    # # load test set
    # f = gzip.open('datasets/mnist.pkl.gz', 'rb')
    # _, _, test_set = cPickle.load(f)
    # f.close()
    # test_data = test_set[0]
    # test_targets = test_set[1]
    # if type(dbm) is CDBM:
    #     pass

    # minimal test
    layers = [4, 3, 2]
    dbm = CDBM(layers, labels=10, vbias_init=np.random.rand(layers[0]))
    data = np.random.randint(2, size=(12, dbm.n_visible))
    targets = np.random.randint(10, size=12)

    run_ais(dbm, data, targets=targets)

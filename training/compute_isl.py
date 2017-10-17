from __future__ import division
from __future__ import print_function
import numpy as np
import multiprocessing as mp
from functools import partial
from utils.data_mgmt import load_rbm
from utils import to_1_of_c, logsum
from rbm import RBM, CRBM


def loglik_helper(x, beta, y):
    assert x.shape[1] == len(y)
    # y[1xd], x[nxd]
    matches = (y == x).sum(axis=1)
    summands = matches * np.log(beta) + (len(y) - matches) * np.log(1 - beta)
    return -np.log(len(x)) + logsum(summands)


# when plotting n_samples vs LL, use ll-values from earlier computations
def loglik_helper_unnorm_add(x, beta, args):
    ll_prev = args[0]
    y = args[1]
    assert x.shape[1] == len(y)
    # y[1xd], x[nxd]
    matches = (y == x).sum(axis=1)
    summands = np.zeros(len(matches) + 1)
    summands[0] = ll_prev
    summands[1:] = matches * np.log(beta) + \
        (len(y) - matches) * np.log(1 - beta)
    return logsum(summands)


class ISL_density_model(object):
    def __init__(self, beta=None, x_data=None):
        self.beta = beta
        self.x = x_data

    def fit(self, s_data, t_data=None, quick=False):
        if t_data is not None:
            self.x = np.concatenate((s_data, t_data), axis=0)
        else:
            self.x = s_data

        if quick:
            # for speed (also done in Wei's paper):
            self.beta = .95
            return

        # grid search for optimal beta parameter (minimzie negative loglik)
        betas = np.linspace(.5, 1, 100, endpoint=False)
        beta_curr = 0
        ll_curr = float('-inf')   # log-likelihood
        for beta in betas:
            self.beta = beta
            tmp = self.avg_loglik_serial(s_data)
            # from the paper it's not totally clear how they train the density
            # model. But if I use self.x here, the best beta will always be 1
            if tmp > ll_curr:
                beta_curr = beta
                ll_curr = tmp

        self.beta = beta_curr

    def avg_loglik_serial(self, y_data):
        if self.beta is None or self.x is None:
            raise NameError('Attributes (beta, x) not initialised.')
        if len(y_data.shape) == 1:
            y_data = np.expand_dims(y_data, 0)
        result = 0
        for y in y_data:
            result += loglik_helper(self.x, self.beta, y)
        result /= y_data.shape[0]
        return result

    def avg_loglik(self, y_data, all_values=False):
        if self.beta is None or self.x is None:
            raise NameError('Attributes (beta, x) not initialised.')
        if len(y_data.shape) == 1:
            y_data = np.expand_dims(y_data, 0)
        helper = partial(loglik_helper, self.x, self.beta)
        pool = mp.Pool(processes=8)
        result = pool.map(helper, y_data)
        pool.close()
        if all_values:
            return np.array(result)
        return np.mean(result)

    def avg_loglik_vs_samples(self, y_data, nx):
        max_samples = len(self.x)
        assert max_samples > 100
        n_samples = np.logspace(2, np.log10(max_samples), nx).astype(int)
        tmp = [-np.inf] * len(y_data)
        isls = []
        for i, n in enumerate(n_samples):
            print('Compute ISL for n = {:.1f}'.format(n))
            if len(y_data.shape) == 1:
                y_data = np.expand_dims(y_data, 0)
            nprev = n_samples[i-1] if i > 0 else 0
            helper = partial(loglik_helper_unnorm_add,
                             self.x[nprev:n], self.beta)
            pool = mp.Pool(processes=4)
            tmp = pool.map(helper, zip(tmp, y_data))
            isls.append(-np.log(n) + np.mean(tmp))
        return n_samples, isls


if __name__ == '__main__':
    # import gzip
    # img_shape = (28, 28)
    # with gzip.open('../shared_data/datasets/mnist.pkl.gz', 'rb') as f:
    #     train_set, _, test_set = np.load(f)
    #     train_data = np.hstack(((train_set[0] > .5)*1,
    #                             to_1_of_c(train_set[1], 10)))[::100]
    #     test_data = np.hstack(((test_set[0] > .5)*1,
    #                            to_1_of_c(test_set[1], 10)))
    # good_rbm = load_rbm('mnist_disc_rbm')
    # bad_rbm = CRBM(good_rbm.n_inputs, good_rbm.n_hidden, good_rbm.n_labels)

    # good_samples = good_rbm.draw_samples(1e4, binary=True)[:, :good_rbm.n_visible]
    # bad_samples = bad_rbm.draw_samples(1e4, binary=True)[:, :bad_rbm.n_visible]

    # isl_model = ISL_density_model()
    # isl_model.fit(train_data, quick=True)
    # print('LL with ISL (only train): {}'.format(
    #     isl_model.avg_loglik(test_data)))
    # isl_model.fit(bad_samples, train_data, quick=True)
    # print('LL with ISL (bad + train): {}'.format(
    #     isl_model.avg_loglik(test_data)))
    # isl_model.fit(good_samples, train_data, quick=True)
    # print('LL with ISL (good + train): {}'.format(
    #     isl_model.avg_loglik(test_data)))
    # isl_model.fit(bad_samples, quick=True)
    # print('LL with ISL (bad + train): {}'.format(
    #     isl_model.avg_loglik(test_data)))
    # isl_model.fit(good_samples, quick=True)
    # print('LL with ISL (good + train): {}'.format(
    #     isl_model.avg_loglik(test_data)))

    # ==== testing with minimal RBM ====
    # nv = 4
    # nh = 3
    # w_small = 2*(np.random.beta(1.5, 1.5, (nv, nh)) - .5)

    # for i in range(1):
    #     target_rbm = RBM(nv, nh, w_small, np.zeros(nv), np.zeros(nh))
    #     samples = target_rbm.draw_samples(int(1e4), binary=True)[:, :nv]
    #     train_samples = samples[1000:7000]
    #     test_samples = samples[8000:]
    #     my_rbm = RBM(nv, nh)
    #     samples = target_rbm.draw_samples(int(5e3), binary=True)[:, :nv]
    #     gen_samples = samples[1000:]

    #     isl_model = ISL_density_model()
    #     isl_model.fit(gen_samples)
    #     print(isl_model.beta)
    #     print('LL with ISL: {}'.format(isl_model.avg_loglik(test_samples)))

    #     isl_model_quick = ISL_density_model()
    #     isl_model_quick.fit(gen_samples, quick=True)
    #     print('LL with quick ISL: {}'.format(
    #         isl_model_quick.avg_loglik_serial(test_samples)))

    #     z_true = my_rbm.compute_partition_sum()
    #     my_rbm.run_ais(test_samples)
    #     ll_true = -my_rbm.free_energy(test_samples).mean() - z_true
    #     print('True Z: ' + str(z_true))
    #     print('True LL: {}'.format(ll_true))

    # import timeit
    # setup = 'from __main__ import isl_model, test_samples'
    # print(timeit.Timer('isl_model.avg_loglik(test_samples)',
    #                    setup=setup).timeit(number=100))
    # print(timeit.Timer('isl_model.avg_loglik_serial(test_samples)',
    #                    setup=setup).timeit(number=100))

    nv = 4
    nh = 3
    w_small = 2*(np.random.beta(1.5, 1.5, (nv, nh)) - .5)

    target_rbm = RBM(nv, nh, w_small, np.zeros(nv), np.zeros(nh))
    samples = target_rbm.draw_samples(int(1e4), binary=True)[:, :nv]
    train_samples = samples[1000:7000]
    test_samples = samples[8000:]
    my_rbm = RBM(nv, nh)
    samples = target_rbm.draw_samples(int(5e3), binary=True)[:, :nv]
    gen_samples = samples[1000:]

    model = ISL_density_model()
    model.fit(train_samples, quick=True)
    x, y = model.avg_loglik_vs_samples(test_samples, 10)
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    plt.plot(x, y, 'o')
    plt.savefig('test.png')

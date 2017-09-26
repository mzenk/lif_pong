from __future__ import division
from __future__ import print_function
import numpy as np
import multiprocessing as mp
from functools import partial
import timeit
from utils.data_mgmt import load_rbm
from rbm import RBM, CRBM


def loglik_helper(x, beta, y):
    bernoulli_factors = (y == x)*(2*beta - 1) + 1 - beta
    return np.log(np.mean(np.prod(bernoulli_factors, axis=1)))


class ISL_density_model(object):
    def __init__(self, beta=None, x_data=None):
        self.beta = beta
        self.x = x_data

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

    def avg_loglik(self, y_data):
        if self.beta is None or self.x is None:
            raise NameError('Attributes (beta, x) not initialised.')
        if len(y_data.shape) == 1:
            y_data = np.expand_dims(y_data, 0)
        helper = partial(loglik_helper, self.x, self.beta)
        pool = mp.Pool(processes=8)
        result = pool.map(helper, y_data)
        result = np.mean(result)
        return result

    def fit(self, s_data, t_data=None):
        if t_data is not None:
            self.x = np.concatenate((s_data, t_data), axis=0)
        else:
            self.x = s_data

        # grid search for optimal beta parameter (minimzie negative loglik)
        betas = np.linspace(.5, 1, 100)
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


if __name__ == '__main__':
    nv = 10
    nh = 5
    w_small = 2*(np.random.beta(1.5, 1.5, (nv, nh)) - .5)

    target_rbm = RBM(nv, nh, w_small, np.zeros(nv), np.zeros(nh))
    train_samples = target_rbm.draw_samples(int(1e3), binary=True)
    test_samples = target_rbm.draw_samples(int(1e3), binary=True)
    my_rbm = RBM(nv, nh)
    gen_samples = my_rbm.draw_samples(int(1e3), binary=True)

    isl_model = ISL_density_model()
    isl_model.fit(gen_samples, train_samples)

    setup = 'from __main__ import isl_model, test_samples'
    print(timeit.Timer('isl_model.avg_loglik(test_samples)',
                       setup=setup).timeit(number=100))
    print(timeit.Timer('isl_model.avg_loglik_serial(test_samples)',
                       setup=setup).timeit(number=100))
    # print(isl_model.avg_loglik_serial(test_samples))
    # print(isl_model.beta)

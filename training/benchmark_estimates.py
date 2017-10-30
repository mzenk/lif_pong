# benchmark LL-estimation methods
from __future__ import division
from __future__ import print_function
import numpy as np
import timeit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rbm import RBM

time_isl, time_ais, time_pl = [], [], []
# repeat for different RBM-sizes
nv_range = np.arange(200, 2200, 200)
nh = 200
for nv in nv_range:
    rbm = RBM(nv, nh)
    test_data = np.random.randint(2, size=(2000, int(nv)))

    # time measurement
    setup = 'from __main__ import rbm, test_data'

    # ISL
    time_isl.append(timeit.Timer('rbm.estimate_loglik_isl(1e4, test_data)',
                                 setup=setup).timeit(number=3))

    # AIS
    time_ais.append(timeit.Timer('rbm.run_ais(test_data)',
                                 setup=setup).timeit(number=1))

    # logPL
    time_pl.append(timeit.Timer('rbm.compute_logpl(test_data)',
                                setup=setup).timeit(number=10))

print(time_isl, time_ais, time_pl)
n_params = nh * nv_range
plt.plot(n_params, time_isl, '.', label='ISL')
plt.plot(n_params, time_ais, '.', label='AIS')
plt.plot(n_params, time_pl, '.', label='LPL')
plt.xlabel('Number of parameters in model')
plt.ylabel('Time for LL-estimate [s]')
plt.savefig('benchmark.png')

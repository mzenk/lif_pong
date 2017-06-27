# short script for testing the DBM sampling
from __future__ import division
from __future__ import print_function
import numpy as np
from dbm import DBM
from util import boltzmann, bin_to_dec, dec_to_bin, compute_dkl
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Basic DBM testing
# ------ Test the sampling -------
layers = [4, 3, 2]
my_dbm = DBM(layers, vbias_init=np.random.rand(layers[0]))

[w1, w2] = my_dbm.weights

zero_v = np.zeros((layers[0], layers[0]))
zero_1 = np.zeros((layers[1], layers[1]))
zero_2 = np.zeros((layers[2], layers[2]))
zero_v2 = np.zeros((layers[0], layers[2]))

w_tot = np.asarray(np.bmat([[zero_v, w1, zero_v2],
                            [w1.T, zero_1, w2],
                            [zero_v2.T, w2.T, zero_2]]))

b_tot = np.concatenate([my_dbm.vbias] + my_dbm.hbiases)

# evaluate target distribution
dim = sum(layers)
target = np.zeros(2**dim)
nbits = np.floor(np.log2(2**dim - 1)) + 1
for i in range(2**dim):
    target[i] = boltzmann(dec_to_bin(i, nbits), w_tot, b_tot)
p_target = target/np.sum(target)

# compare dkls of gibbs and AST
n_samples = 1e6
samples = my_dbm.draw_samples(n_samples, binary=True, layer_ind='all')[1000:]
decimals = bin_to_dec(samples)
ns = np.logspace(1, np.log10(samples.shape[0]), 100)
dkl = np.zeros_like(ns)
for i, n in enumerate(ns):
    dkl[i] = compute_dkl(samples[:int(n), :], p_target)
plt.figure()
plt.loglog(ns, dkl)
plt.savefig('dkl.png')

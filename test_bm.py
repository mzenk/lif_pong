from __future__ import division
from __future__ import print_function
import numpy as np
import cPickle
from rbm import RBM, CRBM
from util import boltzmann, bin_to_dec, dec_to_bin, compute_dkl
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Basic RBM testing
# ------ Test the sampling -------
nv = 3
nh = 2
dim = nv + nh
w_small = np.random.rand(nv, nh)
myrbm = RBM(nv, nh, w=w_small, vbias=np.zeros(nv), hbias=np.zeros(nh))
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

# # compare dkls of gibbs and AST
# n_samples = 1e5
# samples = myrbm.draw_samples_ast(n_samples, binary=True)[1000:, :]
# decimals = bin_to_dec(samples)
# ns = np.logspace(1, np.log10(samples.shape[0]), 100)
# dkl = np.zeros_like(ns)
# for i, n in enumerate(ns):
#     dkl[i] = compute_dkl(samples[:int(n), :], p_target)
# plt.figure()
# plt.loglog(ns, dkl, label='ast')

# samples = myrbm.draw_samples(n_samples, binary=True)[1000:, :]
# decimals = bin_to_dec(samples)
# ns = np.logspace(1, np.log10(samples.shape[0]), 100)
# dkl = np.zeros_like(ns)
# for i, n in enumerate(ns):
#     dkl[i] = compute_dkl(samples[:int(n), :], p_target)
# plt.loglog(ns, dkl, 'g', label='gibbs')
# plt.legend()
# plt.savefig('dkl.png')

# # compare histograms
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

# ----- test training by learning a known (simple) distribution -----
# nv = 3
# nh = 2
# w_small = 2*(np.random.beta(1.5, 1.5, (nv,nh)) - .5)
# myrbm = RBM(nv, nh)

# # generate samples from true distribution and train bm
# train = RBM(nv, nh, w_small, np.zeros(nv), np.zeros(nh))
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

with open('saved_rbms/minimal_rbm.pkl', 'wb') as output:
        cPickle.dump(myrbm, output, cPickle.HIGHEST_PROTOCOL)

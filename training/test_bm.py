from __future__ import division
from __future__ import print_function
import numpy as np
import cPickle
from rbm import RBM, CRBM
from lif_pong.utils import boltzmann, bin_to_dec, dec_to_bin, compute_dkl
from lif_pong.utils.data_mgmt import make_data_folder
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# # Basic RBM testing
# # ------ Test the sampling -------
# nv = 5
# nh = 4
# dim = nv + nh
# w_small = .1*np.random.randn(nv, nh)
# myrbm = RBM(nv, nh)
# w, b = myrbm.bm_params()

# # evaluate target distribution
# target = np.zeros(2**dim)
# nbits = np.floor(np.log2(2**dim - 1)) + 1
# for i in range(2**dim):
#     target[i] = boltzmann(dec_to_bin(i, nbits), w, b)
# p_target = target/np.sum(target)

# # compare dkls of gibbs and AST
# n_samples = 1e5
# # samples = myrbm.draw_samples_ast(n_samples, binary=True)[1000:, :]
# # decimals = bin_to_dec(samples)
# # ns = np.logspace(1, np.log10(samples.shape[0]), 100)
# # dkl = np.zeros_like(ns)
# # for i, n in enumerate(ns):
# #     dkl[i] = compute_dkl(samples[:int(n), :], p_target)
# # plt.figure()
# # plt.loglog(ns, dkl, label='ast')

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

# ------ Test clamped sampling -------
nv = 5
nh = 4
nl = 2
dim = nv + nh
# myrbm = RBM(nv, nh, w=w_small)
myrbm = CRBM(nv - nl, nh, nl)
w, b = myrbm.bm_params()
v_init = np.random.randint(2, size=nv)
print(v_init)
clamped_idx = np.arange(1, nv)
clamped_val = np.random.randint(2, size=len(clamped_idx))
# evaluate target distribution
free_energy_diff = myrbm.free_energy(np.hstack(([0], clamped_val))) - \
    myrbm.free_energy(np.hstack(([1], clamped_val)))
p_on = 1/(1 + np.exp(-free_energy_diff))
print('correct: {}'.format(p_on))
p_target = np.array([1 - p_on, p_on])

# compare clamped sampling methods
n_samples = 1e6
samples = myrbm.sample_with_clamped_units(
    n_samples, clamped_ind=clamped_idx, clamped_val=clamped_val, v_init=v_init,
    binary=True)[100:]
samples = np.expand_dims(samples, 1)
print(np.mean(samples == 1))
decimals = bin_to_dec(samples)
ns = np.logspace(1, np.log10(samples.shape[0]), 100)
dkl = np.zeros_like(ns)
for i, n in enumerate(ns):
    dkl[i] = compute_dkl(samples[:int(n), :], p_target)
plt.figure()
plt.loglog(ns, dkl, label='split rbms')

samples2 = myrbm.draw_samples(n_samples, clamped=clamped_idx, v_init=v_init,
                              clamped_val=clamped_val, binary=True)[100:, 0]
print(np.mean(samples2 == 1))
samples2 = np.expand_dims(samples2, 1)
decimals = bin_to_dec(samples2)
ns = np.logspace(1, np.log10(samples2.shape[0]), 100)
dkl = np.zeros_like(ns)
for i, n in enumerate(ns):
    dkl[i] = compute_dkl(samples2[:int(n), :], p_target)
plt.loglog(ns, dkl, 'g', label='normal')
plt.legend()
plt.savefig('dkl.png')
np.savez('samples', normal=samples2, split=samples)

# # compare histograms
# plt.figure()
# h = plt.hist(decimals, bins=np.arange(0, 2**dim + 1, 1), normed=True,
#              alpha=0.5, label='sampled', color='g', align='mid',
#              rwidth=.5, log=True)
# plt.bar(np.arange(0, 2**dim, 1), p_target, width=.5,
#         alpha=0.5, label='target', color='b')
# plt.legend(loc='upper left')
# plt.savefig("histo.png")

# # ----- test training by learning a known (simple) distribution -----
# nv = 3
# nh = 2
# w_small = 2*(np.random.beta(1.5, 1.5, (nv, nh)) - .5)
# myrbm = RBM(nv, nh)

# # generate samples from true distribution and train bm
# train = RBM(nv, nh, w_small, np.zeros(nv), np.zeros(nh))
# train_samples = train.draw_samples(int(1e4), binary=True)
# v_train_samples = train_samples[1000:, :nv]

# valid = train.draw_samples(2000, binary=True)[1500:, :nv]
# myrbm.train(v_train_samples, valid_set=valid, cast=False)

# # run bm and compare histograms
# samples = myrbm.draw_samples(int(1e5), binary=True)

# decimals = bin_to_dec(samples[:, :nv])
# decimals_train = bin_to_dec(v_train_samples)
# plt.figure()
# h1 = plt.hist(decimals, bins=np.arange(0, 2**nv + 1, 1), normed=True,
#               alpha=0.5, label='Trained', color='g', align='mid',
#               rwidth=0.5, log=True)
# h2 = plt.hist(decimals_train, bins=np.arange(0, 2**nv + 1, 1), normed=True,
#               alpha=0.5, label='groundtruth', color='b', align='mid',
#               rwidth=0.5, log=True)
# plt.legend(loc='upper left')
# plt.savefig('test.png')
# plt.close()

# p_target = np.histogram(decimals_train, bins=np.arange(0, 2**nv + 1, 1),
#                         normed=True)[0]

# print("DKL = " + str(compute_dkl(samples[:, :nv], p_target)))

# with open('../shared_data/saved_rbms/', 'wb') as output:
# myrbm.save(make_data_folder('saved_rbms', True) + 'minimal_rbm.pkl')

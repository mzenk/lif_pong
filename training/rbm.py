# Restricted Boltzmann machine
from __future__ import division
from __future__ import print_function
import numpy as np
from lif_pong.utils import logsum, logdiff
import compute_isl as isl
import itertools
import time
import logging
import cPickle


def load(rbm_dict):
    rbm_type = rbm_dict.pop('type')
    if rbm_type == 'rbm':
        return RBM(**rbm_dict)
    if rbm_type == 'crbm':
        return CRBM(**rbm_dict)


class RBM(object):
    def __init__(self, n_visible, n_hidden,
                 w=None, vbias=None, hbias=None,
                 numpy_seed=None, dbm_factor=[1, 1]):
        # Set seed for reproducatbility
        self.np_rng = np.random.RandomState(numpy_seed)

        # shape of w: (n_visible, n_hidden)
        self.n_hidden = n_hidden
        self.n_visible = n_visible

        if w is None:
            w = .01*self.np_rng.randn(n_visible, n_hidden)

        if vbias is None:
            # For best initialization see Hinton's guide; needs to be passed
            vbias = np.zeros(n_visible)
        if hbias is None:
            hbias = np.zeros(n_hidden)

        self.w = w
        self.vbias = vbias
        self.hbias = hbias

        # This factor is needed during the layerwise DBM training.
        # Making it a instance variable is not good OOP-style but
        # a simple solution
        self.dbm_factor = dbm_factor

    def save(self, filename):
        rbm_dict = {
            'type': 'rbm',
            'n_visible': self.n_visible,
            'n_hidden': self.n_hidden,
            'w': self.w,
            'vbias': self.vbias,
            'hbias': self.hbias
        }
        with open(filename, 'wb') as output:
            cPickle.dump(rbm_dict, output, cPickle.HIGHEST_PROTOCOL)

    def set_seed(self, seed):
        self.np_rng = np.random.RandomState(seed)

    # get parameters in BM notation
    def bm_params(self):
        w = np.vstack(
            (np.hstack((np.zeros(2*(self.n_visible,)), self.w)),
             np.hstack((self.w.T, np.zeros(2*(self.n_hidden,)))))
            )
        b = np.concatenate((self.vbias, self.hbias))
        return w, b

    def energy(self, v, h):
        return - np.einsum('ik,il,kl->i', v, h, self.w) - \
            v.dot(self.vbias) - h.dot(self.hbias)

    def free_energy(self, v_data):
        if len(v_data.shape) == 1:
            v_data = np.expand_dims(v_data, 0)
        activation = v_data.dot(self.w) + self.hbias
        hidden_term = np.sum(np.log(1 + np.exp(activation)), axis=1)
        return - v_data.dot(self.vbias) - hidden_term

    def compute_partition_sum(self):
        # visible units are too many to sum over
        h_all = np.array(list(itertools.product([0, 1], repeat=self.n_hidden)))
        log_ph = np.zeros(0)
        n_chunks = np.ceil(h_all.shape[0] * self.n_visible / 1e8)
        for h_chunk in np.array_split(h_all, n_chunks):
            expW = np.exp(h_chunk.dot(self.w.T) + self.vbias)
            tmp = h_chunk.dot(self.hbias) + np.sum(np.log(1 + expW), axis=1)
            log_ph = np.concatenate((log_ph, tmp))
        return logsum(log_ph)

    # ======== Sampling methods ========

    # generalized method for sampling quickly from many chains (numpy).
    # AST does not work with multiple chains
    def draw_samples(self, n_samples, v_init=None, n_chains=1, binary=False,
                     clamped=None, clamped_val=None, ast=False):
        clamped_idx = clamped  # quick fix (don't want to touch scripts)
        if ast:
            if clamped_idx is not None:
                print('No clamped_idx sampling with AST')
                return 0
            return self.draw_samples_ast(n_samples, v_init, binary)

        # initialize the chains
        # for clamping each chain can take different clamped_val
        if clamped_idx is not None:
            assert clamped_val is not None
            if len(clamped_val.shape) == 1:
                clamped_val = np.expand_dims(clamped_val, 0)
            n_chains = clamped_val.shape[0]

        if v_init is None:
            v_init = self.np_rng.randint(2, size=(n_chains, self.n_visible))
        elif len(v_init.shape) == 1:
            v_init = np.expand_dims(v_init, 0)

        v_curr = v_init.astype(float)
        if clamped_idx is not None:
            v_curr[:, clamped_idx] = clamped_val
            assert n_chains == v_init.shape[0]

        n_samples = int(n_samples)
        samples = np.empty((n_samples, n_chains, self.n_visible +
                            self.n_hidden))
        samples = samples.squeeze()
        # update units and renew clamping if necessary
        for t in range(n_samples):
            pv, v_curr, ph, h_curr = self.gibbs_vhv(v_curr)
            if clamped_idx is not None:
                if n_chains == 1:
                    v_curr = np.expand_dims(v_curr, 0)
                    pv = np.expand_dims(pv, 0)
                v_curr[:, clamped_idx] = clamped_val
                pv[:, clamped_idx] = clamped_val

            if binary:
                samples[t] = np.hstack((v_curr.squeeze(), h_curr))
            else:
                samples[t] = np.hstack((pv.squeeze(), ph))
        return samples.squeeze()

    # doesn't work for multiple chains yet
    def draw_samples_ast(self, n_samples, v_init=None, binary=False):
        n_samples = int(n_samples)
        # setup ast
        betas = np.linspace(1., .5, 10)
        g_weights = np.ones(betas.size)
        gamma = 10

        samples = np.empty((n_samples, self.n_visible + self.n_hidden))
        if v_init is None:
            v_init = self.np_rng.randint(2, size=self.n_visible)
        v_curr = v_init
        k_curr = np.array(0)
        h_curr = np.zeros(self.n_hidden)

        # draw samples; since the user can specify how many samples he wants
        # the chain may have to run quite long
        for t in range(n_samples):
            gamma = 10/(1 + t/1e2)
            while True:
                pv, v_curr, ph, h_curr, k_curr, g_weights = \
                    self.ast_step(v_curr, k_curr, betas, g_weights, gamma,
                                  starting_from_h=False)
                if k_curr == 0:
                    break

            if binary:
                samples[t] = np.concatenate((v_curr, h_curr))
            else:
                samples[t] = np.concatenate((pv, ph))
        return samples

    # for simplicity one could replace this method by modifying the normal
    # sampling method as in DBM
    def sample_with_clamped_units(self, n_samples, clamped_ind, clamped_val,
                                  binary=False, v_init=None):
        """
            clamped_ind: indices of clamped values (same for each input)
            clamped_val: (n_instances, n_clamped)
            returns: samples in shape (n_samples, n_instances, n_unclamped)
            - For efficiency, do not use the whole weight matrix in the hv-step
            - So far, this method works only with Gibbs sampling
            - If you want to sample from the same input multiple times, it
              might be worth passing a tiled repetition to speed things up
            - Method can be used as well for:
              classification (clamped input, sampled labels)
              or generative mode (clamped label, sample input)
        """
        n_samples = int(n_samples)
        unclamped_ind = np.setdiff1d(np.arange(self.n_visible), clamped_ind)

        # # this is a less efficient variant but uses draw_samples
        # samp = self.draw_samples(n_samples, binary=binary, v_init=v_init,
        #                          clamped=clamped_ind, clamped_val=clamped_val)
        # if len(samp.shape) == 3:
        #     return samp[:, :, unclamped_ind]
        # else:
        #     return samp[:, unclamped_ind]

        # Create an extra RBM for the h->v step for efficiency
        unclamped_rbm = RBM(n_visible=unclamped_ind.size,
                            n_hidden=self.n_hidden,
                            w=self.w[unclamped_ind, :],
                            vbias=self.vbias[unclamped_ind],
                            hbias=self.hbias)

        if len(clamped_val.shape) == 1:
            clamped_val = np.expand_dims(clamped_val, 0)

        samples = np.zeros((n_samples, clamped_val.shape[0],
                            unclamped_ind.size))
        hid_samples = np.zeros((n_samples, clamped_val.shape[0],
                               self.n_hidden))

        if v_init is None:
            v_init = self.np_rng.randint(2, size=(clamped_val.shape[0],
                                                  self.n_visible))
        elif len(v_init.shape) == 1:
            v_init = np.expand_dims(v_init, 0)
        v_curr = v_init.astype(float)
        v_curr[:, clamped_ind] = clamped_val

        # sample starting from  multiple initial values
        for t in range(n_samples):
            ph, h_samples = self.sample_h_given_v(v_curr)
            unclamped_mean, unclamped_samples = \
                unclamped_rbm.sample_v_given_h(h_samples)
            v_curr[:, unclamped_ind] = unclamped_samples
            if binary:
                samples[t] = unclamped_samples
                hid_samples[t] = h_samples
            else:
                samples[t] = unclamped_mean
                hid_samples[t] = ph
        return samples.squeeze(), hid_samples.squeeze()

    def gibbs_hvh(self, h_start, beta=1.):
        pv, v = self.sample_v_given_h(h_start, beta)
        ph, h = self.sample_h_given_v(v, beta)
        return [pv, v, ph, h]

    def gibbs_vhv(self, v_start, beta=1.):
        ph, h = self.sample_h_given_v(v_start, beta)
        pv, v = self.sample_v_given_h(h, beta)
        return [pv, v, ph, h]

    def ast_step(self,
                 x_start,
                 k_start,
                 betas,
                 g_weights,
                 gamma,
                 starting_from_h=True):
        # update state (v,h) with Gibbs sampler at given temperature
        if starting_from_h:
            pv_new, v_new, ph_new, h_new = self.gibbs_hvh(x_start,
                                                          betas[k_start])
        else:
            pv_new, v_new, ph_new, h_new = self.gibbs_vhv(x_start,
                                                          betas[k_start])

        # if input is just one sample
        if len(v_new.shape) == 1:
            v_new = np.expand_dims(v_new, 0)
            h_new = np.expand_dims(h_new, 0)
        if len(k_start.shape) == 0:
            k_start = np.expand_dims(k_start, 0)
            g_weights = np.expand_dims(g_weights, 0)

        # mc step for temperature change
        k_start_at_bottom = (k_start == 0)
        k_start_at_top = (k_start == betas.size - 1)
        k_proposal = k_start + \
            ((self.np_rng.rand(k_start.size) < .5)*1. - .5)*2
        k_proposal[k_start_at_bottom] = 1
        k_proposal[k_start_at_top] = betas.size - 2
        k_proposal = k_proposal.astype(np.int32)

        q_ratio = np.ones(k_start.size)
        q_ratio[np.logical_or(k_start_at_bottom, k_start_at_top)] = .5
        q_ratio[np.logical_or(k_proposal == 0,
                              k_proposal == betas.size - 1)] = 2.

        p_accept = np.minimum(np.ones_like(k_start), q_ratio *
                              g_weights[np.arange(k_start.size), k_start] /
                              g_weights[np.arange(k_start.size), k_proposal] *
                              np.exp(-(betas[k_proposal] - betas[k_start]) *
                              self.energy(v_new, h_new)))

        k_new = k_start.copy()
        accepted = (p_accept > self.np_rng.rand(p_accept.size))
        k_new[accepted] = k_proposal[accepted]

        # weight update
        g_weights[np.arange(k_new.size), k_new] *= 1 + gamma

        # renormalize so that weights do not explode
        g_weights /= np.expand_dims(np.max(g_weights, axis=1), 1)

        return [pv_new.squeeze(), v_new.squeeze(), ph_new.squeeze(),
                h_new.squeeze(), k_new.squeeze(), g_weights.squeeze()]

    # conditional sampling functions are vectorized
    # -> take n conditional states and yield sample for each
    # beta factor is used in AST
    def sample_h_given_v(self, v_in, beta=1.):
        if len(v_in.shape) == 1:
            v_in = np.expand_dims(v_in, 0)
        if hasattr(beta, "__len__"):
            # for ast we get an array with 'n_instances' entries
            beta = np.expand_dims(beta, 1)

        u = beta * self.dbm_factor[0] * (v_in.dot(self.w) + self.hbias)
        p_on = 1./(1 + np.exp(-u))
        h_samples = (self.np_rng.rand(*p_on.shape) < p_on)*1.
        return [p_on.squeeze(), h_samples.squeeze()]

    def sample_v_given_h(self, h_in, beta=1.):
        if len(h_in.shape) == 1:
            h_in = np.expand_dims(h_in, 0)
        if hasattr(beta, "__len__"):
            # for ast we get an array with 'n_instances' entries
            beta = np.expand_dims(beta, 1)

        u = beta * self.dbm_factor[1] * (h_in.dot(self.w.T) + self.vbias)
        p_on = 1./(1 + np.exp(-u))
        v_samples = (self.np_rng.rand(*p_on.shape) < p_on)*1.
        return [p_on.squeeze(), v_samples.squeeze()]

    # ======== Training --- (P)CD and CAST ========

    def compute_grad_cdn(self, batch, n_steps, persistent=None,
                         cast_variables=None):
        """
            This method computes the gradient of the complete batch.
            persistent: state of the hidden variables in previous state
             of persistent chain
            cast_variables: dictionary with additional cast stuff;
             is updated inside this method
        """

        # data must be normalized to [0,1]
        v0 = batch

        # sampling step for train_data average
        ph0, h0 = self.sample_h_given_v(v0)

        if persistent is not None:
            h0 = persistent

        h = h0
        v, pv, ph = np.zeros_like(batch), np.zeros_like(batch),\
            np.zeros_like(batch)
        # obtain samples for model average
        for i in range(n_steps):
            # ++++++++++++
            # new for cast
            if cast_variables is not None:
                h_slow = h0[:(h0.shape[0] // 2)]
                h_fast = h0[(h0.shape[0] // 2):]
                pv_slow, v_slow, ph_slow, h_slow = self.gibbs_hvh(h_slow)
                pv_fast, v_fast, ph_fast, h_fast,\
                    cast_variables['k_start'], cast_variables['g_weights'] = \
                    self.ast_step(h_fast, **cast_variables)
            # ++++++++++++
            else:
                pv, v, ph, h = self.gibbs_hvh(h)

        vis_recon = v
        hid_recon = h

        # ++++++++++++
        # new for cast
        # for readability
        if cast_variables is not None:
            pv, v, ph, h = pv_slow, v_slow, ph_slow, h_slow
            hid_recon = np.concatenate((h_slow, h_fast), axis=0)
        # ++++++++++++

        # to work with einsum function. If persistent ph0 and pv,ph can have
        # different shapes
        if len(ph0.shape) == 1:
            ph0 = np.expand_dims(ph0, 0)
        if len(pv.shape) == 1:
            ph = np.expand_dims(ph, 0)
            pv = np.expand_dims(pv, 0)

        # compute AVERAGED gradients
        grad_w = np.einsum('ij,ik', batch, ph0)/ph0.shape[0] - \
            np.einsum('ij,ik', pv, ph)/ph.shape[0]
        grad_bh = np.average(ph0, axis=0) - np.average(ph, axis=0)
        grad_bv = np.average(batch, axis=0) - np.average(pv, axis=0)

        return [grad_w, grad_bv, grad_bh], [vis_recon, hid_recon]

    def train(self, train_data, n_epochs=5, batch_size=10, lrate=.01,
              cd_steps=1, persistent=False, cast=False, valid_set=None,
              momentum=0, weight_cost=1e-5, log_name=None, log_isl=False):
        # initializations
        n_instances = train_data.shape[0]
        n_batches = int(np.ceil(n_instances/batch_size))
        shuffled_data = train_data[self.np_rng.permutation(n_instances), :]
        pchain_init = None
        w_incr = np.zeros_like(self.w)
        vb_incr = np.zeros_like(self.vbias)
        hb_incr = np.zeros_like(self.hbias)
        initial_lrate = lrate

        # logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        # create console handler and set level to debug
        if log_name is None:
            log_name = time.strftime('%y%m%d%-H%M') + 'train.log'
        ch = logging.FileHandler(log_name)
        ch.setLevel(logging.INFO)
        # create formatter
        formatter = logging.Formatter('%(asctime)s : %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.info('\n---------------------------------------------\n'
                    '#hidden: {}\nBatch size: {}\nLearning_rate: {}\n'
                    'CD-steps: {}\nPersistent: {}\nCAST: {}'
                    '\n---------------------------------------------'
                    ''.format(self.n_hidden, batch_size, lrate, cd_steps,
                              persistent, cast))
        logger.debug('Step | Free energy difference | log Pseudolikelihood')

        # ++++++++++++
        # new for cast
        cast_variables = None
        swap_state = None
        if cast:
            persistent = True
            # same number of fast and slow chains is implicit everywhere
            if batch_size % 2 == 1:
                print('CAST implementation does not support odd batch sizes')
                return
            n_fast_chains = batch_size // 2
            n_betas = 10
            cast_variables = {
                'k_start': np.zeros(n_fast_chains, dtype=int),
                'g_weights': np.ones((n_fast_chains, n_betas)),
                'gamma': 10.,
                'betas': np.linspace(1., .8, n_betas)
            }
            swap_state = np.zeros((n_fast_chains, self.n_hidden))
        # +++++++++++++

        monitoring_time = 0
        self.monitor_progress(train_data, valid_set, logger)

        for epoch_index in range(n_epochs):
            logger.info('Epoch {}'.format(epoch_index + 1))
            # if momentum != 0 and epoch_index > 5:
            #     # momentum = min(momentum + .1, .9)
            #     momentum = .9

            for batch_index in range(n_batches):
                update_step = batch_index + n_batches * epoch_index

                # Other lrate schedules are possible
                lrate = initial_lrate * 2000 / (2000 + update_step)

                # pick mini-batch randomly; actually
                # http://leon.bottou.org/publications/pdf/tricks-2012.pdf
                # says that shuffling and sequential picking is also ok
                batch = shuffled_data[batch_index*batch_size:
                                      min((batch_index + 1)*batch_size,
                                          n_instances), :]

                # compute "gradient" on batch
                grad, reconstruction = \
                    self.compute_grad_cdn(batch, cd_steps,
                                          persistent=pchain_init,
                                          cast_variables=cast_variables)

                # update parameters including momentum
                w_incr = momentum * w_incr + \
                    lrate * (grad[0] - weight_cost * self.w)
                vb_incr = momentum * vb_incr + lrate * grad[1]
                hb_incr = momentum * hb_incr + lrate * grad[2]
                self.w += w_incr
                self.vbias += vb_incr
                self.hbias += hb_incr

                # ++++++++++++
                # new for cast
                if cast:
                    mask = np.array(cast_variables['k_start'] == 0)
                    fast_recon = reconstruction[1][-n_fast_chains:]
                    slow_recon = reconstruction[1][:n_fast_chains]
                    # remember the last states for which k==0 for the next swap
                    if np.any(mask):
                        if mask.size == 1:
                            swap_state = fast_recon.copy()
                        else:
                            swap_state[mask] = fast_recon[mask].copy()

                    if update_step != 0 and update_step % 50 == 0:
                        # swap slow and fast chain states
                        reconstruction[1][-n_fast_chains:] = slow_recon
                        reconstruction[1][:n_fast_chains] = swap_state
                # +++++++++++++

                if persistent:
                    pchain_init = reconstruction[1]

                # More detailed monitoring for debugging
                if update_step % 10 == 0 and valid_set is not None:
                    # random subset
                    subset_ind = self.np_rng.choice(
                        valid_set.shape[0], min(1000, valid_set.shape[0]),
                        replace=False)

                    # pseudo-likelihood and delta F
                    df = self.free_energy_diff(shuffled_data, valid_set)
                    pl = self.compute_logpl(valid_set[subset_ind, :])

                    logger.debug('{} {} {}'.format(update_step, df, pl))

                    # # Other options:
                    # # make a histogram of the weights and relative increments
                    # w_histo = np.histogram(...)
                    # incr_histo = np.histogram(...)

            # run monitoring after epoch
            start_time = time.time()
            self.monitor_progress(train_data, valid_set, logger, isl=log_isl)
            monitoring_time += time.time() - start_time
            # save rbm state after every epoch (for later inspection)
            self.save('temp_rbm{:03d}'.format(epoch_index))
        print('Monitoring took {:.1f} min'.format(monitoring_time/60.))

    def monitor_progress(self, train_set, valid_set, logger, isl=False):
        # logger.info('LL-estimate with AIS: {:.3f}'.format(
        #     self.run_ais(valid_set, logger)))
        subset_ind = self.np_rng.choice(
            train_set.shape[0], min(5000, train_set.shape[0]), replace=False)
        logger.info('log-PL of training subset: {:.3f}'.format(
            self.compute_logpl(train_set[subset_ind])))
        if valid_set is not None:
            logger.info('log-PL of validation set: {:.3f}'.format(
                self.compute_logpl(valid_set)))
            logger.info('Free energy difference: {:.4f}'.format(
                self.free_energy_diff(train_set, valid_set)))
            if isl:
                subset_ind = self.np_rng.choice(
                    valid_set.shape[0], min(5000, valid_set.shape[0]), False)
                logger.info('LL-estimate with ISL: {:.3f}'.format(
                    self.estimate_loglik_isl(1e4, valid_set[subset_ind])))

    # free energy difference between training subset and validation set
    def free_energy_diff(self, train_set, valid_set):
        f_valid = self.free_energy(valid_set)
        f_train = self.free_energy(train_set[-len(valid_set):])
        return np.mean(f_valid) - np.mean(f_train)

    # ======== Functions for estimating the LL ========
    # with ISL-method
    def estimate_loglik_isl(self, n_samples, test_vis):
        if type(self) is CRBM:
            nv = self.n_inputs
        else:
            nv = self.n_visible
        vis_samples = self.draw_samples(n_samples, binary=True)[:, :nv]
        dm = isl.ISL_density_model()
        dm.fit(vis_samples, quick=True)
        # ISL needs binarized data
        return dm.avg_loglik(np.round(test_vis))

    # pseudo-likelihood like Bengio in his online tutorial
    def compute_logpl(self, test_set):
        # binarize images
        data = np.round(test_set)
        n_data = data.shape[0]
        # flip randomly one bit in each data vector
        flip_indices = self.np_rng.randint(data.shape[1], size=n_data)
        data_flip = data.copy()
        data_flip[np.arange(n_data), flip_indices] = 1 - \
            data[np.arange(n_data), flip_indices]

        # calculate free energies
        fe_data = self.free_energy(data)
        fe_data_flip = self.free_energy(data_flip)

        # from rbm tutorial (deeplearning.net)
        log_pl = self.n_visible * np.log(1. / (1. +
                                         np.exp(-(fe_data_flip - fe_data))))
        return np.average(log_pl)

    # AIS
    def estimate_partition_sum(self, n_runs, betas):
        # draw samples from the base model (uniform distr) and initialise logw
        samples = np.random.randint(2, size=(n_runs, self.n_visible))
        logw = 0

        # main AIS loop
        for beta in betas[1:-1]:
            # compute unnormalized probabilities p_k+1(v_k)
            expWh = np.exp(beta * (samples.dot(self.w) + self.hbias))
            logw += beta * samples.dot(self.vbias) + \
                np.sum(np.log(1 + expWh), axis=1)

            # apply transition operators
            samples = self.gibbs_vhv(samples, beta=beta)[1]

            # compute unnormalized probabilities p_k+1(v_k+1)
            expWh = np.exp(beta * (samples.dot(self.w) + self.hbias))
            logw -= beta * samples.dot(self.vbias) + \
                np.sum(np.log(1 + expWh), axis=1)

        # add target probability p_K(v_K)
        expWh = np.exp(samples.dot(self.w) + self.hbias)
        logw += samples.dot(self.vbias) + np.sum(np.log(1 + expWh), axis=1)

        r_ais = logsum(logw) - np.log(n_runs)
        # print(r_ais, np.log(np.average(np.exp(logw))))
        # numerical stability
        logw_avg = np.mean(logw)
        logstd_rais = np.log(np.std(np.exp(logw - logw_avg))) + logw_avg -\
            np.log(n_runs)/2
        logZ_base = self.n_visible * np.log(2)
        logZ = r_ais + logZ_base
        logZ_up = logsum([logstd_rais + np.log(3), r_ais]) + logZ_base
        logZ_down = logdiff([logstd_rais + np.log(3), r_ais]) + logZ_base
        return logZ, logstd_rais + logZ_base, logZ_up, logZ_down

    def run_ais(self, test_data, logger, train_data=None, n_runs=100,
                exact=False):
        if exact:
            # compute the true partition sum of the RBM (if possible)
            logZ_true = self.compute_partition_sum()
            avg_ll_true = np.mean(-self.free_energy(test_data)) - logZ_true
            logger.debug('True partition sum: {:.2f}'.format(logZ_true))
            logger.debug('True average loglik: {:.2f}'.format(avg_ll_true))

        # Use AIS to estimate the partition sum
        betas = np.concatenate((np.linspace(0, .5, 500, endpoint=False),
                                np.linspace(.5, .9, 10000, endpoint=False),
                                np.linspace(.9, 1., 4000)))
        # betas = np.linspace(0, 1, 20000)
        logZ_est, logstdZ, est_up, est_down = \
            self.estimate_partition_sum(n_runs, betas)

        # compute the estimated average log likelihood of a test set
        ll_est_test = np.mean(-self.free_energy(test_data)) - logZ_est
        logger.debug('Est. partition sum (+- 3*std): {:.2f}, {:.2f}, {:.2f}'
                     ''.format(logZ_est, est_up, est_down))
        logger.debug('Est. average loglik (test): {:.2f}'.format(ll_est_test))
        if train_data is not None:
            ll_est_train = np.mean(-self.free_energy(train_data)) - logZ_est
            logger.debug('Est. average loglik (train): {:.2f}'
                         ''.format(ll_est_train))
            return ll_est_test, ll_est_train
        return ll_est_test


class CRBM(RBM):
    """
        Training should work exactly the same, so the super methods are used.
        Class internals must be adapted, however, and some classification
        methods are added.
    """
    def __init__(self,
                 n_inputs, n_hidden, n_labels,
                 wv=None, wl=None,
                 input_bias=None, vbias=None, hbias=None, lbias=None,
                 numpy_seed=None,
                 dbm_factor=[1, 1]):
        # Set seed for reproducatbility
        self.np_rng = np.random.RandomState(numpy_seed)

        self.n_hidden = n_hidden
        self.n_visible = n_inputs + n_labels
        self.n_inputs = n_inputs
        self.n_labels = n_labels

        if wv is None:
            wv = .01*self.np_rng.randn(n_inputs, n_hidden)
        if wl is None:
            wl = .01*self.np_rng.randn(n_labels, n_hidden)
        # For best initialization see Hinton's guide; needs to be passed
        if input_bias is None:
            input_bias = np.zeros(n_inputs)
        if hbias is None:
            hbias = np.zeros(n_hidden)
        if lbias is None:
            lbias = np.zeros(n_labels)

        # shape of w: (n_visible + n_labels, n_hidden)
        self.w = np.concatenate((wv, wl), axis=0)
        self.wv = self.w[:self.n_inputs, :]
        self.wl = self.w[self.n_inputs:, :]

        # vbias includes the label bias when the CRBM is interpreted as an RBM
        # with additional label neurons in the visible layer
        if vbias is None:
            self.vbias = np.concatenate((input_bias, lbias))
        else:
            self.vbias = vbias
        self.lbias = self.vbias[n_inputs:]
        self.ibias = self.vbias[:n_inputs]
        self.hbias = hbias

        # This factor is needed during the layerwise DBM training.
        # Making it a instance variable is not good OOP-style but
        # a simple solution
        self.dbm_factor = dbm_factor

    def save(self, filename):
        rbm_dict = {
            'type': 'crbm',
            'n_inputs': self.n_inputs,
            'n_hidden': self.n_hidden,
            'n_labels': self.n_labels,
            'wv': self.wv,
            'wl': self.wl,
            'vbias': self.vbias,
            'hbias': self.hbias
        }
        with open(filename, 'wb') as output:
            cPickle.dump(rbm_dict, output, cPickle.HIGHEST_PROTOCOL)

    def to_rbm(self):
        return RBM(self, self.n_inputs, self.n_hidden,
                   w=self.wv, vbias=self.ibias, hbias=self.hbias)

    # # For the CDBM I need a special sampling method for the top layer, which
    # # is a CRBM.
    # def sample_v_given_h(self, h_in, beta=1.):
    #     if len(h_in.shape) == 1:
    #         h_in = np.expand_dims(h_in, 0)
    #     if hasattr(beta, "__len__"):
    #         # for ast we get an array with 'n_instances' entries
    #         beta = np.expand_dims(beta, 1)

    #     u_inp = beta * self.dbm_factor[1] * (h_in.dot(self.wv.T) + self.ibias)
    #     u_lab = beta * (h_in.dot(self.wl.T) + self.lbias)
    #     pi_on = 1./(1 + np.exp(-u_inp))
    #     pl_on = 1./(1 + np.exp(-u_lab))
    #     p_on = np.hstack((pi_on, pl_on))
    #     v_samples = (self.np_rng.rand(*p_on.shape) < p_on)*1.
    #     return [p_on.squeeze(), v_samples.squeeze()]

    def classify(self, v_data, class_prob=False):
        if len(v_data.shape) == 1:
            v_data = np.expand_dims(v_data, 0)

        # This is the analytical way described in LaRochelle et al. (2012)
        # Compute free energy of each class -> reuse hidden activations
        activation = v_data.dot(self.wv) + self.hbias
        free_energies = np.zeros((v_data.shape[0], self.n_labels))
        for y in range(self.n_labels):
            hidden_term = np.sum(np.log(1 +
                                 np.exp(self.wl[y, :] + activation)), axis=1)
            free_energies[:, y] = - (self.lbias[y] + hidden_term)
        # Compute probability from free energies
        # for numerical stability
        fe = free_energies - np.expand_dims(np.max(free_energies, axis=1), 1)
        p_class = np.exp(-fe) / np.expand_dims(np.sum(np.exp(-fe), axis=1), 1)

        if class_prob:
            return p_class
        return np.argmax(p_class, axis=1)

    # for neural sampling and later LIF, this is the only way to classify
    def classify_by_sampling(self, v_data):
        # sample h given v (100 times)
        label_samples = \
            self.sample_with_clamped_units(100, np.arange(self.n_inputs),
                                           v_data, binary=True)
        return np.argmax(np.sum(label_samples, axis=0), axis=1)

    def sample_from_label(self, label, n_samples):
        # assuming label is just a number
        bin_labels = np.zeros(self.n_labels)
        bin_labels[label] = 1
        clamped_ind = np.arange(self.n_inputs, self.n_visible)
        return self.sample_with_clamped_units(n_samples,
                                              clamped_ind=clamped_ind,
                                              clamped_val=bin_labels)

    def monitor_progress(self, train_set, valid_set, logger, isl=False):
        train_vis = train_set[:, :self.n_inputs]
        train_lab = train_set[:, self.n_inputs:]
        valid_vis = valid_set[:, :self.n_inputs]
        valid_lab = valid_set[:, self.n_inputs:]
        prediction = self.classify(train_vis)
        labels = np.argmax(train_lab, axis=1)
        logger.info('Correct classifications on training set: '
                    '{:.3f}'.format(np.average(prediction == labels)))
        if valid_set is not None:
            prediction = self.classify(valid_vis)
            labels = np.argmax(valid_lab, axis=1)
            logger.info('Correct classifications on validation set: '
                        '{:.3f}'.format(np.average(prediction == labels)))
            logger.info('Free energy difference: {:.4f}'.format(
                self.free_energy_diff(train_set, valid_set)))
            if isl:
                subset_ind = self.np_rng.choice(
                    valid_vis.shape[0], min(2000, valid_vis.shape[0]), False)
                # compare to LL-estimate
                logger.info('LL-estimate with ISL: {:.3f}'.format(
                    self.estimate_loglik_isl(1e4, valid_vis[subset_ind])))

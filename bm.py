# General Boltzmann machine
from __future__ import division
from __future__ import print_function
import numpy as np


# class Bm(object):
#     def __init__(self, n_units, w, b, seed=None):
#         # np.random.seed(seed)
#         self.state = np.random.randint(2, size=n_units)
#         self.w = w
#         self.b = b

#     def energy(self):
#         return -(state.dot(self.w.dot(state)) + self.b.dot(state))

#     # Gibbs sampling
#     def draw_samples(self, n_samples):
#         samples = np.empty((n_samples, self.state.size))
#         z_curr = self.state
#         for t in range(1, n_samples):
#             for k in range(self.state.size):
#                 # update z_k
#                 u_k = np.dot(self.w[k, :], z_curr) + self.b[k]
#                 # sigmoid transition probability
#                 p_t = 1./(1 + np.exp(-u_k))
#                 z_curr[k] = (np.random.rand() < p_t)*1.
#             samples[t, :] = z_curr
#         return samples


class Rbm(object):
    def __init__(
                self,
                n_visible,
                n_hidden,
                rate=.01,
                n_epochs=5,
                batch_size=10,
                w=None,
                bv=None,
                bh=None):
        # shape of w: (n_visible, n_hidden)
        self.n_hidden = n_hidden
        self.n_visible = n_visible

        if w is None:
            w = .01*np.random.randn(n_visible, n_hidden)

        if bv is None:
            # For best initialization see Hinton's guide; needs to be passed
            bv = np.zeros(n_visible)
        if bh is None:
            bh = np.zeros(n_hidden)

        self.rate = rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.w = w
        self.bv = bv
        self.bh = bh

    def energy(self, v, h):
        return - np.einsum('ik,il,kl->i', v, h, self.w) - \
            v.dot(self.bv) - h.dot(self.bh)

    def free_energy(self, v_data):
        if len(v_data.shape) == 1:
            v_data = np.expand_dims(v_data, 0)
        activation = v_data.dot(self.w) + self.bh
        hidden_term = np.sum(np.log(1 + np.exp(activation)), axis=1)
        return - v_data.dot(self.bv) - hidden_term

    # generalized method for sampling quickly from many chains (numpy).
    # AST still causes problems: for single chain indexing with k, for
    # multiple chains need to figure out how to store samples (asynchronous)
    def draw_samples(self, n_samples, v_start=None, n_chains=1, binary=False,
                     ast=False):
        if ast:
            return self.draw_samples_ast(n_samples, binary=binary)

        samples = np.empty((n_chains, n_samples, self.n_visible +
                            self.n_hidden))
        if v_start is None:
            v_start = np.random.randint(2, size=(n_chains, self.n_visible))

        ax = 0
        if n_chains > 1:
            ax = 1

        v_curr = v_start
        h_curr = 0
        for t in range(n_samples):
            # schedule for gamma?
            # gamma = 10/(1 + t/1e2)
            if not ast:
                pv, v_curr, ph, h_curr = self.gibbs_vhv(v_curr)
                if binary:
                    samples[:, t, :] = \
                        np.concatenate((v_curr, h_curr), axis=ax)
                else:
                    samples[:, t, :] = np.concatenate((pv, ph), axis=ax)
        return samples.squeeze()

    # doesn't work for multiple chains yet
    def draw_samples_ast(self, n_samples, v_start=None, binary=False):
        # setup ast
        betas = np.linspace(1., .5, 10)
        g_weights = np.ones(betas.size)
        gamma = 10

        samples = np.empty((n_samples, self.n_visible + self.n_hidden))
        if v_start is None:
            v_start = np.random.randint(2, size=self.n_visible)
        v_curr = v_start
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

    def sample_with_clamped_units(self, n_samples, clamped_ind, clamped_val,
                                  binary=False):
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

        # Create an extra RBM for the h->v step for efficiency
        unclamped_ind = np.setdiff1d(np.arange(self.n_visible), clamped_ind)

        unclamped_rbm = Rbm(n_visible=unclamped_ind.size,
                            n_hidden=self.n_hidden,
                            w=self.w[unclamped_ind, :],
                            bv=self.bv[unclamped_ind],
                            bh=self.bh)

        if len(clamped_val.shape) == 1:
            clamped_val = np.expand_dims(clamped_val, 0)

        samples = np.zeros((n_samples, clamped_val.shape[0],
                            unclamped_ind.size))
        v_curr = np.zeros((clamped_val.shape[0], self.n_visible))
        v_curr[:, clamped_ind] = clamped_val

        # sample starting from  multiple initial values
        for t in range(n_samples):
            _, h_samples = self.sample_h_given_v(v_curr)
            unclamped_mean, unclamped_samples = \
                unclamped_rbm.sample_v_given_h(h_samples)
            v_curr[:, unclamped_ind] = unclamped_samples
            if binary:
                samples[t] = unclamped_samples
            else:
                samples[t] = unclamped_mean

        return samples.squeeze()

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
        k_proposal = k_start + ((np.random.rand(k_start.size) < .5)*1. - .5)*2
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
        accepted = (p_accept > np.random.rand(p_accept.size))
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

        u = beta * (v_in.dot(self.w) + self.bh)
        p = 1./(1 + np.exp(-u))
        h_out = (np.random.rand(v_in.shape[0], self.n_hidden) < p)*1
        return [p.squeeze(), h_out.squeeze()]

    def sample_v_given_h(self, h_in, beta=1.):
        if len(h_in.shape) == 1:
            h_in = np.expand_dims(h_in, 0)
        if hasattr(beta, "__len__"):
            # for ast we get an array with 'n_instances' entries
            beta = np.expand_dims(beta, 1)

        u = beta * (h_in.dot(self.w.T) + self.bv)
        p = 1./(1 + np.exp(-u))
        v_out = (np.random.rand(h_in.shape[0], self.n_visible) < p)*1
        return [p.squeeze(), v_out.squeeze()]

    # compute CSL; this method can be overwritten for CRBM
    # speed up by numpy array operations is limited by memory consumption of
    # resulting arrays
    def compute_csl(self, valid_set):
        # compute CSL for validation set
        n_data = valid_set.shape[0]
        n_samples = 1000

        h_samples = self.draw_samples(100 + n_samples,
                                      n_chains=n_data)[:, 100:,
                                                       -self.n_hidden:]
        v_activation = h_samples.reshape((n_data * n_samples,
                                          self.n_hidden)).dot(self.w.T) +\
            self.bv
        model_pv = 1./(1 + np.exp(-v_activation.reshape((n_samples,
                                                         n_data,
                                                         self.n_visible))))
        data_prob = np.prod((2*model_pv - 1)*valid_set + 1 - model_pv, axis=2)
        # I use a linear interpolation between v=0 and v=1
        csl = np.log(np.average(data_prob, axis=1))

        return np.average(csl)

    def compute_logpl(self, valid_set):
        # pseudo-likelihood:
        # binarize images
        data = np.round(valid_set)
        n_data = data.shape[0]
        # flip randomly one bit in each data vector
        flip_indices = np.random.randint(data.shape[1], size=n_data)
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

    def compute_grad_cdn(self,
                         batch,
                         n_steps,
                         persistent=None,
                         regularizer=0,
                         cast_variables=None):
        """
            This method computes the gradient of the complete batch.
            persistent: state of the hidden variables in previous state
             of persistent chain
            cast_variables: dictionary with additional cast stuff;
             is updated inside this method
        """

        # data must be normalized to [0,1]
        # v0 = (np.random.rand(*batch.shape) < batch)*1.
        # v0 = (batch > .5)*1.
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

    def train(self, train_data, cd_steps=1, persistent=False, cast=False,
              valid_set=None, momentum=0.):
        # initializations
        n_instances = train_data.shape[0]
        n_batches = int(np.ceil(n_instances/self.batch_size))
        n_updates = int(n_instances/self.batch_size*self.n_epochs)
        shuffled_data = train_data[np.random.permutation(n_instances), :]
        pchain_init = None
        prev_grad = [np.zeros_like(self.w), np.zeros_like(self.bv),
                     np.zeros_like(self.bh)]
        pl = []
        delta_f = []
        steps = []

        # regularization was not necessary for my applications
        regularizer = 0
        # ++++++++++++
        # new for cast
        cast_variables = None
        swap_state = None
        if cast:
            persistent = True
            # same number of fast and slow chains is implicit everywhere
            if self.batch_size % 2 == 1:
                print('CAST implementation does not support odd batch sizes')
                return
            n_fast_chains = self.batch_size // 2
            n_betas = 10
            cast_variables = {
                'k_start': np.zeros(n_fast_chains, dtype=int),
                'g_weights': np.ones((n_fast_chains, n_betas)),
                'gamma': 10.,
                'betas': np.linspace(1., .8, n_betas)
            }
            swap_state = np.zeros((n_fast_chains, self.n_hidden))
        # +++++++++++++

        for epoch_index in range(self.n_epochs):
            print('Epoch {}'.format(epoch_index + 1))

            for batch_index in range(n_batches):
                update_step = batch_index + n_batches * epoch_index

                # Other rate schedules are possible
                rate = self.rate * 2000 / (2000 + update_step)

                # pick mini-batch randomly; actually
                # http://leon.bottou.org/publications/pdf/tricks-2012.pdf
                # says that shuffling and sequential picking is also ok
                batch = shuffled_data[batch_index*self.batch_size:
                                      min((batch_index + 1)*self.batch_size,
                                          n_instances), :]

                # compute "gradient" on batch
                grad, reconstruction = \
                    self.compute_grad_cdn(batch, cd_steps,
                                          persistent=pchain_init,
                                          regularizer=regularizer,
                                          cast_variables=cast_variables)

                # update parameters including momentum
                self.w += rate * (momentum*prev_grad[0] + (1 - momentum) *
                                  (grad[0] - regularizer * self.w))
                self.bv += rate * (momentum*prev_grad[1] +
                                   (1 - momentum) * grad[1])
                self.bh += rate * (momentum*prev_grad[2] +
                                   (1 - momentum) * grad[2])
                prev_grad = grad

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

                if update_step % (1+n_updates//500) == 0 and \
                   valid_set is not None:
                    # Monitor the free energy difference between training
                    # subset and validation set
                    f_valid = self.free_energy(valid_set)
                    f_train = \
                        self.free_energy(shuffled_data[-valid_set.shape[0]:])
                    delta_f.append(np.mean(f_valid) - np.mean(f_train))

                    # random subset
                    subset_ind = np.random.randint(valid_set.shape[0],
                                                   size=1000)
                    # # CSL monitoring -> much too slow
                    # csl.append(self.compute_csl(valid_set[subset_ind, :]))

                    # pseudo-likelihood
                    pl.append(self.compute_logpl(valid_set[subset_ind, :]))
                    steps.append(update_step)

            if valid_set is not None:
                self.monitor_progress(train_data, valid_set)

        return steps, delta_f, pl

    def monitor_progress(self, valid_set):
        pass
        # print('Log-PL of validation set: '
        #       '{}'.format(self.compute_logpl(valid_set)))
        # print('CSL of validation set: '
        #       '{}'.format(self.compute_csl(valid_set)))


class ClassRbm(Rbm):
    """
        Training should work exactly the same, so the super methods are used.
        Class internals must be adapted, however, and some classification
        methods are added.
    """
    def __init__(
                 self,
                 n_inputs,
                 n_hidden,
                 n_labels,
                 wv=None,
                 wl=None,
                 bias_vis=None,
                 bias_hid=None,
                 bias_lab=None,
                 rate=.01,
                 n_epochs=5,
                 batch_size=10,):
        self.n_hidden = n_hidden
        self.n_visible = n_inputs + n_labels
        self.n_inputs = n_inputs
        self.n_labels = n_labels

        if wv is None:
            wv = .01*np.random.randn(n_inputs, n_hidden)
        if wl is None:
            wl = .01*np.random.randn(n_labels, n_hidden)
        # For best initialization see Hinton's guide; needs to be passed
        if bias_vis is None:
            bias_vis = np.zeros(n_inputs)
        if bias_hid is None:
            bias_hid = np.zeros(n_hidden)
        if bias_lab is None:
            bias_lab = np.zeros(n_labels)

        # shape of w: (n_visible + n_labels, n_hidden)
        self.w = np.concatenate((wv, wl), axis=0)
        self.wv = self.w[:self.n_inputs, :]
        self.wl = self.w[self.n_inputs:, :]

        # the bias names are a bit unfortunate...
        self.bv = np.concatenate((bias_vis, bias_lab))
        self.bias_lab = self.bv[n_inputs:]
        self.bias_vis = self.bv[:n_inputs]
        self.bh = bias_hid

        self.rate = rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def classify(self, v_data):
        if len(v_data.shape) == 1:
            v_data = np.expand_dims(v_data, 0)

        # This is the analytical way described in LaRochelle et al. (2012)
        # Compute free energy of each class -> reuse hidden activations
        activation = v_data.dot(self.wv) + self.bh
        free_energies = np.zeros((v_data.shape[0], self.n_labels))
        for y in range(self.n_labels):
            hidden_term = np.sum(np.log(1 +
                                 np.exp(self.wl[y, :] + activation)), axis=1)
            free_energies[:, y] = - (self.bias_lab[y] + hidden_term)
        # Compute probability from free energies
        # for numerical stability
        fe = free_energies - np.expand_dims(np.max(free_energies, axis=1), 1)
        p_class = np.exp(-fe) / np.expand_dims(np.sum(np.exp(-fe), axis=1), 1)

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

    def monitor_progress(self, train_set, valid_set=None):
        prediction = self.classify(train_set[:, :self.n_inputs])
        labels = np.argmax(train_set[:, self.n_inputs:], axis=1)
        print('Correct classifications on training set:'
              '{:.3f}'.format(np.average(prediction == labels)), end='')
        if valid_set is not None:
            prediction = self.classify(valid_set[:, :self.n_inputs])
            labels = np.argmax(valid_set[:, self.n_inputs:], axis=1)
            print('; validation set:'
                  '{:.3f}'.format(np.average(prediction == labels)))
        else:
            print('')

# Deep Boltzmann machine
from __future__ import division
from __future__ import print_function
import numpy as np
from copy import deepcopy
from rbm import RBM, CRBM


class DBM(object):
    def __init__(self, layer_sizes, vbias_init=None,
                 numpy_seed=None):
        if len(layer_sizes) < 2:
            'Please use the RBM class for just one layer'
        # Set seed for reproducatbility
        self.rng = np.random.RandomState(numpy_seed)
        # layers should be a list (nv, nh1, nh2, ..., nhl)
        self.n_visible = layer_sizes[0]
        self.hidden_layers = layer_sizes[1:]
        self.n_layers = len(self.hidden_layers)
        self.weights = []
        self.hbiases = []
        self.rbms = []
        if vbias_init is None:
            vbias_init = np.zeros(self.n_visible)
        self.vbias = vbias_init

        # initialize weights and biases
        for i in range(self.n_layers):
            if i == 0:
                dbm_factor = [2, 1]
                input_size = self.n_visible
                input_bias = self.vbias
            else:
                dbm_factor = [1, 1]
                input_size = self.hidden_layers[i - 1]
                input_bias = self.hbiases[i - 1]
                if i == self.n_layers - 1:
                    dbm_factor = [1, 2]

            self.hbiases.append(np.zeros(self.hidden_layers[i]))
            w = .01*self.rng.randn(input_size, self.hidden_layers[i])
            self.weights.append(w)

            # The RBMs share the parameters with the DBM; unlike Bengio, I
            # share the biases as well
            rbm = RBM(input_size, self.hidden_layers[i],
                      w=self.weights[i],
                      vbias=input_bias,
                      hbias=self.hbiases[i],
                      dbm_factor=dbm_factor)
            self.rbms.append(rbm)

    # greedy layerwise training
    # all training hyperparameters from RBM.train can be used as kwargs
    def train(self, train_data, **kwargs):
        curr_input = train_data
        for i, layer in enumerate(self.rbms):
            print('Training layer {} of {}'.format(i + 1,
                                                   self.n_layers))
            layer.train(curr_input, **kwargs)
            # If binary samples are desired, use 2nd return value
            # For the top layer, 2W will be used, but this is desired, right?
            curr_input, _ = layer.sample_h_given_v(curr_input)
            kwargs['valid_set'] = None
            # The weights of the inner layers must be reweighted because they
            # receive inputs from two layers in the DBM
            if i > 0 and i < len(self.rbms) - 1:
                self.weights[i] *= .5

    # conditional sampling functions are vectorized
    # beta factor is used in AST
    def sample_inner_cond(self, index, h_upper, h_lower, beta=1.):
        act = h_upper.dot(self.weights[index + 1].T) + \
              h_lower.dot(self.weights[index]) + self.hbiases[index]
        p_on = 1./(1 + np.exp(-beta*act))
        h_samples = (self.rng.rand(*p_on.shape) < p_on)*1.
        return [p_on.squeeze(), h_samples.squeeze()]

    def sample_outer_cond(self, h_in, visible=True, beta=1.):
        if visible:
            bias = self.vbias
            weight = self.weights[0].T
        else:
            bias = self.hbiases[-1]
            weight = self.weights[-1]

        act = h_in.dot(weight) + bias
        p_on = 1./(1 + np.exp(-beta*act))
        samples = (self.rng.rand(*p_on.shape) < p_on)*1.
        return [p_on.squeeze(), samples.squeeze()]

    # update even-numbered layers
    def update_even(self, state, mean, clamped=None, clamped_val=None, beta=1):
        for i in np.arange(0, self.n_layers + 1, 2):
            if i == 0:
                mean[i], state[i] = \
                    self.sample_outer_cond(state[1], visible=True, beta=beta)
            # special sampling for top layer
            elif i == self.n_layers:
                mean[i], state[i] = \
                    self.sample_outer_cond(state[i - 1], visible=False,
                                           beta=beta)
            else:
                mean[i], state[i] = \
                    self.sample_inner_cond(i - 1, h_upper=state[i + 1],
                                           h_lower=state[i - 1], beta=beta)
            # reset clamped units
            if clamped is not None and clamped[i] is not None:
                # state[i] = state[i].astype(float)
                if len(state[i].shape) == 2:
                    state[i][:, clamped[i]] = clamped_val[i]
                    mean[i][:, clamped[i]] = clamped_val[i]
                if len(state[i].shape) == 1:
                    state[i][clamped[i]] = clamped_val[i]
                    mean[i][clamped[i]] = clamped_val[i]

        return state

    # update odd numbered layers
    def update_odd(self, state, mean, clamped=None, clamped_val=None, beta=1):
        for i in np.arange(1, self.n_layers + 1, 2):
            # special sampling for top layer
            if i == self.n_layers:
                mean[i], state[i] = \
                    self.sample_outer_cond(state[i - 1], visible=False,
                                           beta=beta)
            else:
                mean[i], state[i] = \
                    self.sample_inner_cond(i - 1, h_upper=state[i + 1],
                                           h_lower=state[i - 1], beta=beta)
            # reset clamped units
            if clamped is not None and clamped[i] is not None:
                # float cast is needed because binaty samples
                # state[i] = state[i].astype(float)
                if len(state[i].shape) == 2:
                    state[i][:, clamped[i]] = clamped_val[i]
                    mean[i][:, clamped[i]] = clamped_val[i]
                if len(state[i].shape) == 1:
                    state[i][clamped[i]] = clamped_val[i]
                    mean[i][clamped[i]] = clamped_val[i]

        return state

    # make a gibbs step starting from the state of v and all other even
    # numbered layers.
    def gibbs_from_v(self, state, clamped=None, clamped_val=None, beta=1.):
        assert len(state) == self.n_layers + 1
        means = [0] * len(state)
        state = self.update_odd(state, means, clamped, clamped_val, beta)
        state = self.update_even(state, means, clamped, clamped_val, beta)
        return state, means

    # make a gibbs step starting from the state of h1 and all other odd
    # numbered layers.
    def gibbs_from_h(self, state, clamped=None, clamped_val=None, beta=1.):
        assert len(state) == self.n_layers + 1
        means = [0] * len(state)
        state = self.update_even(state, means, clamped, clamped_val, beta)
        state = self.update_odd(state, means, clamped, clamped_val, beta)
        return state, means

    # draw visible samples using Gibbs sampling
    def draw_samples(self, n_samples, n_chains=1, v_init=None, binary=False,
                     clamped=None, clamped_val=None, layer_ind=0):
        # clamped are the indices of the clamped units:
        # clamped = list(layers, indices_in layer)
        # clamped_val = list(layers, values_in_layer)
        # list entry None for layers w/o clamped units
        # layer_ind: specifies from which layer samples are returned

        # initialize the chain
        if v_init is None:
            if clamped is not None:
                n_chains = clamped_val.shape[0]
            if binary:
                v_init = self.rng.randint(2, size=(n_chains, self.n_visible)).\
                    astype(float)
            else:
                v_init = self.rng.rand(n_chains, self.n_visible)
        elif len(v_init.shape) == 2:
            n_chains = v_init.shape[0]

        curr_state = [v_init]
        for i, size in enumerate(self.hidden_layers):
            if binary:
                curr_state.append(self.rng.randint(2, size=(n_chains, size)).
                                  astype(float))
            else:
                curr_state.append(self.rng.rand(n_chains, size))

        # initialize clamped units correctly
        if clamped is not None:
            if v_init is not None:
                assert n_chains == clamped_val.shape[0]
            for i, i_clamped in enumerate(clamped):
                if i_clamped is not None:
                    curr_state[i][:, i_clamped] = clamped_val[i]

        n_samples = int(n_samples)
        if layer_ind == 0:
            sample_size = self.n_visible
        elif layer_ind == 'all':
            sample_size = self.n_visible + np.sum(self.hidden_layers)
        else:
            sample_size = self.hidden_layers[layer_ind - 1]
        samples = np.empty((n_samples, n_chains, sample_size))

        # draw samples and save for one layer
        for t in range(n_samples):
            curr_state, curr_means = \
                self.gibbs_from_v(curr_state, clamped, clamped_val)
            if binary:
                if layer_ind == 'all':
                    samples[t] = np.concatenate(deepcopy(curr_state), axis=1)
                else:
                    samples[t] = curr_state[layer_ind].copy()
            else:
                if layer_ind == 'all':
                    samples[t] = np.concatenate(deepcopy(curr_means), axis=1)
                else:
                    samples[t] = curr_means[layer_ind].copy()
        return samples.squeeze()


# Sampling for CDBM works exactly as DBM, just with an additional label layer
# at the top; training is modified
class CDBM(DBM):
    def __init__(self, layer_sizes, labels, vbias_init=None, numpy_seed=None):
        if len(layer_sizes) < 3:
            'Please use the RBM class for just one layer'
        # Set seed for reproducatbility
        self.rng = np.random.RandomState(numpy_seed)
        # layers should be a list (nv, nh1, nh2, ..., nhl)
        self.n_visible = layer_sizes[0]
        self.hidden_layers = layer_sizes[1:] + [labels]
        self.n_layers = len(self.hidden_layers)
        self.weights = []
        self.hbiases = []
        if vbias_init is None:
            vbias_init = np.zeros(self.n_visible)
        self.vbias = vbias_init
        # initialize weights and biases
        for i in range(self.n_layers):
            if i == 0:
                input_size = self.n_visible
            else:
                input_size = self.hidden_layers[i - 1]

            self.hbiases.append(np.zeros(self.hidden_layers[i]))
            w = .01*self.rng.randn(input_size, self.hidden_layers[i])
            self.weights.append(w)

    # greedy layerwise training
    # all training hyperparameters from RBM.train can be used as kwargs
    def train(self, train_data, train_targets, **kwargs):
        if len(train_targets.shape) == 1:
            print('Please pass the labels in one-hot representation.')
            return
        curr_input = train_data
        for i in range(self.n_layers - 1):
            print('Training layer {} of {}'.format(i + 1, self.n_layers - 1))
            if i == 0:
                dbm_factor = [2, 1]
                input_size = self.n_visible
                input_bias = self.vbias
            else:
                dbm_factor = [1, 1]
                input_size = self.hidden_layers[i - 1]
                input_bias = self.hbiases[i - 1]
                if i == self.n_layers - 2:
                    dbm_factor = [1, 2]

            # The RBMs share the parameters with the DBM
            layer = RBM(input_size, self.hidden_layers[i],
                        w=self.weights[i],
                        vbias=input_bias,
                        hbias=self.hbiases[i],
                        dbm_factor=dbm_factor)
            if i == self.n_layers - 2:
                curr_input = np.hstack((curr_input, train_targets))
                # the top layer must be a CRBM
                layer = CRBM(input_size, self.hidden_layers[i],
                             self.hidden_layers[i + 1],
                             wv=self.weights[i], wl=self.weights[i + 1].T,
                             input_bias=input_bias, hbias=self.hbiases[i],
                             lbias=self.hbiases[i + 1],
                             dbm_factor=dbm_factor)

            layer.train(curr_input, **kwargs)
            if i == self.n_layers - 2:
                # the CRBM does not operate on references
                self.weights[i] = layer.wv
                self.weights[i + 1] = layer.wl.T
                self.hbiases[i - 1] = layer.ibias
                self.hbiases[i] = layer.hbias
                self.hbiases[i + 1] = layer.lbias
            else:
                assert layer.w is self.weights[i]

            # If binary samples are desired, use 2nd return value
            curr_input, _ = layer.sample_h_given_v(curr_input)
            kwargs['valid_set'] = None

            # The weights of the inner layers must be reweighted because they
            # receive inputs from two layers in the DBM
            if i > 0 and i < self.n_layers - 2:
                self.weights[i] *= .5

    def train_mf(self, train_data, train_targets, n_epochs=5, batch_size=10,
                 lrate=.01, cd_steps=5, valid_set=None, momentum=0,
                 weight_cost=1e-5, filename='train_log.txt'):

        # initializations
        n_instances = train_data.shape[0]
        n_batches = int(np.ceil(n_instances/batch_size))
        rand_perm = self.rng.permutation(n_instances)
        shuffled_data = train_data[rand_perm]
        shuffled_targets = train_targets[rand_perm]
        initial_lrate = lrate
        wincr = [0] * self.n_layers
        hbincr = [0] * (self.n_layers - 1)
        vbincr = 0
        lbincr = 0

        # initialize persistent state
        pinit = self.rng.choice(shuffled_data.shape[0], batch_size,
                                replace=False)
        pers_state = [(self.rng.rand(batch_size, self.n_visible) <
                       shuffled_data[pinit])*1.]
        for i, size in enumerate(self.hidden_layers):
            hidprob = 1/(1 + np.exp(-pers_state[-1].dot(2 * self.weights[i]) -
                                    self.hbiases[i]))
            pers_state.append((self.rng.rand(*hidprob.shape) < hidprob)*1.)

        # log monitoring quantities in this file
        log_file = open(filename, 'w')
        log_file.write('---------------------------------------------\n')
        log_file.write('layers: {}\nBatch size: {}\nLearning_rate: {}\n'
                       'CD-steps: {}\n'.format(self.hidden_layers, batch_size,
                                               lrate, cd_steps))
        for epoch in range(n_epochs):
            print('Epoch {}'.format(epoch + 1))
            # if momentum != 0 and epoch > 5:
            #     momentum = .9
            err_sum = 0
            for batch_index in range(n_batches):
                # Other lrate schedules are possible
                update_step = batch_index + n_batches * epoch
                lrate = initial_lrate * 2000 / (2000 + update_step)
                # lrate /= (1.000015**(epoch*600))

                # pick mini-batch randomly
                start = batch_index*batch_size
                end = min((batch_index + 1)*batch_size, n_instances)
                batch = shuffled_data[start:end]
                batch_targets = shuffled_targets[start:end]
                n_batch = batch.shape[0]

                # Start of positive phase
                # Compute approximate posterior and from it positive gradient
                posteriors = self.get_mf_posterior(batch, batch_targets)
                gradw = [0] * self.n_layers
                gradhb = [0] * (self.n_layers - 1)
                gradvb = 0
                gradlb = 0

                for l in range(self.n_layers - 1):
                    if l == 0:
                        gradw[l] += batch.T.dot(posteriors[l]) / n_batch
                        gradvb += np.mean(batch, axis=0)
                    else:
                        gradw[l] += \
                            posteriors[l-1].T.dot(posteriors[l]) / n_batch
                    gradhb[l] += np.mean(posteriors[l], axis=0)

                # label layer
                gradlb += np.mean(batch_targets, axis=0)
                gradw[-1] += posteriors[-1].T.dot(batch_targets) / n_batch
                # End of positive phase

                # Reconstruction error
                act = posteriors[0].dot(self.weights[0].T) + self.vbias
                recon_batch = 1/(1 + np.exp(-act))
                err_sum += np.sum((batch - recon_batch)**2)

                # Start of negative phase
                # Sample from model distribution and estimate model average
                for k in range(cd_steps):
                    pers_state, pers_mean = self.gibbs_from_v(pers_state)
                for l in range(self.n_layers):
                    if l == 0:
                        gradvb -= np.mean(pers_mean[0], axis=0)
                    if l == self.n_layers - 1:
                        gradlb -= np.mean(pers_mean[-1], axis=0)
                    else:
                        gradhb[l] -= np.mean(pers_mean[l+1], axis=0)
                    gradw[l] -= pers_mean[l].T.dot(pers_mean[l+1]) / batch_size

                # End of negative phase

                # Update parameters
                vbincr = momentum * vbincr + lrate * gradvb
                lbincr = momentum * lbincr + lrate * gradlb
                for l in range(self.n_layers - 1):
                    wincr[l] = momentum * wincr[l] + \
                        lrate * (gradw[l] - weight_cost * self.weights[l])
                    hbincr[l] = momentum * hbincr[l] + lrate * gradhb[l]
                    self.weights[l] += wincr[l]
                    self.hbiases[l] += hbincr[l]
                self.hbiases[-1] += lbincr
                self.vbias += vbincr
            log_file.write('Reconstruction error: {}\n'.format(err_sum))
            self.monitor_progress((train_data, train_targets), valid_set,
                                  log_file)

    def get_mf_posterior(self, data, targets=None, iterations=10):
        # compute some data-dependent quantities in advance
        data_bias = data.dot(self.weights[0])
        if targets is not None:
            lab_bias = targets.dot(self.weights[-1].T)

        # initialize the posterior distributions\
        totin = data.dot(self.weights[0]) + self.hbiases[0]
        mus = [1/(1 + np.exp(-totin))]
        for l in range(1, self.n_layers):
            if l == self.n_layers - 2 and type(self) is CDBM:
                # label layer
                totin = mus[l - 1].dot(self.weights[l]) + self.hbiases[l] + \
                    lab_bias
                mus.append(1/(1 + np.exp(-totin)))
                break
            else:
                totin = mus[l - 1].dot(self.weights[l]) + self.hbiases[l]
                mus.append(1/(1 + np.exp(-totin)))

        # do mean field updates until convergence
        for n in range(iterations):
            diff_h = 0
            for l in range(len(mus)):
                mu_old = mus[l]
                if type(self) is CDBM and l == self.n_layers - 2:
                    totin = mus[l - 1].dot(self.weights[l]) + lab_bias + \
                        self.hbiases[l]
                    mus[l] = 1/(1 + np.exp(-totin))
                    break
                if l == 0:
                    totin = mus[l + 1].dot(self.weights[l + 1].T) + \
                                data_bias + self.hbiases[l]
                elif l == self.n_layers - 1:
                    totin = mus[l - 1].dot(self.weights[l]) + self.hbiases[l]
                else:
                    totin = mus[l - 1].dot(self.weights[l]) + \
                        mus[l + 1].dot(self.weights[l + 1].T) + self.hbiases[l]
                mus[l] = 1/(1 + np.exp(-totin))
                diff_h += np.mean(np.abs(mus[l] - mu_old))

            if diff_h < 1e-7 * data.shape[0]:
                break
        return mus

    def classify(self, v_data, class_prob=False):
        # draw samples with clamped visible units and argmax the label units
        clamped = [None] * (1 + self.n_layers)
        clamped[0] = np.arange(self.n_visible)
        clamped_val = [None] * (1 + self.n_layers)
        burn_in = 10

        if len(v_data.shape) == 1:
            v_data = np.expand_dims(v_data, 0)

        clamped_val[0] = v_data
        samples = self.draw_samples(burn_in + 100, n_chains=v_data.shape[0],
                                    clamped=clamped,
                                    clamped_val=clamped_val,
                                    layer_ind=self.n_layers)[burn_in:]
        labprobs = np.average(samples, axis=0)
        if class_prob:
            return labprobs
        else:
            return np.argmax(labprobs, axis=1)

    def monitor_progress(self, train_set, valid_set, output_file):
        subt = self.rng.choice(train_set[0].shape[0], 1000, replace=False)
        prediction = self.classify(train_set[0][subt])
        labels = np.argmax(train_set[1][subt], axis=1)
        s = 'Correct classifications on training set: '\
            '{:.3f}'.format(np.average(prediction == labels))
        output_file.write(s)

        if valid_set is not None:
            subv = self.rng.choice(valid_set[0].shape[0], 1000,
                                   replace=False)
            prediction = self.classify(valid_set[0][subv])
            labels = np.argmax(valid_set[1][subv], axis=1)
            s = '; validation set: '\
                '{:.3f}'.format(np.average(prediction == labels))
            output_file.write(s + '\n')
        else:
            output_file.write('\n')

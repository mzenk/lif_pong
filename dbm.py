# Deep Boltzmann machine
from __future__ import division
from __future__ import print_function
import numpy as np
from rbm import RBM, CRBM


class DBM(object):
    def __init__(self, layer_sizes, vbias_init=None,
                 numpy_seed=None):
        if len(layer_sizes) < 2:
            'Please use the RBM class for just one layer'
        # Set seed for reproducatbility
        self.np_rng = np.random.RandomState(numpy_seed)
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
            w = .01*self.np_rng.randn(input_size, self.hidden_layers[i])
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
    def sample_inner_cond(self, index, h_upper, h_lower):
        act = h_upper.dot(self.weights[index + 1].T) + \
              h_lower.dot(self.weights[index]) + self.hbiases[index]
        p_on = 1./(1 + np.exp(-act))
        h_samples = (self.np_rng.rand(self.hidden_layers[index]) < p_on)*1
        return [p_on.squeeze(), h_samples.squeeze()]

    def sample_outer_cond(self, h_in, visible=True):
        if visible:
            layer_size = self.n_visible
            bias = self.vbias
            weight = self.weights[0].T
        else:
            layer_size = self.hidden_layers[-1]
            bias = self.hbiases[-1]
            weight = self.weights[-1]

        act = h_in.dot(weight) + bias
        p_on = 1./(1 + np.exp(-act))
        samples = (self.np_rng.rand(layer_size) < p_on)*1
        return [p_on.squeeze(), samples.squeeze()]

    # make a gibbs step starting from the state of v and all other even
    # numbered layers. _state_ is passed by reference!
    def gibbs_from_v(self, state, clamped=None, clamped_val=None):
        assert len(state) == self.n_layers + 1
        means = [0] * len(state)
        # the loops can be parallelised -> add if necessary
        # update odd numbered layers
        for i in np.arange(1, self.n_layers + 1, 2):
            # special sampling for top layer
            if i == self.n_layers:
                means[i], state[i] = \
                    self.sample_outer_cond(state[i - 1], visible=False)
            else:
                means[i], state[i] = \
                    self.sample_inner_cond(i - 1, h_upper=state[i + 1],
                                           h_lower=state[i - 1])
            # reset clamped units
            if clamped is not None and clamped[i] is not None:
                state[i] = state[i].astype(float)
                state[i][clamped[i]] = clamped_val[i]

        # update even numbered layers
        for i in np.arange(0, self.n_layers + 1, 2):
            if i == 0:
                means[i], state[i] = \
                    self.sample_outer_cond(state[1], visible=True)
            # special sampling for top layer
            elif i == self.n_layers:
                means[i], state[i] = \
                    self.sample_outer_cond(state[i - 1], visible=False)
            else:
                means[i], state[i] = \
                    self.sample_inner_cond(i - 1, h_upper=state[i + 1],
                                           h_lower=state[i - 1])
            # reset clamped units
            if clamped is not None and clamped[i] is not None:
                state[i] = state[i].astype(float)
                state[i][clamped[i]] = clamped_val[i]
        # # sequential version
        # for i in np.arange(self.n_layers, -1, -1):
        #     # special sampling for top and bottom layer
        #     if i == 0:
        #         means[i], state[i] = \
        #             self.sample_outer_cond(state[i + 1], visible=True)
        #     elif i == self.n_layers:
        #         means[i], state[i] = \
        #             self.sample_outer_cond(state[i - 1], visible=False)
        #     else:
        #         means[i], state[i] = \
        #             self.sample_inner_cond(i - 1, h_upper=state[i + 1],
        #                                    h_lower=state[i - 1])

        #     # reset clamped units
        #     if clamped is not None and clamped[i] is not None:
        #         state[i] = state[i].astype(float)
        #         state[i][clamped[i]] = clamped_val[i]
        return means

    # draw visible samples using Gibbs sampling
    def draw_samples(self, n_samples, init_v=None, binary=False,
                     clamped=None, clamped_val=None, layer_ind=0):
        # clamped are the indices of the clamped units:
        # clamped = list(layers, indices_in layer)
        # clamped_val = list(layers, values_in_layer)
        # list entry None for layers w/o clamped units
        # layer_ind: specifies from which layer samples are returned

        n_samples = int(n_samples)
        if layer_ind == 0:
            sample_size = self.n_visible
        else:
            sample_size = self.hidden_layers[layer_ind - 1]
        samples = np.empty((n_samples, sample_size))

        # initialize the chain
        if init_v is None:
            if binary:
                init_v = self.np_rng.randint(2, size=self.n_visible).astype(float)
            else:
                init_v = self.np_rng.rand(self.n_visible)
        curr_state = [init_v]
        for i, size in enumerate(self.hidden_layers):
            if binary:
                curr_state.append(self.np_rng.randint(2, size=size).astype(float))
            else:
                curr_state.append(self.np_rng.rand(size))

        # initialize clamped units correctly
        if clamped is not None:
            for i, i_clamped in enumerate(clamped):
                if i_clamped is not None:
                    curr_state[i][i_clamped] = clamped_val[i]

        # draw samples and save for one layer
        for t in range(n_samples):
            curr_means = self.gibbs_from_v(curr_state, clamped, clamped_val)
            if binary:
                samples[t, :] = curr_state[layer_ind].copy()
            else:
                samples[t, :] = curr_means[layer_ind].copy()
        return samples


# Sampling for CDBM works exactly as DBM, just with an additional label layer
# at the top
class CDBM(DBM):
    def __init__(self, layer_sizes, labels, vbias_init=None, numpy_seed=None):
        if len(layer_sizes) < 2:
            'Please use the RBM class for just one layer'
        # Set seed for reproducatbility
        self.np_rng = np.random.RandomState(numpy_seed)
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
            w = .01*self.np_rng.randn(input_size, self.hidden_layers[i])
            self.weights.append(w)

    # greedy layerwise training
    # all training hyperparameters from RBM.train can be used as kwargs
    def train(self, train_data, train_targets, **kwargs):
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

    def classify(self, v_data, probability=False):
        # draw samples with clamped visible units and argmax the label units
        clamped = [None] * (1 + self.n_layers)
        clamped[0] = np.arange(self.n_visible)
        clamped_val = [None] * (1 + self.n_layers)
        burn_in = 10

        # when passed multiple instances a loop is hard to avoid
        if len(v_data.shape) == 1:
            v_data = np.expand_dims(v_data, 0)
        if probability:
            labels = np.zeros((v_data.shape[0], self.hidden_layers[-1]))
        else:
            labels = np.zeros(v_data.shape[0])

        for i, v_input in enumerate(v_data):
            clamped_val[0] = v_input
            samples = self.draw_samples(burn_in+100, clamped=clamped,
                                        clamped_val=clamped_val,
                                        layer_ind=self.n_layers)
            if probability:
                labels[i] = np.average(samples, axis=0)
            else:
                labels[i] = np.argmax(np.sum(samples, axis=0))
        return labels

# Deep Boltzmann machine
from __future__ import division
from __future__ import print_function
import numpy as np
from rbm import RBM


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
        self.weights = []
        self.hbiases = []
        self.rbms = []
        if vbias_init is None:
            vbias_init = np.zeros(self.hidden_layers[0])
        self.vbias = vbias_init
        # initialize weights and biases
        for i in range(len(self.hidden_layers)):
            if i == 0:
                dbm_factor = [2, 1]
                input_size = self.n_visible
                input_bias = self.vbias
            else:
                dbm_factor = [1, 1]
                input_size = self.hidden_layers[i - 1]
                input_bias = self.hbiases[i - 1]
                if i == len(self.hidden_layers) - 1:
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
                                                   len(self.hidden_layers)))
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
        if len(h_upper.shape) == 1:
            h_upper = np.expand_dims(h_upper, 0)
            h_lower = np.expand_dims(h_lower, 0)
        if hasattr(beta, "__len__"):
            # for ast we get an array with 'n_instances' entries
            beta = np.expand_dims(beta, 1)

        act = beta * (h_upper.dot(self.weights[index + 1].T) +
                      h_lower.dot(self.weights[index]) + self.hbiases[index])
        p_on = 1./(1 + np.exp(-act))
        h_samples = (self.np_rng.rand(h_upper.shape[0],
                                      self.hidden_layers[index]) < p_on)*1
        return [p_on.squeeze(), h_samples.squeeze()]

    def sample_outer_cond(self, h_in, visible=True, beta=1.):
        if len(h_in.shape) == 1:
            h_in = np.expand_dims(h_in, 0)
        if hasattr(beta, "__len__"):
            # for ast we get an array with 'n_instances' entries
            beta = np.expand_dims(beta, 1)

        if visible:
            layer_size = self.n_visible
            bias = self.vbias
            weight = self.weights[0].T
        else:
            layer_size = self.hidden_layers[-1]
            bias = self.hbiases[-1]
            weight = self.weights[-1]

        act = beta * (h_in.dot(weight) + bias)
        p_on = 1./(1 + np.exp(-act))
        samples = (self.np_rng.rand(h_in.shape[0], layer_size) < p_on)*1
        return [p_on.squeeze(), samples.squeeze()]

    # make a gibbs step starting from the state of v and all other even
    # numbered layers. _state_ is given as a reference and changed inside the
    # method
    def gibbs_from_v(self, state, binary=False,
                     clamped=None, clamped_val=None):
        assert len(state) == len(self.hidden_layers) + 1
        means = [0] * len(state)
        samples = [0] * len(state)
        # # the loops can be parallelised -> add if necessary
        # # update odd numbered layers
        # for i in np.arange(1, len(self.hidden_layers) + 1, 2):
        #     # special sampling for top layer
        #     if i == len(self.hidden_layers):
        #         means, samples = \
        #             self.sample_outer_cond(state[i - 1], visible=False)
        #     else:
        #         means, samples = \
        #             self.sample_inner_cond(i - 1, h_upper=state[i + 1],
        #                                    h_lower=state[i - 1])
        #     if binary:
        #         state[i] = samples
        #     else:
        #         state[i] = means
        #     # reset clamped units
        #     if clamped is not None and clamped[i] is not None:
        #         state[i][clamped[i]] = clamped_val[i]

        # # update even numbered layers
        # for i in np.arange(0, len(self.hidden_layers) + 1, 2):
        #     if i == 0:
        #         means, samples = self.sample_outer_cond(state[1], visible=True)
        #     # special sampling for top layer
        #     elif i == len(self.hidden_layers):
        #         means, samples = \
        #             self.sample_outer_cond(state[i - 1], visible=False)
        #     else:
        #         means, samples = \
        #             self.sample_inner_cond(i - 1, h_upper=state[i + 1],
        #                                    h_lower=state[i - 1])
        #     if binary:
        #         state[i] = samples
        #     else:
        #         state[i] = means

        #     # reset clamped units
        #     if clamped is not None and clamped[i] is not None:
        #         state[i][clamped[i]] = clamped_val[i]
        # sequential version
        for i in np.arange(len(self.hidden_layers), -1, -1):
            # special sampling for top and bottom layer
            if i == 0:
                means, samples = \
                    self.sample_outer_cond(state[1], visible=True)
            elif i == len(self.hidden_layers):
                means, samples = \
                    self.sample_outer_cond(state[i - 1], visible=False)
            else:
                means, samples = \
                    self.sample_inner_cond(i - 1, h_upper=state[i + 1],
                                           h_lower=state[i - 1])
            if binary:
                state[i] = samples
            else:
                state[i] = means

            # reset clamped units
            if clamped is not None and clamped[i] is not None:
                state[i][clamped[i]] = clamped_val[i]

    # draw visible samples
    # AST is not implemented and only one chain possible
    def draw_samples(self, n_samples, init_v=None, binary=False,
                     clamped=None, clamped_val=None):
        # clamped are the indices of the clamped units:
        # clamped = list(layers, indices_in layer)
        # clamped_val = list(layers, values_in_layer)
        # list entry None for layers w/o clamped units

        n_samples = int(n_samples)
        samples = np.empty((n_samples, self.n_visible))

        # initialize the chain
        if init_v is None:
            if binary:
                init_v = self.np_rng.randint(2, size=self.n_visible)
            else:
                init_v = self.np_rng.rand(self.n_visible)
        curr_state = [init_v]
        for i, size in enumerate(self.hidden_layers):
            if binary:
                curr_state.append(self.np_rng.randint(2, size=size))
            else:
                curr_state.append(self.np_rng.rand(size))

        # draw samples for the visible layer
        for t in range(n_samples):
            self.gibbs_from_v(curr_state, binary, clamped, clamped_val)
            samples[t, :] = curr_state[0].copy()
        return samples

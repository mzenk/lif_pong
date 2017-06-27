# Deep Boltzmann machine
from __future__ import division
from __future__ import print_function
import numpy as np
from rbm import Rbm


class Dbn(object):
    def __init__(self, n_visible, hidden_layer_sizes, vbias_init=None,
                 numpy_seed=None):
        # Set seed for reproducatbility
        self.np_rng = np.random.RandomState(numpy_seed)
        # layers should be an list (nv, nh1, nh2, ..., nhl)
        self.n_layers = len(hidden_layer_sizes)
        self.n_visible = n_visible
        self.weights = []
        self.hbiases = []
        self.rbms = []
        self.trained = False
        # initialize weights and biases
        for i in range(self.n_layers):
            if i == 0:
                input_size = n_visible
                if vbias_init is None:
                    vbias_init = np.zeros(n_visible)

            else:
                input_size = hidden_layer_sizes[i-1]
                vbias_init = np.zeros(input_size)

            self.hbiases.append(np.zeros(hidden_layer_sizes[i]))
            w = .01*self.np_rng.randn(input_size, hidden_layer_sizes[i])
            self.weights.append(w)

            # The RBMs share the parameters with the DBN; Bengio lets each
            # RBMs keep its visible bias variable, but he calls it a
            # "philosophical" issue
            rbm = Rbm(input_size, hidden_layer_sizes[i],
                      w=self.weights[-1],
                      vbias=vbias_init,
                      hbias=self.hbiases[-1])
            self.rbms.append(rbm)

    def train(self, train_data, **kwargs):
        # all training hyperparameters from Rbm.train can be used as kwargs
        curr_input = train_data
        # greedy layerwise training
        for i, layer in enumerate(self.rbms):
            print('Training layer {} of {}'.format(i + 1, self.n_layers))
            layer.train(curr_input, **kwargs)
            # If samples are desired, use 2nd return value
            curr_input, _ = layer.sample_h_given_v(curr_input)
            kwargs['valid_set'] = None

        # add monitoring?

    def draw_samples(self, n_samples, clamped_ind=None, clamped_val=None):
        if clamped_ind is not None:
            self.draw_samples_while_clamped(n_samples, clamped_ind=None,
                                            clamped_val=None)
        # let the topmost RBM produce n_samples and propagate them down towards
        # the visible layer
        burn_in = 100
        samples = self.rbms[-1].draw_samples(burn_in + n_samples, binary=False)
        # keep only samples of the visible layer
        samples = samples[burn_in:, :self.rbms[-1].n_visible]
        # propagate through layers from top to bottom
        for layer in self.rbms[-2::-1]:
            samples, _ = layer.sample_v_given_h(samples)
        return samples

    def draw_samples_while_clamped(self, n_samples, clamped_ind=None,
                                   clamped_val=None):
        # In the original paper of Hinton, they do it with clamped labels like this:
        # clamped sampling of top layer ("associative memory"), then down-pass
        # the samples from equilibrium distribution
        # However, this technique is not applicable if I want to do pattern
        # completion with arbitrary pixels clamped... Think about/try/look in
        # literature for ways to achieve it
        pass

    # compute CSL for validation set
    # Are there other proxies for the loglikelihood?
    def compute_csl(self, valid_set):
        n_data = valid_set.shape[0]
        n_samples = 100
        top_layer = self.rbms[-1]
        # draw samples from last layer and propagate them towards visible layer
        burn_in = 100
        h_samples = top_layer.draw_samples(burn_in+n_samples, n_chains=n_data)
        h_samples = h_samples[:, burn_in:, -top_layer.n_hidden:
                              ].reshape((n_data * n_samples,
                                         top_layer.n_hidden))

        for layer in self.rbms[::-1]:
            # v_act = h_samples.dot(self.w.T) + self.vbias
            # model_pv = 1./(1 + np.exp(-v_act.reshape((n_samples, n_data,
            #                                          ->n_visible))))
            # I think above is equivalent to
            h_samples, _ = layer.sample_v_given_h(h_samples)
        model_pv = h_samples.reshape((n_samples, n_data, self.n_visible))

        # I use a linear interpolation between v=0 and v=1
        data_prob = np.prod((2*model_pv - 1)*valid_set + 1 - model_pv, axis=2)
        csl = np.log(np.average(data_prob, axis=1))

        return np.average(csl)

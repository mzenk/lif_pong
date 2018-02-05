from __future__ import division
import sbs
from sbs.gather_data import get_callbacks, eta_from_burnin
from sbs.logcfg import log
from lif_pong.utils import get_windowed_image_index
import time
import sys
import numpy as np
import pyNN.nest as sim

# from sbs.logcfg import set_loglevel
# set_loglevel(log, 3)
sbs.gather_data.set_subprocess_silent(True)


def sample_network(config_file, weights, biases, duration, dt=.1,
                   tso_params=None, burn_in_time=500., sim_setup_kwargs=None,
                   sampling_interval=10.):

    """
        How to setup and evaluate a Boltzmann machine. Please note that in
        order to instantiate BMs all needed neuron parameters need to be in the
        database and calibrated.

        Does the same thing as sbs.tools.sample_network(...).
    """
    network = initialise_network(config_file, weights, biases, tso_params)
    if sim_setup_kwargs is None:
        sim_setup_kwargs = {}
    if 'spike_precision' not in sim_setup_kwargs.keys():
        sim_setup_kwargs['spike_precision'] = 'on_grid'

    v_reset =  network.samplers[0].neuron_parameters.v_reset
    v_init = np.random.randint(2, size=len(biases))*(-v_reset) + v_reset


    network.gather_spikes(duration=duration, dt=dt, burn_in_time=burn_in_time,
                          sim_setup_kwargs=sim_setup_kwargs, initial_vmem=v_init)

    if len(network.biases_theo) < 7:
        log.info("DKL joint: {}".format(
            sbs.utils.dkl(network.dist_joint_theo.flatten(),
                          network.dist_joint_sim.flatten())))

    samples = network.get_sample_states(sampling_interval)

    return samples


def sample_network_clamped(
        config_file, weights, biases, duration, dt=.1, tso_params=None,
        burn_in_time=500., clamp_fct=None, sim_setup_kwargs=None,
        sampling_interval=10.):
    if sim_setup_kwargs is None:
        sim_setup_kwargs = {}
    if 'spike_precision' not in sim_setup_kwargs.keys():
        sim_setup_kwargs['spike_precision'] = 'on_grid'
    network = initialise_network(config_file, weights, biases, tso_params)

    network.spike_data = gather_network_spikes_clamped(
        network, duration, dt=dt, burn_in_time=burn_in_time,
        sim_setup_kwargs=sim_setup_kwargs, clamp_fct=clamp_fct)

    samples = network.get_sample_states(sampling_interval)

    return samples


def initialise_network(config_file, weights, biases, tso_params=None,
                       weight_scaling=1.):
    if weights is None or biases is None:
        print('Please provide weights and biases.')
        sys.exit()

    sampler_config = sbs.db.SamplerConfiguration.load(config_file)

    assert len(biases) == weights.shape[0] == weights.shape[1]
    bm = sbs.network.ThoroughBM(num_samplers=len(biases),
                                sampler_config=sampler_config)
    bm.weights_theo = weights
    bm.biases_theo = biases
    # NOTE: By setting the theoretical weights and biases, the biological
    # ones automatically get calculated on-demand by accessing
    # bm.weights_bio and bm.biases_bio

    if tso_params is None:
        bm.saturating_synapses_enabled = False
    else:
        bm.saturating_synapses_enabled = True
        # only adjust if not renewing
        log.warning('Check for renewing synapses assumes tau_syn == 10.')
        if not (tso_params['U'] == 1. and 
                tso_params['tau_fac'] == 0 and tso_params['tau_rec'] == 10.):
            bm.tso_params = tso_params
            # weight scaling could be swept over
            if tso_params['U'] != 0:
                bm.weights_bio *= weight_scaling / tso_params['U']
        bm.use_proper_tso = True
    return bm


def gather_network_spikes_clamped(
        network, duration, dt=0.1, burn_in_time=500., create_kwargs=None,
        sim_setup_kwargs=None, initial_vmem=None, clamp_fct=None,
        on_thresh=.5, off_thresh=None):
    """
        create_kwargs: Extra parameters for the networks creation routine.

        sim_setup_kwargs: Extra parameters for the setup command (random seeds
        etc.).

        clamp_fct: method handle that returns clamped indices and values;
        is called by the ClampCallback object, which then adjusts the biases
        accordingly
    """
    log.info("Gathering spike data...")
    if sim_setup_kwargs is None:
        sim_setup_kwargs = {}
    if 'spike_precision' not in sim_setup_kwargs.keys():
        sim_setup_kwargs['spike_precision'] = 'on_grid'

    if off_thresh is None:
        off_thresh = on_thresh
    elif off_thresh > on_thresh:
        log.error('Pixel threshold for off-state larger than for on-state.'
                  ' Aborting...')
        return None

    sim.setup(timestep=dt, **sim_setup_kwargs)

    if create_kwargs is None:
        create_kwargs = {}
    population, projections = network.create(duration=duration + burn_in_time,
                                             **create_kwargs)

    population.record("spikes")
    if initial_vmem is None:
        v_reset =  network.samplers[0].neuron_parameters.v_reset
        v_init = np.random.randint(2, size=len(network.samplers))*(-v_reset) + v_reset
    population.initialize(v=v_init)

    callbacks = get_callbacks(sim, {
            "duration": duration,
            "offset": burn_in_time,
            })

    t_start = time.time()
    if burn_in_time > 0.:
        log.info("Burning in samplers for {} ms".format(burn_in_time))
        sim.run(burn_in_time)
        eta_from_burnin(t_start, burn_in_time, duration)


    # add clamping functionality
    if clamp_fct is not None:
        callbacks.append(
            ClampCallback(network, clamp_fct, duration, offset=burn_in_time,
                          off_thresh=off_thresh, on_thresh=on_thresh))
    log.info("Starting data gathering run.")
    sim.run(duration, callbacks=callbacks)

    if isinstance(population, sim.common.BasePopulation):
        spiketrains = population.get_data("spikes").segments[0].spiketrains
    else:
        spiketrains = np.vstack(
                [pop.get_data("spikes").segments[0].spiketrains[0]
                 for pop in population])

    # we need to ignore the burn in time
    clean_spiketrains = []
    for st in spiketrains:
        clean_spiketrains.append(np.array(st[st > burn_in_time])-burn_in_time)

    return_data = {
            "spiketrains": clean_spiketrains,
            "duration": duration,
            "dt": dt,
            }
    sim.end()

    return return_data


def gather_network_spikes_clamped_bn(
        network, duration, nv, dt=0.1, burn_in_time=500., create_kwargs=None,
        sim_setup_kwargs=None, initial_vmem=None, clamp_fct=None,
        clamp_tso_params=None, wp_fit_params=None):
    """
        create_kwargs: Extra parameters for the networks creation routine.

        sim_setup_kwargs: Extra parameters for the setup command (random seeds
        etc.).

        clamp_fct: method handle that returns clamped indices and values;
        is called by the ClampCallback object, which then adjusts the biases
        accordingly
    """
    log.info("Gathering spike data...")
    if sim_setup_kwargs is None:
        sim_setup_kwargs = {}
    if 'spike_precision' not in sim_setup_kwargs.keys():
        sim_setup_kwargs['spike_precision'] = 'on_grid'

    sim.setup(timestep=dt, **sim_setup_kwargs)

    if create_kwargs is None:
        create_kwargs = {}
    population, projections = network.create(duration=duration + burn_in_time,
                                             **create_kwargs)
    # set up second population and connect
    spike_interval = 1.   # ms
    tau_syn = population.get('tau_syn_E')
    if clamp_tso_params['U'] == 1 and clamp_tso_params['tau_rec'] == tau_syn:
        accum_correction = 1.
    else:
        accum_correction = 1 - np.exp(-spike_interval/tau_syn)

    cont_firing_train = burn_in_time + np.arange(dt, duration, spike_interval)
    spikesource = sim.SpikeSourceArray(spike_times=cont_firing_train)
    # bias_neurons = sim.Population(nv, spikesource)
    # bn_connector = sim.OneToOneConnector()
    bias_neurons = sim.Population(1, spikesource)
    bn_connector = sim.AllToAllConnector()
    if clamp_tso_params is None:
        bn_synapse = sim.StaticSynapse(weight=0.)
    else:
        sim.nest.CopyModel("tsodyks2_synapse", "avoid_pynn_trying_to_be_smart")
        bn_synapse = sim.native_synapse_type("avoid_pynn_trying_to_be_smart")(
            **clamp_tso_params)

    # make excitatory AND inhibitory connections
    exc_bn_proj = sim.Projection(bias_neurons, population[:nv],
                                 connector=bn_connector,
                                 receptor_type='excitatory',
                                 synapse_type=bn_synapse)

    inh_bn_proj = sim.Projection(bias_neurons, population[:nv],
                                 connector=bn_connector,
                                 receptor_type='inhibitory',
                                 synapse_type=bn_synapse)

    bn_projections = {'exc': exc_bn_proj, 'inh': inh_bn_proj}

    population.record("spikes")
    if initial_vmem is None:
        v_reset =  network.samplers[0].neuron_parameters.v_reset
        v_init = np.random.randint(2, size=len(network.samplers))*(-v_reset) + v_reset
    population.initialize(v=v_init)

    callbacks = get_callbacks(sim, {
            "duration": duration,
            "offset": burn_in_time,
            })

    t_start = time.time()
    if burn_in_time > 0.:
        log.info("Burning in samplers for {} ms".format(burn_in_time))
        sim.run(burn_in_time)
        eta_from_burnin(t_start, burn_in_time, duration)

    # add clamping functionality
    if clamp_fct is not None:
        callbacks.append(
            ClampCallbackBN(bn_projections, network.biases_theo[:nv],
                            clamp_fct, duration, offset=burn_in_time,
                            acc_corr=accum_correction,
                            wp_fit_params=wp_fit_params))

    log.info("Starting data gathering run.")
    sim.run(duration, callbacks=callbacks)

    if isinstance(population, sim.common.BasePopulation):
        spiketrains = population.get_data("spikes").segments[0].spiketrains
    else:
        spiketrains = np.vstack(
                [pop.get_data("spikes").segments[0].spiketrains[0]
                 for pop in population])

    # we need to ignore the burn in time
    clean_spiketrains = []
    for st in spiketrains:
        clean_spiketrains.append(np.array(st[st > burn_in_time])-burn_in_time)

    return_data = {
            "spiketrains": clean_spiketrains,
            "duration": duration,
            "dt": dt,
            }
    sim.end()

    return return_data


# Clamping implemented with bias neurons that fire at 1kHz whenever unit "on"
def gather_network_spikes_clamped_sf(
        network, duration, nv, dt=0.1, burn_in_time=500., create_kwargs=None,
        sim_setup_kwargs=None, initial_vmem=None, clamp_fct=None,
        clamp_tso_params=None, wp_fit_params=None, on_thresh=.5,
        off_thresh=None):
    """
        create_kwargs: Extra parameters for the networks creation routine.

        sim_setup_kwargs: Extra parameters for the setup command (random seeds
        etc.).

        clamp_fct: method handle that returns clamped indices and values;
        is called by the ClampCallback object, which then adjusts the biases
        accordingly
    """
    log.info("Gathering spike data...")
    if sim_setup_kwargs is None:
        sim_setup_kwargs = {}
    if 'spike_precision' not in sim_setup_kwargs.keys():
        sim_setup_kwargs['spike_precision'] = 'on_grid'

    sim.setup(timestep=dt, **sim_setup_kwargs)

    if create_kwargs is None:
        create_kwargs = {}
    population, projections = network.create(duration=duration + burn_in_time,
                                             **create_kwargs)

    start = time.time()
    # Create spike sources for whole simulation
    spike_interval = 1.   # ms
    exc_spiketrains, inh_spiketrains = clampfct_to_spiketrain(
        clamp_fct, nv, duration + burn_in_time, dt, spike_interval,
        offset=burn_in_time, on_thresh=on_thresh)

    exc_bias_neurons = sim.Population(
        nv, sim.SpikeSourceArray, cellparams={'spike_times': exc_spiketrains})
    inh_bias_neurons = sim.Population(
        nv, sim.SpikeSourceArray, cellparams={'spike_times': inh_spiketrains})

    # calculate weights according to the calibration from wp_fit_params
    if wp_fit_params is not None:
        w_on = wp_fit_params['wp05'] + 2*4*wp_fit_params['alpha']
        w_off = wp_fit_params['wp05'] - 2*4*wp_fit_params['alpha']
        bias_factor = wp_fit_params['bias_factor']
        exc_weights = w_on + bias_factor * network.biases_theo[:nv]
        inh_weights = w_off + bias_factor * network.biases_theo[:nv]
        # choose symmetric weights so that decay takes equally long. Necessary?
        weights = np.maximum(np.abs(exc_weights), np.abs(inh_weights))
    elif clamp_tso_params is not None and 'weight' in clamp_tso_params.keys():
        weights = clamp_tso_params['weight'] * np.ones(nv)
    else:
        weights = np.zeros(nv)
        log.warning('No weights provided for bias neuron synapses.')

    # apply corrections (nest units, compensate U, compensate accumulation)
    tau_syn = population.get('tau_syn_E')
    if clamp_tso_params is not None and clamp_tso_params['U'] != 0:
        weights *= 1000. / clamp_tso_params['U']
        if clamp_tso_params['U'] != 1 \
                and clamp_tso_params['tau_rec'] != tau_syn:
            weights *= 1 - np.exp(-spike_interval/tau_syn)

    # Create connections
    exc_connections = []
    inh_connections = []
    for i in range(nv):
        exc_connections.append((i, i, weights[i]))
        inh_connections.append((i, i, -np.abs(weights[i])))

    bn_connector_exc = sim.FromListConnector(
        exc_connections, column_names=['weight'])
    # in contrast to COBA, CUBA needs negative inhibitory weights
    if '_curr_' in str(population.celltype):
        log.info('Using negative inhibitory weights for CUBA.')
        bn_connector_inh = sim.FromListConnector(
            inh_connections, column_names=['weight'])
    else:
        bn_connector_inh = bn_connector_exc

    if clamp_tso_params is None:
        bn_synapse = sim.StaticSynapse(weight=0.)
    else:
        sim.nest.CopyModel("tsodyks2_synapse", "avoid_pynn_trying_to_be_smart")
        bn_synapse = sim.native_synapse_type("avoid_pynn_trying_to_be_smart")(
            **clamp_tso_params)

    bn_projections = {}
    # make excitatory AND inhibitory connections
    bn_projections['exc'] = sim.Projection(exc_bias_neurons, population[:nv],
                                           connector=bn_connector_exc,
                                           receptor_type='excitatory',
                                           synapse_type=bn_synapse)

    bn_projections['inh'] = sim.Projection(inh_bias_neurons, population[:nv],
                                           connector=bn_connector_inh,
                                           receptor_type='inhibitory',
                                           synapse_type=bn_synapse)

    log.info('Initial overhead from clamping: {}s'.format(time.time() - start))

    # this is a slow (no idea why) alternative to the connection above
    # bn_connector = sim.OneToOneConnector()
    # make projections...
    # # apply weights to BN-synapses
    # for rt in ['exc', 'inh']:
    #     weight_matrix = bn_projections[rt].get('weight', format='array')
    #     weight_matrix[np.diag_indices(len(weight_matrix))] = weights

    #     start = time.time()
    #     bn_projections[rt].set(weight=weight_matrix)
    #     log.info('Setting weights took {}s'.format(time.time() - start))

    population.record("spikes")
    if initial_vmem is None:
        v_reset =  network.samplers[0].neuron_parameters.v_reset
        v_init = np.random.randint(2, size=len(network.samplers))*(-v_reset) + v_reset
    population.initialize(v=v_init)

    callbacks = get_callbacks(sim, {
            "duration": duration,
            "offset": burn_in_time,
            })

    t_start = time.time()
    if burn_in_time > 0.:
        log.info("Burning in samplers for {} ms".format(burn_in_time))
        sim.run(burn_in_time)
        eta_from_burnin(t_start, burn_in_time, duration)

    log.info("Starting data gathering run.")
    sim.run(duration, callbacks=callbacks)

    if isinstance(population, sim.common.BasePopulation):
        spiketrains = population.get_data("spikes").segments[0].spiketrains
    else:
        spiketrains = np.vstack(
                [pop.get_data("spikes").segments[0].spiketrains[0]
                 for pop in population])

    # we need to ignore the burn in time
    clean_spiketrains = []
    for st in spiketrains:
        clean_spiketrains.append(np.array(st[st > burn_in_time])-burn_in_time)

    return_data = {
            "spiketrains": clean_spiketrains,
            "duration": duration,
            "dt": dt,
            }
    sim.end()

    return return_data


def clampfct_to_spiketrain(clamp_fct, n_neurons, duration, dt, spike_interval,
                           offset=0, off_thresh=None, on_thresh=.5):
    if off_thresh is not None:
        raise NotImplementedError
    # clamp_fct has time as argument and returns time to next call,
    # index and value of clamped neurons
    t = offset
    exc_spiketimes_array = []
    inh_spiketimes_array = []
    for i in range(n_neurons):
        exc_spiketimes_array.append([])
        inh_spiketimes_array.append([])
    while t < duration:
        delta_t, curr_idx, curr_val = clamp_fct(t - offset)
        log.debug('t={}, #clamped={}'.format(t, len(curr_idx)))
        # can try smooth clamping (var. spike_interval) or different threshold
        t_next = min(t + delta_t, duration)
        st_firing = np.arange(t + dt, t_next, spike_interval)
        for i, x in zip(curr_idx, curr_val):
            if x >= on_thresh:
                exc_spiketimes_array[i] += st_firing.tolist()
            else:
                inh_spiketimes_array[i] += st_firing.tolist()
        t = t_next
    return exc_spiketimes_array, inh_spiketimes_array


def inv_sigma(p):
    return np.log(p / (1 - p))


class ClampCallback(object):

    def __init__(self, network, clamp_fct, duration,
                 offset=0., off_thresh=.5, on_thresh=.5):
        self.network = network
        self.initial_biases = network.biases_theo.copy()
        self.duration = duration
        self.offset = offset
        self.clamp_fct = clamp_fct
        self.clamped_idx = []
        self.clamped_val = []
        self.on_bias = 100.
        assert on_thresh >= off_thresh
        self.on_thresh = on_thresh
        self.off_thresh = off_thresh

    def __call__(self, t):
        if np.isclose(t, self.duration + self.offset):
            # clean up for later experiments
            log.info('Unclamp all neurons...')
            self.network.biases_theo = self.initial_biases
            return float('inf')

        dt, curr_idx, curr_val = self.clamp_fct(t - self.offset)
        log.debug(len(curr_idx))
        tmp_bias = self.network.biases_theo.copy()
        # leave unchanged units clamped for efficiency -> is it worth it?
        released = np.setdiff1d(self.clamped_idx, curr_idx).astype(int)
        tmp_bias[released] = self.initial_biases[released]
        if len(curr_idx) != 0:
            tmp_bias[curr_idx] = self.soft_biases(curr_val, self.off_thresh,
                                                  self.on_thresh)
            # # if I don't want to add negative biases to the initial bias (cf. notes)
            # tmp_bias[curr_idx] = np.minimum(self.initial_biases[curr_idx],
            #                                 tmp_bias[curr_idx])
        self.network.biases_theo = tmp_bias
        self.clamped_idx = curr_idx
        self.clamped_val = curr_val

        return min(t + dt, self.duration + self.offset)

    def binary_biases(self, clamped_val, thresh=.5):
        # binarize clamped value
        binary_val = 1.*(clamped_val > thresh)
        return 2 * (binary_val - .5) * self.on_bias

    def soft_biases(self, clamped_val, off_thresh=.2, on_thresh=.8):
        hard_on = clamped_val >= on_thresh
        hard_off = clamped_val <= off_thresh
        soft = np.logical_not(np.logical_or(hard_on, hard_off))
        biases = np.ones_like(clamped_val)
        biases[hard_off] = -self.on_bias
        biases[hard_on] = self.on_bias
        biases[soft] = inv_sigma(clamped_val[soft])
        return biases


class ClampCallbackBN(object):
    def __init__(self, bn_projections, bv_theo, clamp_fct, duration,
                 acc_corr=1., offset=0., wp_fit_params=None):
        self.bn_projections = bn_projections
        self.bv_theo = bv_theo.copy()
        self.duration = duration
        self.offset = offset
        self.clamp_fct = clamp_fct
        # correction factor for spike accumulation must be supplied
        self.acc_corr = acc_corr
        if wp_fit_params is None:
            log.warning('No fit parameters for w-p_on suppied')
            wp_fit_params = {'wp05': 0., 'alpha': 1., 'bias_factor': 0.}
        assert 'wp05' in wp_fit_params.keys() \
            and 'alpha' in wp_fit_params.keys() \
            and 'bias_factor' in wp_fit_params.keys(), \
            log.error('Wrong format of fit parameters.')

        self.w_on = wp_fit_params['wp05'] + 2*4*wp_fit_params['alpha']
        self.bias_factor = wp_fit_params['bias_factor']

    def __call__(self, t):
        if np.isclose(t, self.duration + self.offset):
            # clean up for later experiments
            log.info('Unclamp all neurons...')
            for rt in ['exc', 'inh']:
                self.bn_projections[rt].set(weight=0)
            return float('inf')

        dt, curr_idx, curr_val = self.clamp_fct(t - self.offset)

        # caluclate weights
        # !!! this might have to be corrected as in gather..._sf
        weights = np.zeros_like(self.bv_theo)
        weights[curr_idx] = (2.*(curr_val > .5) - 1) * \
            (self.w_on + self.bias_factor * self.bv_theo[curr_idx])

        # apply corrections
        if self.bn_projections['exc'].synapse_type.nest_name == \
                'avoid_pynn_trying_to_be_smart':
            # weight has to be increased so that U<1 is compensated
            u_tso = self.bn_projections['exc'].get('U', format='array')
            # using native nest model with different weight units
            weights *= 1000. / np.diag(u_tso) * self.acc_corr

        # apply weights to BN-synapses
        for rt in ['exc', 'inh']:
            weight_matrix = \
                self.bn_projections[rt].get('weight', format='array')
            # take only pos/neg weights for exc/inh
            sign_w = -1 if rt == 'inh' else 1
            # weight_matrix[np.diag_indices(len(weight_matrix))] = \
            #     np.clip(sign_w * weights, 0, np.inf)
            weight_matrix = \
                np.expand_dims(np.clip(sign_w * weights, 0, np.inf), 0)

            start = time.time()
            self.bn_projections[rt].set(weight=weight_matrix)
            log.info('Setting weights took {}s'.format(time.time() - start))

        return min(t + dt, self.duration + self.offset)


# Custom clamping methods -> functors that are called from ClampCallback
class Clamp_anything(object):
    def __init__(self, refresh_times, clamped_idx, clamped_val):
        # please pass lists (special case below)
        if len(refresh_times) == 1 and len(refresh_times) != len(clamped_idx):
            # it is possible to pass a one-element list refresh-times and
            # 1d-arrays for idx and val to the method; for compability with
            # __call__ the latter two have to be expanded then
            self.clamped_idx = [clamped_idx]
            self.clamped_val = [clamped_val]
        else:
            assert len(refresh_times) == len(clamped_idx) == len(clamped_val)
            self.clamped_idx = clamped_idx
            self.clamped_val = clamped_val

        # method is always called with t==0, so we need to add this if necess.
        if refresh_times[0] != 0:
            refresh_times.insert(0, 0.)
            self.clamped_idx.insert(0, [])
            self.clamped_val.insert(0, [])
        self.refresh_times = refresh_times

    def set_clamped_val(self, clamped_val):
        if type(clamped_val) is not list and \
                len(clamped_val.shape) != len(self.clamped_val.shape):
            if len(clamped_val.shape) == 1 and self.clamped_val.shape[0] == 1:
                self.clamped_val = np.expand_dims(clamped_val, 0)
            else:
                print('clamped_val could not be set because (shape mismatch)')
                print('Expected: ' + str(self.clamped_val))
        else:
            assert len(clamped_val) == len(self.clamped_val)
            self.clamped_val = clamped_val

    def __call__(self, t):
        try:
            i = np.where(np.isclose(self.refresh_times, t))[0][0]
        except IndexError:
            log.warning('No matching clamping time stamp: t={}, '
                        'clamp_times={}'.format(t, self.refresh_times))
            return float('inf'), [], []

        if i < len(self.refresh_times) - 1:
            dt = self.refresh_times[i + 1] - t
        else:
            dt = float('inf')
        return dt, self.clamped_idx[i], self.clamped_val[i]


class Clamp_window(object):
    def __init__(self, interval, clamp_img, win_size=None):
        self.interval = interval
        self.clamp_img = clamp_img
        if win_size is None:
            win_size = clamp_img.shape[1]
        self.win_size = win_size

    def __call__(self, t):
        end = min(int(t / self.interval), self.clamp_img.shape[1])
        clamped_idx = get_windowed_image_index(
            self.clamp_img.shape, end, self.win_size)
        clamped_val = self.clamp_img.flatten()[clamped_idx]

        return self.interval, clamped_idx, clamped_val

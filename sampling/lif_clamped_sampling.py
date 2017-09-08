import sbs
from sbs.gather_data import get_callbacks, eta_from_burnin
from sbs.logcfg import log
from utils import get_windowed_image_index
import time
import sys
import numpy as np
import pyNN.nest as sim


sbs.gather_data.set_subprocess_silent(True)


def sample_network(network, duration, dt=.1, burn_in=500., tso_params=None,
                   seed=42, sampling_interval=10.):

    """
        How to setup and evaluate a Boltzmann machine. Please note that in
        order to instantiate BMs all needed neuron parameters need to be in the
        database and calibrated.

        Does the same thing as sbs.tools.sample_network(...).
    """
    # np.random.seed(seed)
    sim_setup_kwargs = {
        'rng_seeds_seed': seed
    }

    if tso_params is None:
        network.saturating_synapses_enabled = False
    else:
        network.saturating_synapses_enabled = True
        if tso_params != 'default':
            network.tso_params = tso_params
            network.weights_bio /= tso_params['U']
        network.use_proper_tso = True

    network.gather_spikes(duration=duration, dt=dt, burn_in_time=burn_in,
                          sim_setup_kwargs=sim_setup_kwargs)
    # network.save(save)

    if len(network.biases_theo) < 7:
        log.info("DKL joint: {}".format(
            sbs.utils.dkl(network.dist_joint_theo.flatten(),
                          network.dist_joint_sim.flatten())))

    samples = network.get_sample_states(sampling_interval)

    return samples


def setup_simulation(dt=.1, sim_setup_kwargs=None):
    if sim_setup_kwargs is None:
        sim_setup_kwargs = {}

    sim.setup(timestep=dt, **sim_setup_kwargs)


def end_simulation():
    sim.end()


def initialise_network(config_file, weights, biases, load=None):
    if load is not None:
        bm = sbs.network.ThoroughBM.load(load)
    else:
        # No network loaded, we need to create it. We need to specify how many
        # samplers we want and what neuron parameters they should have. Refer
        # to the documentation for all the different ways this is possible.
        if weights is None or biases is None:
            print('Please provide weights and biases.')
            return

        sampler_config = sbs.db.SamplerConfiguration.load(config_file)

        assert len(biases) == weights.shape[0] == weights.shape[1]
        bm = sbs.network.ThoroughBM(num_samplers=len(biases),
                                    sampler_config=sampler_config)
        bm.weights_theo = weights
        bm.biases_theo = biases
        # NOTE: By setting the theoretical weights and biases, the biological
        # ones automatically get calculated on-demand by accessing
        # bm.weights_bio and bm.biases_bio
    return bm


def make_network_connections(network, duration, burn_in_time=100.,
                             tso_params=None, create_kwargs=None):
    if create_kwargs is None:
        create_kwargs = {}
    if tso_params is None:
        network.saturating_synapses_enabled = False
    else:
        network.saturating_synapses_enabled = True
        if tso_params != 'default':
            network.tso_params = tso_params
            network.weights_bio /= tso_params['U']
        network.use_proper_tso = True
    network.create(duration=duration + burn_in_time, **create_kwargs)


def simulate_network(network, duration, dt=0.1, burn_in_time=100.,
                     initial_vmem=None, clamp_fct=None):
    # network must have been created and simulation setup
    try:
        sim.reset()
    except:
        print('Resetting simulation failed. Maybe it was not setup correctly?')
        print('Simulation aborted.')
        sim.end()
        sys.exit()

    population = network.population
    if initial_vmem is not None:
        population.initialize(v=initial_vmem)

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
    callbacks.append(ClampCallback(network, clamp_fct, offset=burn_in_time))
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

    return return_data


def gather_network_spikes_clamped(
        network, duration, dt=0.1, burn_in_time=0., create_kwargs=None,
        sim_setup_kwargs=None, initial_vmem=None, clamp_fct=None):
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

    sim.setup(timestep=dt, **sim_setup_kwargs)

    if create_kwargs is None:
        create_kwargs = {}
    population, projections = network.create(duration=duration + burn_in_time,
                                             **create_kwargs)

    population.record("spikes")
    if initial_vmem is not None:
        population.initialize(v=initial_vmem)

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
    callbacks.append(ClampCallback(network, clamp_fct, offset=burn_in_time))
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


# Custom clamping methods -> functors that are called from ClampCallback
class Clamp_anything(object):
    # refresh times must be a list
    def __init__(self, refresh_times, clamped_idx, clamped_val):
        if len(refresh_times) == 1 and len(refresh_times) != len(clamped_idx):
            self.clamped_idx = np.expand_dims(clamped_idx, 0)
            self.clamped_val = np.expand_dims(clamped_val, 0)
        else:
            self.clamped_idx = clamped_idx
            self.clamped_val = clamped_val
        self.refresh_times = refresh_times

    def set_clamped_val(self, clamped_val):
        if len(clamped_val.shape) != len(self.clamped_val.shape):
            if len(clamped_val.shape) == 1 and self.clamped_val.shape[0] == 1:
                self.clamped_val = np.expand_dims(clamped_val, 0)
            else:
                print('clamped_val could not be set because (shape mismatch)')
                print('Expected: ' + str(self.clamped_val))
        else:
            self.clamped_val = clamped_val

    def __call__(self, t):
        try:
            i = np.where(np.isclose(self.refresh_times, t))[0][0]
        except IndexError:
            print('No matching clamping time stamp; this should not happen.')
            return float('inf'), [], []

        binary_val = np.round(self.clamped_val[i])
        if i < len(self.refresh_times) - 1:
            dt = self.refresh_times[i + 1] - t
        else:
            dt = float('inf')
        return dt, self.clamped_idx[i], binary_val.astype(float)


class ClampCallback(object):

    def __init__(self, network, clamp_fct, offset=0.):
        self.network = network
        self.initial_biases = network.biases_theo
        self.offset = offset
        self.clamp_fct = clamp_fct
        self.clamped_idx = []
        self.clamped_val = []

    def __call__(self, t):
        # pdb.set_trace()
        dt, curr_idx, curr_val = self.clamp_fct(t - self.offset)
        assert np.all(np.in1d(curr_val, [0, 1]))
        # assuming binary clamped values

        tmp_bias = self.network.biases_theo.copy()
        # leave unchanged units clamped for efficiency -> is it worth it?
        released = np.setdiff1d(self.clamped_idx, curr_idx).astype(int)
        tmp_bias[released] = self.initial_biases[released]
        if len(curr_idx) != 0:
            tmp_bias[curr_idx] = 2 * (curr_val - .5) * 100.
        self.network.biases_theo = tmp_bias
        self.clamped_idx = curr_idx
        self.clamped_val = curr_val
        return t + dt


class Clamp_window(object):
    def __init__(self, interval, clamp_img, win_size):
        self.interval = interval
        self.clamp_img = clamp_img
        self.win_size = win_size

    def __call__(self, t):
        end = min(int(t / self.interval), self.clamp_img.shape[1])
        clamped_idx = get_windowed_image_index(
            self.clamp_img.shape, end, self.win_size)
        # binarized version
        clamped_val = np.round(self.clamp_img.flatten()[clamped_idx])

        return self.interval, clamped_idx, clamped_val

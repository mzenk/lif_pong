import sbs
from sbs.gather_data import get_callbacks, eta_from_burnin
from sbs.logcfg import log
import time
import numpy as np
import pyNN.nest as sim
# import pdb


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

    exec "import {} as sim".format(network.sim_name) in globals(), locals()
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

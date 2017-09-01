import sbs
from sbs.gather_data import get_callbacks, eta_from_burnin
from sbs.logcfg import log
from copy import deepcopy
import time
import numpy as np
import pyNN.nest as sim
import cPickle
import sys
sys.path.insert(0, '../gibbs_rbm')
from rbm import RBM, CRBM
#example of custom(standalone) gather_spikes function for sbs

#how to use (this is just a partitioned and modified version of gather_spikes):
#first use initiate_setup to start pynn.setup
#then use connect_neurons_of_BMs to create neurons, sources and connections
#then run the network, return: spike-data
#if network is your ThroroughBM network, you use network.spike_data = run_network(network, ...), afterwards you can use the get_states fct as usual

def initiate_setup(network, dt = 0.1, sim_setup_kwargs=None):
#network is the ThoroughBM object
    
    if sim_setup_kwargs is None:
            sim_setup_kwargs = {}

    #exec "import {} as sim".format(network.sim_name) in globals(), locals()
    sim.setup(timestep=dt, **sim_setup_kwargs)

def connect_neurons_of_BMs(network, BM_weights, duration, burn_in_time = 500.,
                           saturating_syn=True, create_kwargs=None):
    #exec "import {} as sim".format(network.sim_name) in globals(), locals()
    
    if create_kwargs is None:
        create_kwargs = {}

    network.saturating_synapses_enabled = saturating_syn
    network.use_proper_tso = True
    network.weights_theo = BM_weights

    network.population, network.projections = network.create(duration=duration+burn_in_time, **create_kwargs)
    log.info("Sampling BMs created.")


def run_network(network, test_images, clamp_duration = 4e2, dt=0.1, burn_in_time=500., initial_vmem=None):

    exec "import {} as sim".format(network.sim_name) in globals(), locals() 
    population = network.population

    population.record("spikes")
    if initial_vmem is not None:
        population.initialize(v=initial_vmem)

    callbacks = get_callbacks(sim, {
            "duration" : clamp_duration*len(test_images)*repeats,
            "offset" : burn_in_time})

    t_start = time.time()
    if burn_in_time > 0.:
        log.info("Burning in samplers for {} ms".format(burn_in_time))
        sim.run(burn_in_time)
        eta_from_burnin(t_start, burn_in_time, clamp_duration*len(test_images)*repeats)

    log.info("Starting data gathering run.")

#clamp each image in test_images for clamp_duration ms
    visibles_to_clamp = 28*28#794
    test_images = test_images[:, :visibles_to_clamp]
    clamped_biases = (np.round(test_images) - 0.5)*2.*50.
    for j in range(len(clamped_biases)):
        old_bias = network.biases_theo
        old_bias[:visibles_to_clamp] = clamped_biases[j]
        network.biases_theo = old_bias

        sim.run(clamp_duration, callbacks=callbacks)
#

    if isinstance(population, sim.Population):
        spiketrains = population.get_data("spikes").segments[0].spiketrains
    else:
        spiketrains = np.vstack([pop.get_data("spikes").segments[0].spiketrains[0] for pop in population])

        # we need to ignore the burn in time
    clean_spiketrains = []
    burnin_spiketrains = []
    for st in spiketrains:
        clean_spiketrains.append(np.array(st[st > burn_in_time])-burn_in_time)
        burnin_spiketrains.append(np.array(st[st <= burn_in_time]))

    return_data = {"spiketrains" : clean_spiketrains,
            "duration" : clamp_duration*len(test_images)*repeats,
            "dt" : dt}

    return return_data

sim_name = "pyNN.nest"
with open('../gibbs_rbm/saved_rbms/mnist_disc_rbm.pkl', 'rb') as f:
    rbm = cPickle.load(f)
w_rbm = rbm.w
b = np.concatenate((rbm.vbias, rbm.hbias))
nv, nh = rbm.n_visible, rbm.n_hidden

# Bring weights and biases into right form
w = np.concatenate((np.concatenate((np.zeros((nv, nv)), w_rbm), axis=1),
                   np.concatenate((w_rbm.T, np.zeros((nh, nh))), axis=1)),
                   axis=0)

sampler_config = sbs.db.SamplerConfiguration.load('dodo_calib.json')
bm = sbs.network.ThoroughBM(num_samplers=len(b),
                            sim_name=sim_name,
                            sampler_config=sampler_config)
bm.biases_theo = b

# NOTE: By setting the theoretical weights and biases, the biological
# ones automatically get calculated on-demand by accessing
# bm.weights_bio and bm.biases_bio
repeats = 1
test_images = np.ones((1, 28**2))
clamp_duration = 2e3
dt = .1
initiate_setup(bm, dt=dt)
connect_neurons_of_BMs(bm, w, clamp_duration, saturating_syn=False)
bm.spike_data = run_network(bm, test_images, clamp_duration=clamp_duration, dt=dt)
np.save('dod_1_false.npy', bm.get_sample_states())

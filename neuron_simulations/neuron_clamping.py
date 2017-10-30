# testing clamping mechanisms
import numpy
import pyNN.nest as sim
import numpy as np
import cPickle
from utils.data_mgmt import make_figure_folder, make_data_folder, get_data_path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sbs.cutils import generate_states


def clamping_expt(n_neurons, duration, neuron_params, noise_params,
                  synweight=.1, tso_params=None, dt=.1, savename='test'):
    sim.setup(timestep=dt, spike_precision="on_grid", quit_on_end=False)

    spike_times = numpy.arange(dt, duration, 10.)
    spike_source = sim.Population(
        1, sim.SpikeSourceArray(spike_times=spike_times))
    exc_source = sim.Population(
        n_neurons, sim.SpikeSourcePoisson(rate=noise_params['rate_exc']))
    inh_source = sim.Population(
        n_neurons, sim.SpikeSourcePoisson(rate=noise_params['rate_inh']))

    if tso_params is None:
        bn_synapse = sim.StaticSynapse()
        weight_factor = 1.
    else:
        sim.nest.CopyModel("tsodyks2_synapse", "avoid_pynn_trying_to_be_smart")
        bn_synapse = sim.native_synapse_type("avoid_pynn_trying_to_be_smart")(
            **tso_params)
        weight_factor = 1000. / tso_params['U']

    # setup
    coba_lif = sim.IF_cond_exp(**neuron_params)
    population = sim.Population(n_neurons, coba_lif)

    population.record(['v', 'spikes'])
    projections = {}
    projections['exc'] = sim.Projection(exc_source, population,
                                        connector=sim.OneToOneConnector(),
                                        receptor_type='excitatory',
                                        synapse_type=sim.StaticSynapse())
    projections['exc'].set(weight=noise_params['w_exc'])
    projections['inh'] = sim.Projection(inh_source, population,
                                        connector=sim.OneToOneConnector(),
                                        receptor_type='inhibitory',
                                        synapse_type=sim.StaticSynapse())
    projections['inh'].set(weight=-noise_params['w_inh'])
    projections['input'] = sim.Projection(spike_source, population,
                                          connector=sim.AllToAllConnector(),
                                          receptor_type='excitatory',
                                          synapse_type=bn_synapse)
    projections['input'].set(weight=synweight * weight_factor)

    sim.run(duration)

    # === Save Data ===========================================================
    vmems = population.get_data().segments[0].filter(name='v')[0].T
    spiketrains = population.get_data().segments[0].spiketrains
    spike_list = []
    for st in spiketrains:
        spike_id = st.annotations['source_index']
        id_t = np.vstack((np.repeat(spike_id, len(st.magnitude)), st.magnitude))
        spike_list.append(id_t)

    spike_data = np.hstack(spike_list)
    ordered_spike_data = spike_data[:, np.argsort(spike_data[1])]
    samples = generate_states(
        spike_ids=ordered_spike_data[0].astype(int),
        spike_times=np.array(ordered_spike_data[1]/dt, dtype=int),
        tau_refrac_pss=np.array([neuron_params['tau_refrac']/dt] * n_neurons,
                                dtype=int),
        num_samplers=n_neurons,
        steps_per_sample=int(neuron_params['tau_refrac']/dt),
        duration=np.array(duration/dt, dtype=int)
        )

    with open(make_data_folder() + savename + '.pkl', 'w') as f:
        cPickle.dump({'vmems': vmems, 'samples': samples}, f)

    # === Clean up and quit =======================================================
    sim.end()


if __name__ == '__main__':
    mpl.rcParams['font.size'] = 14
    # lif-sampling framework --- dodo_params
    dodo_params = {
        "cm"         : .1,
        "tau_m"      : 1.,
        "e_rev_E"    : 0.,
        "e_rev_I"    : -90.,
        "v_thresh"   : -52.,
        "tau_syn_E"  : 10.,
        "v_rest"     : -65.,
        "tau_syn_I"  : 10.,
        "v_reset"    : -53.,
        "tau_refrac" : 10.,
        "i_offset"   : 0.,
    }

    dodo_noise = {
        'rate_inh' : 2000.,
        'rate_exc' : 2000.,
        'w_exc'    : .001,
        'w_inh'    : -.0035
    }

    clamp_tso_params = {
        "U": .01,
        "tau_rec": 200.,
        "tau_fac": 0.,
        "weight": 0.*1000.
        # by trying: this is close to the critical weight ie necessary for cont. spiking
    }

    renewing_tso_params = {
        "U": 1.,
        "tau_rec": dodo_params['tau_syn_E'],
        "tau_fac": 0.,
        "weight": 0.*1000.
        # by trying: this is close to the critical weight ie necessary for cont. spiking
    }

    duration = 2e4
    n_neurons = 200

    # === Run experiments =============================
    weights = np.linspace(.01, .1, n_neurons)
    # for w in weights:
    #     clamping_expt(n_neurons, duration, dodo_params, dodo_noise,
    #                   synweight=w, tso_params=clamp_tso_params,
    #                   savename='weight{:.2f}'.format(w))
    # clamping_expt(n_neurons, duration, dodo_params, dodo_noise,
    #               synweight=weights, tso_params=clamp_tso_params,
    #               savename='weight_dependence')

    # === Plot a figure =============================

    # # t vs p_on given synaptic weight
    # filenames = ['weight{:.2f}.pkl'.format(w) for w in weights]
    # samples = []
    # p_on = []
    # plt.figure()
    # plt.xlabel('t')
    # plt.ylabel('p_on')
    # for i, fn in enumerate(filenames):
    #     with open(get_data_path('neuron_clamping') + fn, 'r') as f:
    #         d = cPickle.load(f)
    #         samples.append(d['samples'])

    #     p_on.append(samples[i].mean(axis=1))
    #     plt.plot(np.linspace(0, duration/10., len(p_on[i])), p_on[i], '.',
    #              label='Weight {}'.format(i))

    # plt.savefig(make_figure_folder() + 'activity.png')

    # weight vs p_on given synapse type
    filenames = ['weight_dependence.pkl']
    samples = []
    p_on = []
    plt.figure()
    plt.xlabel('Synapse weight [muS]')
    plt.ylabel('fraction in on state')
    for i, fn in enumerate(filenames):
        with open(get_data_path('neuron_clamping') + fn, 'r') as f:
            d = cPickle.load(f)
            samples.append(d['samples'])
        p_on.append(samples[i].mean(axis=0))
        plt.plot(weights, p_on[i], '.', label=fn)
    plt.legend()
    plt.tight_layout()
    plt.savefig(make_figure_folder() + 'activity.png')

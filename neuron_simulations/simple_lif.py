# script for plotting simple LIF-neuron dynamics
import os
import numpy as np
from lif_pong.utils.data_mgmt import make_figure_folder
import pyNN.nest as sim
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams['font.size'] = 14

# parameters from Mihai's thesis
curr_params = {
        "cm"         : .1,
        "tau_m"      : 20.,
        "v_thresh"   : -40.,
        "tau_syn_E"  : 10.,
        "v_rest"     : -50.,
        "tau_syn_I"  : 10.,
        "v_reset"    : -53.,
        "tau_refrac" : 10.,
        "i_offset"   : 0.,
}

cond_params = {
        "cm"         : .1,
        "tau_m"      : 20.,
        "v_thresh"   : 0.,
        "tau_syn_E"  : 10.,
        "v_rest"     : -50.,
        "tau_syn_I"  : 10.,
        "v_reset"    : -53.,
        "tau_refrac" : 10.,
        "i_offset"   : 0.,
        "e_rev_E"    : 0.,
        "e_rev_I"    : -90.,
}


def lif_iext_expt():
    # === Configure the simulator ============================================
    dt = .01
    sim.setup(timestep=dt, **{"spike_precision": "on_grid", 'quit_on_end': False})

    # === Build and instrument the network ===================================
    duration = 500.
    # sources
    switch_times = np.linspace(0, 1, 3, endpoint=False)*duration
    i_max = 1.2*curr_params['cm']/curr_params['tau_m']*(curr_params['v_thresh'] - curr_params['v_rest'])
    i_ext = np.linspace(0, 1, 3) * i_max
    current_source = sim.StepCurrentSource(times=switch_times, amplitudes=i_ext)

    cuba_lif = sim.IF_curr_exp(**curr_params)
    lif_pop = sim.Population(1, cuba_lif)
    lif_pop.initialize(v=.5*(curr_params['v_thresh'] + curr_params['v_rest']))

    # w_syn = .01
    # connector = sim.AllToAllConnector()
    # projection = sim.Projection(
    #         current_source, lif_pop, connector, receptor_type='excitatory',
    #         synapse_type=sim.StaticSynapse(weight=w_syn))

    lif_pop.inject(current_source)

    lif_pop.record(['v', 'spikes'])

    # === Run the simulation =================================================
    sim.run(duration)

    # === Plotting ===========================================================

    figure_filename = 'test.png'
    fig, ax1 = plt.subplots(figsize=(10, 7))

    vmem = lif_pop.get_data().segments[0].filter(name='v')[0]
    spiketrain = lif_pop.get_data().segments[0].spiketrains[0]
    t = np.linspace(0, duration, len(vmem))
    ax1.plot(t, vmem, 'C0-')
    ax1.set_ylabel('u [mV]', color='C0')
    ax1.tick_params('y', colors='C0')

    i_data = np.repeat(i_ext.reshape(-1, 1), 2, axis=1).flatten()
    t_i = np.zeros_like(i_data)
    t_i[1:-1] = np.repeat(switch_times[1:].reshape(-1, 1), 2, axis=1).flatten()
    t_i[-1] = duration
    ax2 = ax1.twinx()
    ax2.plot(t_i, i_data, 'C1-')
    ax2.set(xlabel='t [ms]')
    ax2.set_ylabel('I_\mathrm{ext} [\mu A]', color='C1')
    ax2.tick_params('y', colors='C1')
    # plt.tight_layout()
    plt.savefig(os.path.join(make_figure_folder(), figure_filename))

    # === Clean up and quit ===================================================
    sim.end()


def lif_isyn_expt():
    # === Configure the simulator ============================================
    dt = .01
    sim.setup(timestep=dt, **{"spike_precision": "on_grid", 'quit_on_end': False})

    # === Build and instrument the network ===================================
    duration = 500.
    # source
    spike_interval = 20.
    spike_times_exc = np.hstack(([.25*duration], .6*duration + np.arange(5)*spike_interval))
    spike_times_inh = []
    spike_source_exc = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times_exc))

    cuba_lif = sim.IF_curr_exp(**curr_params)
    lif_pop = sim.Population(1, cuba_lif)
    lif_pop.initialize(v=curr_params['v_rest'])

    # w_syn = .01
    # connector = sim.AllToAllConnector()
    # projection = sim.Projection(
    #         current_source, lif_pop, connector, receptor_type='excitatory',
    #         synapse_type=sim.StaticSynapse(weight=w_syn))
    weight = .01
    synapse_type = sim.StaticSynapse(weight=weight)
    proj_exc = sim.Projection(spike_source_exc, lif_pop, sim.OneToOneConnector(),
                              receptor_type='excitatory', synapse_type=synapse_type)

    lif_pop.record(['v'])

    # === Run the simulation =================================================
    sim.run(duration)

    # === Plotting ===========================================================

    figure_filename = 'test.png'
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    vmem = lif_pop.get_data().segments[0].filter(name='v')[0]
    t = np.linspace(0, duration, len(vmem))
    ax1.plot(t, vmem, '-')
    ax1.set(ylabel='u [mV]')

    isyn = 0
    for ts in spike_times_exc:
        isyn += weight*exp_kernel(t, ts, curr_params['tau_syn_E'])
    for ts in spike_times_inh:
        isyn -= weight*exp_kernel(t, ts, curr_params['tau_syn_I'])
    ax2.plot(t, isyn, '-')
    ax2.set(xlabel='t [ms]', ylabel='I_ext [uA]')
    # plt.tight_layout()
    plt.savefig(os.path.join(make_figure_folder(), figure_filename))

    # === Clean up and quit ===================================================
    sim.end()


def lif_tso_expt():
    # === Configure the simulator =============================================
    sim.setup(**{"spike_precision": "on_grid", 'quit_on_end': False})

    # === Build and instrument the network ====================================
    duration = 800.

    # model used in sbs
    sim.nest.CopyModel("tsodyks2_synapse", "avoid_pynn_trying_to_be_smart")

    # synapse parameters
    weight = .01
    # nest has different weight units (x1000)
    dep_params = {"U": 0.2, "tau_rec": 500.0, "tau_fac": 0.0,
                  "weight": 1000 * weight}
    # # can normalize weights so that height of first PSP is identical
    # dep_params['weight'] /= dep_params['U']
    renewing_params = {"U": 1., "tau_rec": curr_params['tau_syn_E'],
                       "tau_fac": 0., "weight": 1000 * weight}

    synapse_types = {
        'static': sim.StaticSynapse(weight=weight, delay=0.5),
        'depressing': sim.native_synapse_type("avoid_pynn_trying_to_be_smart")
        (**dep_params),
        'renewing': sim.native_synapse_type("avoid_pynn_trying_to_be_smart")
        (**renewing_params)
    }

    spike_interval = 50.
    first_burst = .2*duration + np.arange(7)*spike_interval
    second_burst = first_burst[-1] + 150. + np.arange(3)*20.
    spike_times = np.hstack((first_burst, second_burst))
    spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times))

    connector = sim.AllToAllConnector()
    populations = {}
    projections = {}
    coba_lif = sim.IF_cond_exp(**cond_params)
    for label in 'static', 'depressing', 'renewing':
        populations[label] = sim.Population(1, coba_lif, label=label)
        populations[label].initialize(v=cond_params['v_rest'])
        populations[label].record(['v', 'gsyn_exc'])
        projections[label] = sim.Projection(spike_source, populations[label],
                                            connector, receptor_type='excitatory',
                                            synapse_type=synapse_types[label])

    spike_source.record('spikes')

    # === Run the simulation ==================================================
    sim.run(duration)

    # === Plot a figure =======================================================
    figure_filename = 'stp_example.png'
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey='col')
    for i, label in zip(range(2), ['static', 'depressing']):
        data = populations[label].get_data().segments[0]
        vmem = data.filter(name='v')[0]
        gsyn = data.filter(name='gsyn_exc')[0]
        t = np.linspace(0, duration, len(vmem))

        axes[i, 0].plot(t, gsyn, label=label, color='C1'.format(i))
        axes[i, 0].set_ylabel('gsyn_exc')

        axes[i, 1].plot(t, vmem, label=label, color='C0'.format(i))
        axes[i, 1].set(xlabel='t [ms]', ylabel='Membrane potential [mV]')

    plt.tight_layout()
    plt.savefig(os.path.join(make_figure_folder(), figure_filename))

    # === Clean up and quit ===================================================
    sim.end()


def exp_kernel(t, t0, tau):
    t = np.array(t)
    result = np.exp(-(t - t0)/tau)
    result[t < t0] = 0
    return result

if __name__ == '__main__':
    # lif_isyn_expt()
    lif_iext_expt()
    # lif_tso_expt()

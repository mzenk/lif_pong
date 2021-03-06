# modified example script from pynn documentation
import numpy as np
from pyNN.utility import get_simulator, init_logging, normalized_filename
import pyNN.nest as sim
from neuron_parameters import wei_curr_params
from lif_pong.utils.data_mgmt import make_figure_folder
from stp_theory import compute_ru_envelope, r_theo_interpolated
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams['font.size'] = 14


def normalize_weight(syn_params):
    syn_params['weight'] /= syn_params['U']  # same PSP height of first spike

# === Configure the simulator ================================================
sim.setup(**{"spike_precision": "on_grid", 'quit_on_end': False})

# === Build and instrument the network =======================================
neuron_params = wei_curr_params
cuba = False
duration = 200.
spike_interval = 10.

# model used in sbs
# sim.nest.CopyModel("tsodyks2_synapse_lbl", "avoid_pynn_trying_to_be_smart_lbl")
sim.nest.CopyModel("tsodyks2_synapse", "avoid_pynn_trying_to_be_smart")

# synapse parameters
weight = .01
# nest has different weight units (x1000)
dep_params = {"U": 0.22, "tau_rec": 50.0, "tau_fac": 0.0,
              "weight": 1000 * weight}
normalize_weight(dep_params)
fac_params = {"U": 0.1, "tau_rec": 10.0, "tau_fac": 300.0,
              "weight": 1000 * weight}
normalize_weight(fac_params)
depfac_params = {"U": 0.1, "tau_rec": 500.0, "tau_fac": 500.0,
                 "weight": 1000 * weight}
normalize_weight(depfac_params)
renewing_params = {"U": 1., "tau_rec": neuron_params['tau_syn_E'], "tau_fac": 0.,
                   "weight": 1000 * weight}

tso_dict = {
    'depressing': dep_params,
    'facilitating': fac_params,
    'dep/fac': depfac_params,
    'renewing': renewing_params
}

synapse_types = {
    'static': sim.StaticSynapse(weight=weight, delay=0.5),
    # 'depressing': sim.TsodyksMarkramSynapse(U=0.5, tau_rec=800.0,
    #                                         tau_facil=0.0, weight=0.01,
    #                                         delay=0.5),
    # 'facilitating': sim.TsodyksMarkramSynapse(U=0.04, tau_rec=100.0,
    #                                           tau_facil=1000.0, weight=0.01,
    #                                           delay=0.5),
    # 'renewing': sim.TsodyksMarkramSynapse(U=1., tau_rec=tau_syn,
    #                                       tau_facil=0., weight=0.01,
    #                                       delay=0.5),
    # properTSO:
    'depressing': sim.native_synapse_type("avoid_pynn_trying_to_be_smart")
    (**dep_params),
    'facilitating': sim.native_synapse_type("avoid_pynn_trying_to_be_smart")
    (**fac_params),
    'dep/fac': sim.native_synapse_type("avoid_pynn_trying_to_be_smart")
    (**depfac_params),
    'renewing': sim.native_synapse_type("avoid_pynn_trying_to_be_smart")
    (**renewing_params)
}

offset = 0. + spike_interval
spike_times = np.arange(offset, duration, spike_interval)
spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=spike_times))

connector = sim.AllToAllConnector()
populations = {}
projections = {}
for label in 'static', 'depressing', 'renewing', 'facilitating':
    if cuba:
        cuba_lif = sim.IF_curr_exp(**neuron_params)
        populations[label] = sim.Population(1, cuba_lif, label=label)
        populations[label].record(['v'])
    else:
        coba_lif = sim.IF_cond_exp(**neuron_params)
        populations[label] = sim.Population(1, coba_lif, label=label)
        populations[label].record(['v', 'gsyn_exc'])
    # populations[label].record('v')
    projections[label] = sim.Projection(spike_source, populations[label],
                                        connector, receptor_type='excitatory',
                                        synapse_type=synapse_types[label])

spike_source.record('spikes')

# === Run the simulation =====================================================
sim.run(duration)

# === Plot a figure ==========================================================
figure_filename = 'stp_example.pdf'
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True, sharey='row')
fig, ax = plt.subplots()
for i, label, alpha in zip(range(2), ['renewing', 'depressing'], [.7, .7]):
    data = populations[label].get_data().segments[0]
    vmem = data.filter(name='v')[0]
    gsyn = data.filter(name='gsyn_exc')[0]
    t = np.linspace(0, duration, len(vmem))

    # theoretical envelope
    kwargs = tso_dict[label]
    w = kwargs.pop('weight')*1e-3

    # calculate theoretical envelope (correction factor takes account of the
    # changed stationary value due to PSC accumulation)
    # r, u = compute_ru_envelope(spike_times, **kwargs)
    if kwargs['tau_fac'] == 0:
        kwargs.pop('tau_fac')
        r = r_theo_interpolated(spike_times, 0., spike_interval, **kwargs)
        u = kwargs['U']
    accum_corr = 1 - np.exp(-spike_interval/neuron_params['tau_syn_E'])
    gsyn_theo = r*u*w / accum_corr

    # # this is what I do in the experiments to make non-renewing comparable to renewing
    # if label != 'renewing':
    #     gsyn_theo *= accum_corr
    #     gsyn *= accum_corr
    # leave out first point because the correction is only correct in the
    # stationary case anyway
    axes[0].plot(t, gsyn, alpha=alpha, label=label, color='C{}'.format(i))
    axes[0].plot(spike_times[1:], gsyn_theo[1:], '--',
                 label='theory ' + label, color='C{}'.format(i))
    axes[0].set_ylabel('gsyn_exc')
    axes[0].legend(loc='upper right')

    axes[1].plot(t, vmem, alpha=alpha, label=label, color='C{}'.format(i))
    axes[1].set_xlabel('t [ms]')
    axes[1].set_ylabel('Membrane potential [mV]')
    axes[1].legend(loc='upper right')
    # ax.plot(t, vmem, alpha=alpha, label=label, color='C{}'.format(i))
    # ax.set_xlabel('t [ms]')
    # ax.set_yticks([])
    # ax.set_ylabel('Membrane potential [a.u.]')
    # ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(make_figure_folder() + figure_filename)

# === Clean up and quit ======================================================
sim.end()

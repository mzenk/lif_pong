# modified example script from pynn documentation
import numpy
from pyNN.utility import get_simulator, init_logging, normalized_filename
import pyNN.nest as sim

# === Configure the simulator ================================================
sim.setup(**{"spike_precision": "on_grid", 'quit_on_end': False})

# === Build and instrument the network =======================================
duration = 2000.
spike_interval = 1.
spike_times = numpy.arange(spike_interval, duration, spike_interval)
spike_source = sim.Population(
    1, sim.SpikeSourceArray(spike_times=spike_times))

# model used in sbs
# sim.nest.CopyModel("tsodyks2_synapse_lbl", "avoid_pynn_trying_to_be_smart_lbl")
sim.nest.CopyModel("tsodyks2_synapse", "avoid_pynn_trying_to_be_smart")
connector = sim.AllToAllConnector()

# parameters
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
weight = .01
# nest has different weight units (x1000)
# tau_syn = 1.
# tau_m = 1.
# cm = tau_m/20.  # To keep g_l constant. pyNN-default for LIF: cm=1., tau_m=20.


def normalize_weight(syn_params):
    syn_params['weight'] /= syn_params['U']  # same PSP height of first spike


dep_params = {"U": 0.002, "tau_rec": 2500.0, "tau_fac": 0.0,
              "weight": 1000 * weight}
print(dep_params)
normalize_weight(dep_params)
print(dep_params)
fac_params = {"U": 0.1, "tau_rec": 10.0, "tau_fac": 300.0,
              "weight": 1000 * weight}
normalize_weight(fac_params)
depfac_params = {"U": 0.1, "tau_rec": 500.0, "tau_fac": 500.0,
                 "weight": 1000 * weight}
normalize_weight(depfac_params)
renewing_params = {"U": 1., "tau_rec": dodo_params['tau_syn_E'], "tau_fac": 0.,
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

populations = {}
projections = {}
for label in 'static', 'depressing', 'renewing', 'facilitating':

    coba_lif = sim.IF_cond_exp(**dodo_params)
    cuba_lif = sim.IF_curr_exp(tau_syn_E=dodo_params['tau_syn_E'],
                               tau_m=dodo_params['tau_m'],
                               cm=dodo_params['cm'])
    populations[label] = sim.Population(1, coba_lif, label=label)
    populations[label].record(['v', 'gsyn_exc'])
    # populations[label].record('v')
    projections[label] = sim.Projection(spike_source, populations[label],
                                        connector, receptor_type='excitatory',
                                        synapse_type=synapse_types[label])

spike_source.record('spikes')

# === Run the simulation =====================================================

sim.run(duration)


# === Save the results, optionally plot a figure =============================

for label, p in populations.items():
    filename = normalized_filename("Results", "tsodyksmarkram_%s" % label,
                                   "pkl", "nest")
    p.write_data(filename, annotations={'script_name': __file__})

import numpy as np
from lif_pong.utils.data_mgmt import make_figure_folder
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams['font.size'] = 14

figure_filename = 'stp_example.png'
fig, axes = plt.subplots(2, 1, figsize=(10, 10))
for label, alpha in zip(['renewing', 'depressing'], [1., .5]):
    data = populations[label].get_data().segments[0]
    vmem = data.filter(name='v')[0]
    gsyn = data.filter(name='gsyn_exc')[0]
    t = np.linspace(0, duration, len(vmem))
    axes[0].plot(t, gsyn, alpha=alpha, label=label)
    axes[1].plot(t, vmem, alpha=alpha, label=label)
    # theoretical envelope
    from stp_theory import compute_ru_envelope
    kwargs = tso_dict[label]
    w = kwargs['weight']*1e-3
    del kwargs['weight']
    r, u = compute_ru_envelope(spike_times, **kwargs)
    gsyn_theo = r*u*w
    acc_correction = 1./(1 - np.exp(-spike_interval/dodo_params['tau_syn_E']))
    axes[0].plot(spike_times, gsyn_theo*acc_correction,
                 '--', label='theory ' + label)
    axes[0].set_ylabel('gsyn_exc')
    axes[0].legend(loc='upper right')
    axes[1].set_xlabel('t [ms]')
    axes[1].set_ylabel('Membrane potential [mV]')
    axes[1].legend(loc='upper right')

plt.tight_layout()
plt.savefig(make_figure_folder() + figure_filename, transparent=True)

# from pyNN.utility.plotting import Figure, Panel
# # figure_filename = normalized_filename("Results", "tsodyksmarkram",
# #                                       "png", "nest")
# panels = []
# # for variable in ('gsyn_exc', 'v'):
# for variable in ['gsyn_exc']:
#     for population in populations.values():
#         data = population.get_data().segments[0].filter(name=variable)[0]
#         panels.append(Panel(data, data_labels=[population.label], yticks=True))
# # add ylabel to top panel in each group
# panels[0].options.update(ylabel=u'Synaptic conductance (muS)')
# # panels[3].options.update(ylabel='Membrane potential (mV)')
# # add xticks and xlabel to final panel
# panels[-1].options.update(xticks=True, xlabel="Time (ms)")

# Figure(*panels, title="Example of plastic synapses"
#        ).save(make_figure_folder() + figure_filename)


# === Clean up and quit =======================================================

sim.end()

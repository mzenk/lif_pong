# encoding: utf-8
"""
Example of depressing and facilitating synapses

Usage: tsodyksmarkram.py [-h] [--plot-figure] [--debug DEBUG] simulator

positional arguments:
  simulator      neuron, nest, brian or another backend simulator

optional arguments:
  -h, --help     show this help message and exit
  --plot-figure  Plot the simulation results to a file.
  --debug DEBUG  Print debugging information

"""

import numpy
from pyNN.utility import get_simulator, init_logging, normalized_filename
import pyNN.nest as sim

# === Configure the simulator ================================================
sim.setup(**{"spike_precision": "on_grid", 'quit_on_end': False})

# === Build and instrument the network =======================================
duration = 500.
spike_times = 10. + numpy.sort(numpy.random.rand(10)*90.)
spike_times = numpy.arange(50., duration, 10.)
spike_source = sim.Population(
    1, sim.SpikeSourceArray(spike_times=spike_times))

# model used in sbs
# sim.nest.CopyModel("tsodyks2_synapse_lbl", "avoid_pynn_trying_to_be_smart_lbl")
sim.nest.CopyModel("tsodyks2_synapse", "avoid_pynn_trying_to_be_smart")

connector = sim.AllToAllConnector()

weight = .01
# nest has different weight units (x1000)
tau_syn = 1.
tau_m = 1.
cm = tau_m/20.  # To keep g_l constant. pyNN-default for LIF: cm=1., tau_m=20.


def normalize_weight(syn_params):
    syn_params['weight'] /= syn_params['U']  # same PSP height of first spike


dep_params = {"U": 0.5, "tau_rec": 1000.0, "tau_fac": 0.0,
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
renewing_params = {"U": 1., "tau_rec": tau_syn, "tau_fac": 0.,
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

    coba_lif = sim.IF_cond_exp(tau_m=tau_m, cm=cm, v_thresh=0.,
                               e_rev_E=0., tau_syn_E=tau_syn,)
    cuba_lif = sim.IF_curr_exp(tau_syn_E=tau_syn, tau_m=tau_m, cm=cm,
                               v_thresh=0.,)
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
from utils.data_mgmt import make_figure_folder
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams['font.size'] = 14

figure_filename = 'stp_example.pdf'
plt.figure()
for label, alpha in zip(['depressing', 'facilitating'], [1., 0.5]):
    data = populations[label].get_data().segments[0]
    u = data.filter(name='v')[0]
    gsyn = data.filter(name='gsyn_exc')[0]
    t = np.linspace(0, duration, len(u))
    plt.plot(t, gsyn, alpha=alpha, label=label)
    # theoretical envelope
    from stp_theory import compute_ru_envelope
    kwargs = tso_dict[label]
    w = kwargs['weight']*1e-3
    del kwargs['weight']
    r, u = compute_ru_envelope(spike_times, **kwargs)
    gsyn_theo = r*u*w
    plt.plot(spike_times, gsyn_theo, '--', label='theory ' + label)
    print(gsyn_theo[0], w, r[0], u[0])

plt.xlabel('t [ms]')
plt.ylabel('gsyn_exc')
plt.xlim([0, 300.])
plt.tight_layout()
plt.legend(loc='upper right')
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
# panels[0].options.update(ylabel=u'Synaptic conductance (µS)')
# # panels[3].options.update(ylabel='Membrane potential (mV)')
# # add xticks and xlabel to final panel
# panels[-1].options.update(xticks=True, xlabel="Time (ms)")

# Figure(*panels, title="Example of plastic synapses"
#        ).save(make_figure_folder() + figure_filename)


# === Clean up and quit =======================================================

sim.end()

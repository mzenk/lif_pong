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


# === Configure the simulator ================================================

sim, options = get_simulator(
    ("--plot-figure", "Plot the simulation results to a file.",
     {"action": "store_true"}),
    ("--debug", "Print debugging information"))

if options.debug:
    init_logging(None, debug=True)

sim.setup(**{"spike_precision": "on_grid", 'quit_on_end': False})

# === Build and instrument the network =======================================
spike_times = 10. + numpy.sort(numpy.random.rand(10)*90.)
spike_times = numpy.arange(110., 200., 10)
spike_source = sim.Population(
    1, sim.SpikeSourceArray(spike_times=spike_times))

# model used in sbs
# sim.nest.CopyModel("tsodyks2_synapse_lbl", "avoid_pynn_trying_to_be_smart_lbl")
sim.nest.CopyModel("tsodyks2_synapse", "avoid_pynn_trying_to_be_smart")

connector = sim.AllToAllConnector()

weight = .01
tau_syn = 1.2
tau_syn_renew = 10.
tau_m = .1
# nest has different weight units (x1000)
dep_params = {"U": 0.1, "tau_rec": 500.0, "tau_fac": 0.0,
              "weight": 1000 * weight}
fac_params = {"U": 0.1,"tau_rec": 10.0, "tau_fac": 500.0,
              "weight": 1000 * weight}
depfac_params = {"U": 0.1,"tau_rec": 500.0, "tau_fac": 500.0,
                 "weight": 1000 * weight}
renewing_params = {"U": 1.,"tau_rec": tau_syn_renew, "tau_fac": 0.,
                   "weight": 1000 * weight}

synapse_types = {
    'static': sim.StaticSynapse(weight=weight, delay=0.5),
    # 'depressing': sim.TsodyksMarkramSynapse(U=0.5, tau_rec=800.0,
    #                                         tau_facil=0.0, weight=0.01,
    #                                         delay=0.5),
    # 'facilitating': sim.TsodyksMarkramSynapse(U=0.04, tau_rec=100.0,
    #                                           tau_facil=1000.0, weight=0.01,
    #                                           delay=0.5),
    # 'renewing': sim.TsodyksMarkramSynapse(U=1., tau_rec=tau_syn_renew,
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
for label in 'static', 'depressing', 'facilitating', 'renewing', 'dep/fac':
    if label == 'renewing':
        tau = tau_syn_renew
    else:
        tau = tau_syn

    populations[label] = sim.Population(
        # 1, sim.IF_cond_exp(e_rev_E=0., tau_syn_E=tau, tau_m=.1),
        1, sim.IF_curr_exp(tau_syn_E=tau, tau_m=tau_m, cm=tau_m/20.),
        label=label)
    # populations[label].record(['v', 'gsyn_exc'])
    populations[label].record('v')
    projections[label] = sim.Projection(spike_source, populations[label],
                                        connector, receptor_type='excitatory',
                                        synapse_type=synapse_types[label])

spike_source.record('spikes')

# === Run the simulation =====================================================

sim.run(300.0)


# === Save the results, optionally plot a figure =============================

for label, p in populations.items():
    filename = normalized_filename("Results", "tsodyksmarkram_%s" % label,
                                   "pkl", options.simulator)
    p.write_data(filename, annotations={'script_name': __file__})


if options.plot_figure:
    from pyNN.utility.plotting import Figure, Panel
    # figure_filename = normalized_filename("Results", "tsodyksmarkram",
    #                                       "png", options.simulator)
    figure_filename = 'stp_test.png'
    panels = []
    # for variable in ('gsyn_exc', 'v'):
    for variable in ('v'):
        for population in populations.values():
            panels.append(
                Panel(population.get_data().segments[0].filter(name=variable)[0],
                      data_labels=[population.label], yticks=True),
            )
    # add ylabel to top panel in each group
    panels[0].options.update(ylabel=u'Synaptic conductance (µS)')
    panels[3].options.update(ylabel='Membrane potential (mV)')
    # add xticks and xlabel to final panel
    panels[-1].options.update(xticks=True, xlabel="Time (ms)")

    Figure(*panels,
           title="Example of static, facilitating, depressing and renewing synapses",
           annotations="Simulated with %s" % options.simulator.upper()
           ).save(figure_filename)
    print(figure_filename)


# === Clean up and quit =======================================================

sim.end()

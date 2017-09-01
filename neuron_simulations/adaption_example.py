"""
A demonstration of the responses of different standard neuron models to current injection.

Usage: python cell_type_demonstration.py [-h] [--plot-figure] [--debug] simulator

positional arguments:
  simulator      neuron, nest, brian or another backend simulator

optional arguments:
  -h, --help     show this help message and exit
  --plot-figure  Plot the simulation results to a file.
  --debug        Print debugging information

"""

from pyNN.utility import get_simulator, init_logging, normalized_filename
from pyNN.utility.plotting import Figure, Panel
import pyNN.nest as sim
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams['font.size'] = 14

# === Configure the simulator ================================================
dt = .001
sim.setup(timestep=dt, min_delay=1.0)


# === Build and instrument the network =======================================
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
dodo_params['i_offset'] = 1.4

default_adex = {
    'tau_refrac': 0.1, 'cm': 0.281, 'tau_m': 9.3667, 'v_rest': -70.6,
    'v_thresh': -50.4, 'v_reset': -70.6, 'i_offset': 0.0,
    'tau_syn_E': 5.0, 'tau_syn_I': 5.0, 'e_rev_E': 0.0, 'e_rev_I': -80.0,
    'v_spike': -40.0, 'delta_T': 2.0, 'a': 4.0, 'b': 0.0805, 'tau_w': 144.0
    }

adex_params = {
    'i_offset': 1.0, 'v_spike': -40.0, 'delta_T': 2.0, 'a': 4.0, 'b': 0.0805,
    'tau_w': 144.0
}

init_burst = {
    'tau_refrac': 0., 'cm': 5./500., 'tau_m': 5., 'v_rest': -70.,
    'v_thresh': -50., 'v_reset': -51., 'i_offset': 65e-3,
    'v_spike': 0.0, 'delta_T': 2.0, 'a': .5e-3, 'b': 7e-3, 'tau_w': 100.0
}

adapting = {
    'tau_refrac': 0., 'cm': 20./500., 'tau_m': 20., 'v_rest': -70.,
    'v_thresh': -50., 'v_reset': -55., 'i_offset': 65e-3,
    'v_spike': 0.0, 'delta_T': 2.0, 'a': 0, 'b': 5e-3, 'tau_w': 100.0
}

coba_exp = sim.Population(1, sim.IF_cond_exp(**dodo_params), label="coba")
adexp = sim.Population(1, sim.EIF_cond_exp_isfa_ista(**adapting),
                       label="adex")

all_neurons = coba_exp + adexp

all_neurons.record('v')
adexp.record('w')

# === Run the simulation =====================================================
duration  = 200.
sim.run(duration)

# === Save the results, optionally plot a figure =============================

filename = normalized_filename("Results", "cell_type_demonstration", "pkl",
                               "nest")
all_neurons.write_data(filename)

figure_filename = 'Figures/adex_example.pdf'
u = adexp.get_data().segments[0].filter(name='v')[0]
u = np.minimum(u.magnitude, -5)
t = np.arange(0, duration + dt, dt)

plt.figure()
plt.title('Adaptive neuron')
plt.plot(t, u, 'b-')
plt.xlabel('t [ms]')
plt.ylabel('u [mV]')
plt.tight_layout()
plt.savefig(figure_filename, transparent=True)
# plt.ylim([])
# t = adexp.times()
# Figure(
#     # Panel(coba_exp.get_data().segments[0].filter(name='v')[0],
#     #       ylabel="Membrane potential (mV)",
#     #       data_labels=[coba_exp.label], yticks=True, ylim=(-53.5, -51.5)),
#     Panel(adexp.get_data().segments[0].filter(name='v')[0],
#           ylabel="u (mV)", yticks=True, ylim=(-60, -20.)),  # data_labels=[adexp.label]
#     # Panel(adexp.get_data().segments[0].filter(name='w')[0],
#     #       ylabel="w (nA)",
#     #       data_labels=[adexp.label], yticks=True, ylim=(0, 400.)),
#     # title="Responses of neurons with different adaption parameters",
# ).save(figure_filename)
# print(figure_filename)

# === Clean up and quit ======================================================
sim.end()

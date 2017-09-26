# testing clamping mechanisms
import numpy
import pyNN.nest as sim
import numpy as np
from utils.data_mgmt import make_figure_folder
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams['font.size'] = 14

# === Configure the simulator ================================================
dt = .1
sim.setup(timestep=dt, **{"spike_precision": "on_grid", 'quit_on_end': False})

# === Build and instrument the network =======================================
spike_times = 10. + numpy.sort(numpy.random.rand(10)*90.)
spike_times = numpy.linspace(10., 260., 25.)
spike_source = sim.Population(
    1, sim.SpikeSourceArray(spike_times=spike_times))

connector = sim.AllToAllConnector()

weight = .01
# nest has different weight units (x1000)
tau_syn = 10.
tau_m = 1.
cm = tau_m/20.  # To keep g_l constant. pyNN-default for LIF: cm=1., tau_m=20.

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
n_biases = 5
v_rest_arr = dodo_params['v_thresh'] + \
    (dodo_params['v_thresh'] - dodo_params['v_rest'])*np.logspace(0, 3, n_biases)
populations = {}
projections = {}
coba_lif = sim.IF_cond_exp(**dodo_params)
population = sim.Population(n_biases, coba_lif)
population.set(v_rest=v_rest_arr)
population.record('v')
# projection = sim.Projection(spike_source, population,
#                             connector, receptor_type='excitatory',
#                             synapse_type=sim.StaticSynapse(weight=weight))

spike_source.record('spikes')

# === Run the simulation =====================================================
duration = 2.
sim.run(duration)

# === Save the results, optionally plot a figure =============================

figure_filename = 'high_bias.pdf'
plt.figure()
vmems = population.get_data().segments[0].filter(name='v')[0].T
for i, v in enumerate(vmems):
    t = np.linspace(0, duration, len(v))
    plt.plot(t, v, '.', label='E_l: {}'.format(v_rest_arr[i]))
plt.xlabel('t [ms]')
plt.ylabel('membrane potential')
plt.tight_layout()
plt.legend()
plt.savefig(make_figure_folder() + figure_filename, transparent=True)

# === Clean up and quit =======================================================

sim.end()

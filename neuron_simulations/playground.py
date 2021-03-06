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
rates = np.linspace(.1, 1, 20) * .1
spike_times = [range(100, 1100, 10), range(1, 100, 10), []]
spike_source = sim.Population(
    3, sim.SpikeSourceArray, cellparams={'spike_times': spike_times})
# for i, st in enumerate(spike_times):
#     spike_source[i].set_parameters(spike_times=st)

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

weights = np.arange(.01, .05, .02)
coba_lif = sim.IF_cond_exp(**dodo_params)
populations = []
projections = []
for i, w in enumerate(weights):
    populations.append(sim.Population(1, coba_lif))
    populations[i].record(['v', 'spikes'])
    projections.append(sim.Projection(
        spike_source, populations[i], connector, receptor_type='excitatory',
        synapse_type=sim.StaticSynapse(weight=w)))

spike_source.record('spikes')

# === Run the simulation =====================================================
duration = 1100.
sim.run(duration)

# === Save the results, optionally plot a figure =============================

figure_filename = 'test.pdf'
plt.figure()

vmem = []
p_on = []
for i, pop in enumerate(populations):
    vmem.append(pop.get_data().segments[0].filter(name='v')[0])
    spiketrain = pop.get_data().segments[0].spiketrains[0]
    p_on.append(len(spiketrain)*10./(duration - 100.))
    t = np.linspace(0, duration, len(vmem[0]))
    plt.plot(t, vmem[i], '-', label='Weight: {:.2f}'.format(weights[i]))
plt.xlabel('t [ms]')
plt.ylabel('membrane potential')
plt.tight_layout()
plt.legend()
plt.savefig(make_figure_folder() + figure_filename)

plt.plot(weights, p_on, '.')
plt.savefig(make_figure_folder() + 'scan.pdf')

# === Clean up and quit =======================================================

sim.end()

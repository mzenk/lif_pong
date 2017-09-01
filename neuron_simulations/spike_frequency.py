import pyNN.nest as sim
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def count_spikes(v_trace, v_trigger):
    return np.sum(np.isclose(v_trace, v_trigger, rtol=1e-4), axis=0)

# === Configure the simulator ================================================
dt = .01
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

n_theta = 100
thetas = np.linspace(-55., -51., n_theta)
adex_params = {
    'i_offset': 1.0, 'v_thresh': thetas,
    'v_spike': -40.0, 'delta_T': 2.0, 'a': 0., 'b': 0., 'tau_w': 144.0
}

coba_exp = sim.Population(1, sim.IF_cond_exp(**dodo_params), label="coba")
adexp = sim.Population(n_theta, sim.EIF_cond_exp_isfa_ista(**adex_params),
                       label="adex")

all_neurons = coba_exp + adexp

all_neurons.record('v')
adexp.record('w')

# === Run the simulation =====================================================
duration = 1000.
sim.run(duration)
all_neurons.write_data('adaption_sweep.pkl')
# === Calculate spike frequency ==============================================

figure_filename = 'Figures/adex_example.png'
time = np.arange(0., duration + dt, dt)
v_adex = adexp.get_data().segments[0].filter(name='v')[0]
v_lif = coba_exp.get_data().segments[0].filter(name='v')[0]

# plt.subplot(121)
# plt.plot(v_lif.times, v_lif, '.')
# plt.subplot(122)
# plt.plot(v_adex.times, v_adex[:, 0], '.')
# plt.show()

ref_adex = .5*(adexp.get('v_thresh') - adexp.get('v_reset')) \
    + adexp.get('v_reset')
ref_coba = .9*(coba_exp.get('v_thresh') - coba_exp.get('v_reset')) \
    + coba_exp.get('v_reset')
nu_adex = count_spikes(v_adex.magnitude, ref_adex) / duration
nu_lif = count_spikes(v_lif.magnitude, ref_coba) / duration
plt.plot(thetas, nu_adex, '.-b', label='AdEx firing rate')
plt.plot([thetas[0], thetas[-1]], [nu_lif, nu_lif], '-g', label='LIF firing rate')
plt.xlabel('Threshold parameter (AdEx)')
plt.ylabel('Mean spiking frequency')
plt.legend()
plt.savefig('adaption_sweep.png')
# === Clean up and quit ======================================================
sim.end()

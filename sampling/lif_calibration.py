# LIF-sampling experiment with MNIST
from __future__ import division
import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl
mpl.use('Agg')
import sbs
import matplotlib.pyplot as plt

sbs.gather_data.set_subprocess_silent(True)
log = sbs.log

# The backend of choice. Both should work but when using neuron, we need to
# disable saturating synapses for now.
sim_name = "pyNN.nest"
# sim_name = "pyNN.neuron"


def calibration(neuron_params, noise_params, calib_name='calibration',
                cuba=False):
    """
        A sample calibration procedure.
    """
    # Since we only have the neuron parameters for now, lets create those first
    if cuba:
        nparams = sbs.db.NeuronParametersCurrentExponential(**{k: v
            for k, v in neuron_params.iteritems() if not k.startswith("e_rev_")
            })
    else:
        nparams = sbs.db.NeuronParametersConductanceExponential(**neuron_params)

    # Now we create a sampler object. We need to specify what simulator we want
    # along with the neuron model and parameters.
    # The sampler accepts both only the neuron parameters or a full sampler
    # configuration as argument.
    sampler = sbs.samplers.LIFsampler(nparams, sim_name=sim_name)

    # Now onto the actual calibration. For this we only need to specify our
    # source configuration and how long/with how many samples we want to
    # calibrate.

    source_config = sbs.db.PoissonSourceConfiguration(
        rates=np.array([noise_params['rate_inh'], noise_params['rate_exc']]),
        weights=np.array([noise_params['w_inh'], noise_params['w_exc']]),
        )

    # We need to specify the remaining calibration parameters
    calibration = sbs.db.Calibration(
            duration=1e5, num_samples=150, burn_in_time=500., dt=0.01,
            source_config=source_config,
            sim_name=sim_name,
            sim_setup_kwargs={"spike_precision": "on_grid"})
    # Do not forget to specify the source configuration!

    # here we could give further kwargs for the pre-calibration phase when the
    # slope of the sigmoid is searched for
    sampler.calibrate(calibration)

    # Afterwards, we need to save the calibration.
    if cuba:
        sampler.write_config(calib_name + "_curr")
    else:
        sampler.write_config(calib_name)

    # plot membrane distribution and activation function
    vmem_dist(calib_name + '.json')


def sigma_fct(x, x0, alpha):
    return 1/(1 + np.exp(-(x - x0) / alpha))


def vmean(v_rest, neuron_params, noise_params):
    g_l = neuron_params['cm'] / neuron_params['tau_m']
    g_inh = noise_params['w_inh'] * noise_params['rate_inh'] * \
        neuron_params['tau_syn_I']
    g_exc = noise_params['w_exc'] * noise_params['rate_exc'] * \
        neuron_params['tau_syn_E']
    return (g_l*v_rest + neuron_params['i_offset'] +
            g_inh*neuron_params['e_rev_I'] +
            g_exc*neuron_params['e_rev_E']) / (g_l + g_inh + g_exc)


def vmem_dist(config_file):
    sampler_config = sbs.db.SamplerConfiguration.load(config_file)

    sampler = sbs.samplers.LIFsampler(sampler_config, sim_name=sim_name)

    sampler.measure_free_vmem_dist(duration=1e6, dt=0.01, burn_in_time=500.)

    # plot calibration and free membrane potential
    vmem_trace = sampler.free_vmem['trace']
    vmem_histo, bin_edges = np.histogram(vmem_trace, bins=200, normed=True)
    plt.figure()
    plt.bar(bin_edges[:-1], vmem_histo, width=np.diff(bin_edges))
    plt.xlabel('Free V_mem')
    plt.ylabel('P')
    plt.savefig('free_vmem_dist.png')


def plot_calibration(config_file, mean_as_x=False, neuron_params=None,
                     noise_params=None):
    sampler_config = sbs.db.SamplerConfiguration.load(config_file)
    sampler = sbs.samplers.LIFsampler(sampler_config, sim_name=sim_name)

    samples_vrest = sampler.calibration.get_samples_v_rest()
    x_data = samples_vrest
    if mean_as_x:
        assert neuron_params is not None and noise_params is not None
        x_data = vmean(samples_vrest, neuron_params, noise_params)
    samples_pon = sampler.calibration.samples_p_on

    p0 = [(x_data.max() + x_data.min())/2,
          .25 * (x_data.max() - x_data.min()) /
          (samples_pon.max() - samples_pon.min())]
    popt, pcov = curve_fit(sigma_fct, x_data, samples_pon, p0=p0)
    sbs_popt = (sampler.calibration.fit.v_p05, sampler.calibration.fit.alpha)
    plt.figure()
    plt.plot(x_data, samples_pon, 'x', label='data')
    plt.plot(x_data, sigma_fct(x_data, *sbs_popt), 'r-', label='sbs-fit')
    plt.plot(x_data, sigma_fct(x_data, *popt), 'g--', alpha=.5, label='fit')
    plt.xlabel('V_rest')
    plt.ylabel('P(on)')
    plt.legend()
    plt.savefig('calibration.png')
    print('SBS-fit parameters: u0 = {}, alpha = {}'.format(*sbs_popt))
    print('My fit parameters: u0 = {2:.4f} +- {0:.4f},'
          ' alpha = {3:.4f} +- {1:.4f}'.format(pcov[0, 0], pcov[1, 1], *popt))


# some example neuron parameters
tutorial_params = {
    "cm"         : .2,
    "tau_m"      : 1.,
    "e_rev_E"    : 0.,
    "e_rev_I"    : -100.,
    "v_thresh"   : -50.,
    "tau_syn_E"  : 10.,
    "v_rest"     : -50.,
    "tau_syn_I"  : 10.,
    "v_reset"    : -50.001,
    "tau_refrac" : 10.,
    "i_offset"   : 0.,
}

tutorial_noise = {
    'rate_inh' : 3000.,
    'rate_exc' : 3000.,
    'w_exc'    : .001,
    'w_inh'    : -.001
}

# parameters from Mihai's thesis ---> use these
sample_params = {
        "cm"         : .1,
        "tau_m"      : 20.,
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

noise_params = {
    'rate_inh' : 5000.,
    'rate_exc' : 5000.,
    'w_exc'    : .0035,
    'w_inh'    : -55./35 * .0035 # haven't understood yet how V_g is determined
}

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

if __name__ == '__main__':
    # calibrate first if necessary
    # calibration(dodo_params, dodo_noise, 'calibrations/dodo_calib')

    # # check HCS
    # vmem_dist('calibrations/dodo_calib.json')

    # re-plot activation function
    plot_calibration('calibrations/dodo_calib.json')

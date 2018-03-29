# testing clamping mechanisms
# --> this script is not optimized for usability
import numpy
import pyNN.nest as sim
import numpy as np
from scipy.optimize import curve_fit
import cPickle
import os
from lif_pong.sampling.lif_clamped_sampling import get_clamp_weights
from lif_pong.utils.data_mgmt import make_figure_folder, make_data_folder, get_data_path
from stp_theory import compute_ru_envelope, r_theo_interpolated
from neuron_parameters import *
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sbs
from sbs.cutils import generate_states


def setup_samplers(n_neurons, neuron_params, noise_params):
    # setup populations
    cuba = 'e_rev_E' not in neuron_params.keys()
    if cuba:
        neuron_model = sim.IF_curr_exp(**neuron_params)
    else:
        neuron_model = sim.IF_cond_exp(**neuron_params)
    population = sim.Population(n_neurons, neuron_model)
    exc_source = sim.Population(
        n_neurons, sim.SpikeSourcePoisson(rate=noise_params['rate_exc']))
    inh_source = sim.Population(
        n_neurons, sim.SpikeSourcePoisson(rate=noise_params['rate_inh']))
    # projections
    proj_exc = sim.Projection(exc_source, population,
                              connector=sim.OneToOneConnector(),
                              receptor_type='excitatory',
                              synapse_type=sim.StaticSynapse())
    proj_exc.set(weight=noise_params['w_exc'])
    proj_inh = sim.Projection(inh_source, population,
                              connector=sim.OneToOneConnector(),
                              receptor_type='inhibitory',
                              synapse_type=sim.StaticSynapse())
    proj_inh.set(weight=-2*(cuba - 0.5)*np.abs(noise_params['w_inh']))
    return population, exc_source, inh_source, proj_exc, proj_inh


def get_samples(spiketrains, dt, tau_refrac, duration):
    spike_list = []
    for st in spiketrains:
        spike_id = st.annotations['source_index']
        id_t = np.vstack((np.repeat(spike_id, len(st.magnitude)),
                          st.magnitude))
        spike_list.append(id_t)
    n_neurons = len(spike_list)
    spike_data = np.hstack(spike_list)
    ordered_spike_data = spike_data[:, np.argsort(spike_data[1])]
    return generate_states(
        spike_ids=ordered_spike_data[0].astype(int),
        spike_times=np.array(ordered_spike_data[1]/dt, dtype=int),
        tau_refrac_pss=np.array([tau_refrac/dt] * n_neurons, dtype=int),
        num_samplers=n_neurons,
        steps_per_sample=int(tau_refrac/dt),
        duration=np.array(duration/dt, dtype=int)
        )


def clamping_expt(n_neurons, duration, neuron_params, noise_params, calib_file,
                  bias_shift=4, tso_params=None, dt=.1, spike_interval=1.,
                  bias=0., savename='test', store_vmem=False, clamp_offset=0.,
                  burn_in_time=0.):
    '''
        Run a simple experiment with the following setup
        populations:
        - spike source with constant firing rate (defautl = 100 Hz)
        - n_neurons LIF neurons
        - n_neurons exc/inh poisson noise sources
        connections:
        - each LIF neuron has poisson input
        - spike sources -> LIF neurons (all2all, tso): synweight (scalar/array)
    '''
    duration += burn_in_time
    clamp_offset += burn_in_time
    sim.setup(timestep=dt, spike_precision="on_grid", quit_on_end=False)
    cuba = 'e_rev_E' not in neuron_params.keys()
    # setup samplers
    population, exc_source, inh_source, exc_proj, inh_proj = \
        setup_samplers(n_neurons, neuron_params, noise_params)
    projections = {}
    projections['exc'] = exc_proj
    projections['inh'] = inh_proj
    # set resting potential according to bias
    sampler_config = sbs.db.SamplerConfiguration.load(calib_file)
    calib_fit = sampler_config.calibration.fit
    population.set(v_rest=calib_fit.v_p05 + calib_fit.alpha*bias)
    print('Set v_rest to {}'.format(population.get('v_rest')))

    if store_vmem:
        if cuba:
            population.record(['v', 'spikes'])
        else:
            population.record(['v', 'gsyn_exc', 'gsyn_inh', 'spikes'])
    else:
        population.record(['spikes'])

    # setup spiking input
    spike_times = numpy.arange(clamp_offset + dt, duration, spike_interval)
    spike_source = sim.Population(
        1, sim.SpikeSourceArray(spike_times=spike_times))
    inh_spike_source = sim.Population(
        1, sim.SpikeSourceArray(spike_times=spike_times))

    clamp_dict = {
        'tso_params': tso_params,
        'spike_interval': spike_interval,
        'bias_shift': np.abs(bias_shift)
    }
    tau_syn = neuron_params['tau_syn_E']
    gl = neuron_params['cm']/neuron_params['tau_m']
    alpha_w = calib_fit.alpha * gl * spike_interval / tau_syn
    exc_weights, inh_weights = \
        get_clamp_weights(clamp_dict, bias, alpha_w, tau_syn)

    exc_weights = np.maximum(exc_weights, np.zeros_like(exc_weights))
    inh_weights = np.minimum(inh_weights, np.zeros_like(inh_weights))

    if tso_params is None:
        # do not correct for accumulation with static synapses
        # (otherwise identical to renewing after burn-in)
        bn_synapse = sim.StaticSynapse()
    else:
        sim.nest.CopyModel("tsodyks2_synapse", "avoid_pynn_trying_to_be_smart")
        bn_synapse = sim.native_synapse_type("avoid_pynn_trying_to_be_smart")(
            **tso_params)
        exc_weights *= 1000.
        inh_weights *= 1000.

    try:
        exc_weights[bias_shift < 0] = 0
        inh_weights[bias_shift >= 0] = 0
    except TypeError, IndexError:
        if bias_shift > 0:
            inh_weights = 0
        else:
            exc_weights = 0

    # if np.array(synweight).size == 1:
    #     exc_weights = (synweight > 0)*synweight
    #     inh_weights = -1*(synweight < 0)*synweight
    # else:
    #     exc_weights = np.maximum(synweight, np.zeros_like(synweight))
    #     inh_weights = np.maximum(-synweight, np.zeros_like(synweight))
    # if cuba:
    #     inh_weights *= -1.

    projections['ext_exc'] = sim.Projection(spike_source, population,
                                            connector=sim.AllToAllConnector(),
                                            receptor_type='excitatory',
                                            synapse_type=bn_synapse)
    projections['ext_inh'] = sim.Projection(inh_spike_source, population,
                                            connector=sim.AllToAllConnector(),
                                            receptor_type='inhibitory',
                                            synapse_type=bn_synapse)
    projections['ext_exc'].set(weight=exc_weights)
    projections['ext_inh'].set(weight=inh_weights)
    sim.run(duration)

    # === Save Data ===========================================================
    spiketrains = population.get_data().segments[0].spiketrains
    samples = get_samples(spiketrains, dt, neuron_params['tau_refrac'],
                          duration)[int(burn_in_time/spike_interval):]

    with open(os.path.join(make_data_folder(), savename + '.pkl'), 'w') as f:
        # apparently, pyNN changes weight units after I set them?!
        pynn_weights = np.diag(projections['ext_exc'].get('weight', format='array')) + \
            np.diag(projections['ext_inh'].get('weight', format='array'))
        if store_vmem:
            vmem = population.get_data().segments[0].filter(name='v')[0]
            if not cuba:
                gexc = population.get_data().segments[0].filter(name='gsyn_exc')[0]
                ginh = population.get_data().segments[0].filter(name='gsyn_inh')[0]

                cPickle.dump({'vmem': vmem, 'gsyn_exc': gexc, 'gsyn_inh': ginh,
                              'samples': samples,
                              'weights': pynn_weights,
                              'bias': bias},
                             f)
            else:
                cPickle.dump({'vmem': vmem, 'samples': samples,
                              'weights': pynn_weights,
                              'bias': bias},
                             f)
        else:
            cPickle.dump({'samples': samples, 'bias': bias,
                          'weights': pynn_weights}, f)

    # === Clean up and quit ===================================================
    sim.end()


def get_calibration(leak_potentials, duration, neuron_params, noise_params,
                    dt=.1, savename='calibration'):
    sim.setup(timestep=dt, spike_precision="on_grid", quit_on_end=False)

    # setup samplers
    population, exc_source, inh_source, exc_proj, inh_proj = \
        setup_samplers(len(leak_potentials), neuron_params, noise_params)
    projections = {}
    projections['exc'] = exc_proj
    projections['inh'] = inh_proj
    population.set(v_rest=leak_potentials)
    population.record(['v', 'spikes'])

    sim.run(duration)

    # === Save Data ===========================================================
    vmem = population.get_data().segments[0].filter(name='v')[0].T
    spiketrains = population.get_data().segments[0].spiketrains
    samples = \
        get_samples(spiketrains, dt, neuron_params['tau_refrac'], duration)
    with open(os.path.join(make_data_folder(), savename + '.pkl'), 'w') as f:
        cPickle.dump({'samples': samples, 'E_l': leak_potentials,
                      'vmem': vmem.magnitude}, f)


def plot_w_vs_activation(filenames, show_fit=True):
    samples = []
    p_on = []
    plt.figure()
    plt.xlabel('Synapse weight [muS]')
    plt.ylabel('fraction in on state')
    for i, fn in enumerate(filenames):
        with open(get_data_path('neuron_clamping') + fn, 'r') as f:
            d = cPickle.load(f)
            samples.append(d['samples'])
            weights = d['weights']
        p_on.append(samples[i].mean(axis=0))
        plt.plot(weights, p_on[i], '.', label=fn)
        if show_fit:
            # fit
            fit_mask = np.logical_and(p_on[i] > .05, p_on[i] < .95)
            if not np.any(fit_mask):
                print('Range of data: [{}, {}]'.format(p_on.min(), p_on.max()))
                fit_mask = np.logical_not(fit_mask)
            x_fit = weights[fit_mask]
            y_fit = p_on[i][fit_mask]
            p0 = [x_fit.mean(), 1./(x_fit.max() - x_fit.min())]
            popt, pcov = curve_fit(sigma_fct, x_fit, y_fit, p0=p0)
            plt.plot(x_fit, sigma_fct(x_fit, *popt), 'C1-', label='fit')
            plt.plot(weights, sigma_fct(weights, *popt), 'C1--')

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(make_figure_folder(), 'w_vs_act.png'))


def plot_activation_fct(data_file, config_file):
    with open(get_data_path('neuron_clamping') + data_file, 'r') as f:
        d = cPickle.load(f)
        samples = d['samples']
        v_rests = d['E_l']
    p_on = samples.mean(axis=0)

    fit_mask = np.logical_and(p_on > .03, p_on < .97)
    if not np.any(fit_mask):
        print('Range of data: [{}, {}]'.format(p_on.min(), p_on.max()))
        fit_mask = np.logical_not(fit_mask)
    x_fit = v_rests[fit_mask]
    y_fit = p_on[fit_mask]
    p0 = [x_fit.mean(), 1./(x_fit.max() - x_fit.min())]
    popt, pcov = curve_fit(sigma_fct, x_fit, y_fit, p0=p0)
    print('Fit parameters: {}, {}'.format(*popt))
    print('Maximum on-fraction: {}'.format(p_on.max()))

    plt.figure()
    plt.xlabel('$E_l$ [mV]')
    plt.ylabel('$p(z = 1)$')

    plt.plot(v_rests, p_on, 'x', label='Simulation')
    # sampler_config = sbs.db.SamplerConfiguration.load(config_file)
    # sampler = sbs.samplers.LIFsampler(sampler_config, sim_name='pyNN.nest')
    # sbs_popt = (sampler.calibration.fit.v_p05, sampler.calibration.fit.alpha)
    # plt.plot(v_rests, sigma_fct(v_rests, *sbs_popt), label='sbs')
    plt.plot(v_rests, sigma_fct(v_rests, *popt),
             label=r'$\alpha$ = ' + '{:.1E} mV,\n'.format(popt[1]) + r'$\bar u^0$ = ' + '{:.1E} mV'.format(popt[0]))
    plt.legend()
    # plt.xticks(-50 + np.arange(-.3, .2, .1))
    plt.tight_layout()

    plt.savefig(os.path.join(make_figure_folder(), 'act_fct.png'))
    plt.savefig(os.path.join(make_figure_folder(), 'act_fct.pdf'))


def plot_vmem_dist(data_file):
    with open(get_data_path('neuron_clamping') + data_file, 'r') as f:
        d = cPickle.load(f)
        vmem = d['vmem']
        v_rests = d['E_l']

    plt.figure()
    plt.xlabel('V_mem [mV]')
    plt.ylabel('Count')
    for i, v in enumerate(vmem):
        plt.hist(v, bins='auto', alpha=.6, label='E_l='.format(v_rests[i]))
    plt.savefig(os.path.join(make_figure_folder(), 'vmem_dist.png'))


def plot_bias_comparison(data_files, biases=None, savename=None):
    if biases is None:
        biases = []
    fit_params = []
    fit_covs = []
    fig1 = plt.figure()
    ax1 = fig1.add_axes([.1, .15, .65, .8])
    ax1.set_xlabel('weight')
    ax1.set_ylabel('p_on')
    cmap = mpl.cm.cool
    color_idx = np.linspace(0, 1, len(data_files))
    for i, f in zip(color_idx, data_files):
        with open(get_data_path('neuron_clamping') + f, 'r') as f:
            d = cPickle.load(f)
            samples = d['samples']
            weights = d['weights']
            if 'bias' in d.keys():
                biases.append(d['bias'])
        p_on = samples.mean(axis=0)
        # fit
        fit_mask = np.logical_and(p_on > .05, p_on < .95)
        if not np.any(fit_mask):
            print('Range of data: [{}, {}]'.format(p_on.min(), p_on.max()))
            fit_mask = np.logical_not(fit_mask)
        x_fit = weights[fit_mask]
        y_fit = p_on[fit_mask]
        p0 = [x_fit.mean(), 1./(x_fit.max() - x_fit.min())]
        popt, pcov = curve_fit(sigma_fct, x_fit, y_fit, p0=p0)
        fit_params.append(popt)
        fit_covs.append(pcov)
        ax1.plot(weights, p_on, '.', color=cmap(i))
        ax1.plot(x_fit, sigma_fct(x_fit, *popt), '-', color=cmap(i))

    ax1c = fig1.add_axes([.8, .15, .05, .8])
    norm = mpl.colors.Normalize(vmin=min(biases), vmax=max(biases))
    cb1 = mpl.colorbar.ColorbarBase(ax1c, cmap=cmap, norm=norm)
    cb1.set_label('Bias')
    plt.savefig(os.path.join(make_figure_folder(), savename + '_actfct.png'))

    assert len(biases) != 0
    fit_params = np.array(fit_params)
    fit_covs = np.array(fit_covs)
    pcoeff, vcoeff = np.polyfit(biases, fit_params[:, 0], 1, cov=True)
    fig2, ax2 = plt.subplots()
    ax2.errorbar(biases, fit_params[:, 0], fmt='.',
                 yerr=np.sqrt(fit_covs[:, 0, 0]))
    ax2.plot(biases, pcoeff[0]*np.array(biases) + pcoeff[1])
    ax2.set_xlabel('bias')
    ax2.set_ylabel('wp05')
    plt.tight_layout()
    plt.savefig(os.path.join(make_figure_folder(), savename + '_wp05.png'))

    fig3, ax3 = plt.subplots()
    ax3.errorbar(biases, fit_params[:, 1], fmt='.',
                 yerr=np.sqrt(fit_covs[:, 1, 1]))
    ax3.set_xlabel('bias')
    ax3.set_ylabel('alpha')
    plt.tight_layout()
    plt.savefig(os.path.join(make_figure_folder(), savename + '_alpha.png'))

    # save fit parameters
    if savename is not None:
        np.savez('calibrations/' + savename, wp05=pcoeff[1],
                 alpha=np.mean(fit_params[:, 1]), bias_factor=pcoeff[0])

    return biases, fit_params, fit_covs


def analyse_sweep(filenames, n_avg=20, filterlength=20):
    samples = []
    p_on = []
    # smooth_p_on = []
    weights = []
    for i, fn in enumerate(filenames):
        with open(get_data_path('neuron_clamping') + fn, 'r') as f:
            d = cPickle.load(f)
            samples.append(d['samples'])
            assert isinstance(d['weights'], (float, np.float64)), \
                'Has type: ' + str(type(d['weights']))
            weights.append(d['weights'])
        p_on.append(samples[i].mean(axis=1))
        # kernel = gaussian(np.linspace(-3, 3, n_avg))
        # kernel = np.ones(n_avg)
        # kernel /= kernel.sum()
        # smooth_p_on.append(np.convolve(p_on[-1], kernel, mode='same'))

    p_on = np.array(p_on)
    weights = np.array(weights)
    # measure stationary value
    p_stat = np.mean(p_on[:, -n_avg:], axis=1)
    p_stat_std = np.std(p_on[:, -n_avg:], axis=1)
    # # measure clamp duration as time until stationary state is reached
    # # much too inaccurate
    # smooth_p_on = np.array(smooth_p_on)
    # smooth_p_on = (smooth_p_on - np.expand_dims(p_stat, 1)) / \
    #     np.expand_dims(smooth_p_on.max(axis=1) - smooth_p_on.min(axis=1), 1)
    # stat_idx = ?

    plt.errorbar(weights*clamp_tso_params['U']*.2, p_stat, yerr=p_stat_std, fmt='.')
    plt.xlabel('Weight')
    plt.ylabel('Stationary activity')
    plt.savefig(os.path.join(make_figure_folder(), 'pstat.png'))
    return (p_stat, p_stat_std)


def plot_time_evolution(data_file, clamp_dict, calib_file, bias=0, labels=None):
    if labels is None:
        labels = ['file {}'.format(i) for i in range(len(data_file))]
    sampler_config = sbs.db.SamplerConfiguration.load(calib_file)
    tau_syn = sampler_config.neuron_parameters.tau_syn_E
    gl = sampler_config.neuron_parameters.cm / \
        sampler_config.neuron_parameters.tau_m
    tso_params = clamp_dict['tso_params']
    clamp_offset = clamp_dict['offset']
    spike_interval = clamp_dict['spike_interval']
    alpha_w = gl*spike_interval/tau_syn*sampler_config.calibration.fit.alpha

    plt.figure()
    plt.xlabel('time [ms]')
    plt.ylabel(r'$p_{on}$')
    for i, fn in enumerate(data_file):
        with open(get_data_path('neuron_clamping') + fn, 'r') as f:
            d = cPickle.load(f)
            samples = d['samples']
            weights = d['weights']
        # assuming file contains multiple runs of same exp't -> average
        p_on = samples.mean(axis=1)
        timeaxis = np.linspace(0, 10*len(samples), len(p_on))
        plt.plot(timeaxis, p_on, 'C{}.'.format(i), label=labels[i], alpha=.8)

        if tso_params is not None:
            r_theo = r_theo_interpolated(
                timeaxis, clamp_offset, spike_interval, tso_params['U'],
                tso_params['tau_rec'])
            effweight = weights*tso_params['U']*r_theo
        else:
            effweight = weights*np.ones_like(timeaxis)

        effweight[timeaxis < clamp_offset] = 0
        p_theo = sigma_fct(bias + effweight/alpha_w)
        plt.plot(timeaxis, p_theo, 'k:'.format(i))
    plt.legend()
    plt.savefig(os.path.join(make_figure_folder(), 'decay.png'))
    plt.savefig(os.path.join(make_figure_folder(), 'decay.pdf'))


def plot_vg_traces(data_files, duration):
    for i, fn in enumerate(data_files):
        with open(get_data_path('neuron_clamping') + fn, 'r') as f:
            d = cPickle.load(f)
            vmem = d['vmem']
            try:
                gexc = d['gsyn_exc']
                ginh = d['gsyn_inh']
            except KeyError:
                print('No conductance data found.')
                gexc = np.zeros_like(vmem)
                ginh = np.zeros_like(vmem)
            samples = d['samples']
            weights = d['weights']
        fig, ax = plt.subplots(4, 1, figsize=(10, 15))
        t = np.linspace(0, duration, len(vmem))
        ax[0].plot(t, vmem, label=weights, alpha=.7)
        ax[1].plot(t, gexc, label=weights, alpha=.7)
        ax[2].plot(t, ginh, label=weights, alpha=.7)
        ax[3].imshow(samples.T, interpolation='Nearest', cmap='gray',
                     extent=(0, duration, 0, samples.shape[1]), aspect='auto')
        lower_lim = vmem[vmem.shape[0]//4:].min().magnitude
        upper_lim = vmem.max().magnitude
        lower_lim -= .05*(upper_lim - lower_lim)
        upper_lim += .05*(upper_lim - lower_lim)
        ax[0].set_ylim([lower_lim, upper_lim])
        try :
            labels = ['{:.2f}'.format(w) for w in weights]
        except TypeError:
            labels = ['{:.2f}'.format(weights)]
        ax[0].legend(labels)
        ax[1].legend(labels)
        ax[2].legend(labels)
        ax[0].set_ylabel('v_mem')
        ax[1].set_ylabel('g_syn (exc)')
        ax[2].set_ylabel('g_syn (inh)')
        ax[3].set_ylabel('neuron index')
        ax[3].set_xlabel('time')
        plt.savefig(os.path.join(make_figure_folder(), 'traces{}.png'.format(i)))


def save_calib_fit(data_file):
    with open(get_data_path('neuron_clamping') + data_file + '.pkl', 'r') as f:
        d = cPickle.load(f)
        samples = d['samples']
        weights = d['weights']
    p_on = samples.mean(axis=0)
    # fit
    fit_mask = np.logical_and(p_on > .05, p_on < .95)
    if not np.any(fit_mask):
        print('Range of data: [{}, {}]'.format(p_on.min(), p_on.max()))
        fit_mask = np.logical_not(fit_mask)
    x_fit = weights[fit_mask]
    y_fit = p_on[fit_mask]
    p0 = [x_fit.mean(), 1./(x_fit.max() - x_fit.min())]
    popt, pcov = curve_fit(sigma_fct, x_fit, y_fit, p0=p0)
    np.savez(os.path.join(make_data_folder(), data_file + '_fit'), popt=popt, pcov=pcov)


def sigma_fct(x, x0=0, alpha=1):
    return 1/(1 + np.exp(-(x - x0)/alpha))


def gaussian(x, mu=0., sigma=1.):
    return np.exp(-.5*(x - mu)**2/sigma**2)/np.sqrt(2 * sigma**2)


if __name__ == '__main__':
    mpl.rcParams['font.size'] = 14
    # Parameters
    config_file = '../sampling/calibrations/wei_curr_calib_improved.json'
    neuron_params = wei_curr_params
    noise_params = wei_curr_noise

    clamp_tso_params = {
        "U": .0006,
        "tau_rec": 40000.,
        "tau_fac": 0.,
        "weight": 0.*1000.
    }

    # clamp_tso_params = None

    renewing_tso_params = {
        "U": 1.,
        "tau_rec": neuron_params['tau_syn_E'],
        "tau_fac": 0.,
        "weight": 0.*1000.
    }

    # === Run experiments =============================
    # # simple run
    # duration = 2000
    # n_neurons = 1
    # weights = np.linspace(-.1, .1, n_neurons)
    # clamping_expt(n_neurons, duration, neuron_params, noise_params, config_file,
    #               synweight=weights, tso_params=renewing_tso_params,
    #               savename='renewing', store_vmem=True)

    # # frequency sweep --- use static synapses!
    # duration = 1e5
    # n_neurons = 200
    # savename = 'frequency_sweep'
    # weights = np.linspace(-.01, .01, n_neurons)
    # frequencies = np.linspace(0.05, 1, 8)  # in kHz
    # for i, freq in enumerate(frequencies):
    #     clamping_expt(n_neurons, duration, neuron_params, noise_params, config_file,
    #                   synweight=weights, tso_params=None, spike_interval=1./freq,
    #                   savename='freq_sweep{:02d}'.format(i), burn_in_time=100.)

    # duration = 1e5
    # n_neurons = 100
    # leak_potentials = -50 + np.linspace(-.005, .005, n_neurons)
    # get_calibration(leak_potentials, duration, neuron_params, noise_params,
    #                 savename='wei_curr_calibdata')

    # activity for depressing synapse
    n_neurons = 1000
    clamp_offset = 200.
    duration = 10000. + clamp_offset
    spike_interval = 1.
    bias_shifts = [6]
    for i, bias_shift in enumerate(bias_shifts):
            clamping_expt(n_neurons, duration, neuron_params, noise_params,
                          config_file, bias=-3., bias_shift=bias_shift, dt=.01,
                          tso_params=clamp_tso_params,
                          spike_interval=spike_interval,
                          savename='decay_test{}'.format(i),
                          clamp_offset=clamp_offset)

    # === Plot a figure =============================

    # # plot voltage and conductance traces
    # plot_vg_traces(['renewing.pkl'], 2000.)

    # # t vs p_on given synaptic weight
    filenames = ['decay_test{}.pkl'.format(i) for i in range(0, 1)]

    clamp_dict = {
        'offset': 200.,
        'spike_interval': 1.,
        'tso_params': clamp_tso_params
    }
    plot_time_evolution(filenames, clamp_dict, config_file,
                        bias=-3.)

    # # activation fct w/o external input
    # fn = 'wei_curr_calibdata.pkl'
    # plot_activation_fct(fn, config_file)
    # plot_vmem_dist(fn)

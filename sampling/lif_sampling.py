# LIF-sampling experiment with MNIST
from __future__ import division
import sys
import numpy as np
from scipy.optimize import curve_fit
from clamped_sampling import gather_network_spikes_clamped
from pprint import pformat as pf
from utils.data_mgmt import make_data_folder, load_images, load_rbm
from utils import get_windowed_image_index
from rbm import RBM, CRBM
import sbs
import matplotlib as mpl
mpl.use( "Agg" )
import matplotlib.pyplot as plt


sbs.gather_data.set_subprocess_silent(True)
log = sbs.log

# The backend of choice. Both should work but when using neuron, we need to
# disable saturating synapses for now.
sim_name = "pyNN.nest"
# sim_name = "pyNN.neuron"

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

noise_params = {
    'rate_inh' : 5000.,
    'rate_exc' : 5000.,
    'w_exc'    : .0035,
    'w_inh'    : -55./35 * .0035 # haven't understood yet how V_g is determined
}

sampling_interval = 10.  # samples are taken every tau_refrac [ms]


def vmean(v_rest, neuron_params, noise_params):
    g_l = neuron_params['cm'] / neuron_params['tau_m']
    g_inh = noise_params['w_inh'] * noise_params['rate_inh'] * \
        neuron_params['tau_syn_I']
    g_exc = noise_params['w_exc'] * noise_params['rate_exc'] * \
        neuron_params['tau_syn_E']
    return (g_l*v_rest + neuron_params['i_offset'] +
            g_inh*neuron_params['e_rev_I'] + g_exc*neuron_params['e_rev_E']) / \
        (g_l + g_inh + g_exc)


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


def vmem_dist(config_file, mean_as_x=False):
    global sample_params, noise_params
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
    plt.savefig('figures/free_vmem_dist.png')

    samples_vrest = sampler.calibration.get_samples_v_rest()
    x_data = samples_vrest
    if mean_as_x:
        x_data = vmean(samples_vrest, sample_params, noise_params)
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
    plt.savefig('figures/calibration.png')


def initialise_network(config_file, weights, biases, load=None):
    if load is not None:
        bm = sbs.network.ThoroughBM.load(load)
    else:
        # No network loaded, we need to create it. We need to specify how many
        # samplers we want and what neuron parameters they should have. Refer
        # to the documentation for all the different ways this is possible.
        if weights is None or biases is None:
            print('Please provide weights and biases.')
            return

        sampler_config = sbs.db.SamplerConfiguration.load(config_file)

        assert len(biases) == weights.shape[0] == weights.shape[1]
        bm = sbs.network.ThoroughBM(num_samplers=len(biases),
                                    sim_name=sim_name,
                                    sampler_config=sampler_config)
        bm.weights_theo = weights
        bm.biases_theo = biases
        # NOTE: By setting the theoretical weights and biases, the biological
        # ones automatically get calculated on-demand by accessing
        # bm.weights_bio and bm.biases_bio
    return bm


def sample_network(network, duration, dt=.1, burn_in=500., tso_params=None,
                   clamp_fct=None, seed=42):

    """
        How to setup and evaluate a Boltzmann machine. Please note that in
        order to instantiate BMs all needed neuron parameters need to be in the
        database and calibrated.

        Does the same thing as sbs.tools.sample_network(...).
    """
    # np.random.seed(seed)
    sim_setup_kwargs = {
        'rng_seeds_seed': seed
    }

    if tso_params is None:
        bm.saturating_synapses_enabled = False
    else:
        bm.saturating_synapses_enabled = True
        # bm.tso_params = tso_params
    bm.use_proper_tso = True

    if bm.sim_name == "pyNN.neuron":
        bm.saturating_synapses_enabled = False

    if clamp_fct is None:
        bm.gather_spikes(duration=duration, dt=dt, burn_in_time=burn_in,
                         sim_setup_kwargs=sim_setup_kwargs)
    else:
        bm.spike_data = gather_network_spikes_clamped(
            bm, duration, dt=dt, burn_in_time=500., clamp_fct=clamp_fct,
            sim_setup_kwargs=sim_setup_kwargs)

    # # saving somehow not compatible with clamping...
    # if save is not None:
    #     bm.save(save)

    if len(bm.biases_theo) < 7:
        log.info("DKL joint: {}".format(sbs.utils.dkl(
            bm.dist_joint_theo.flatten(), bm.dist_joint_sim.flatten())))

    samples = bm.get_sample_states(sampling_interval)    # == tau_refrac

    return samples


# Custom clamping methods
class Clamp_anything(object):
    # refresh times must be a list
    def __init__(self, refresh_times, clamped_idx, clamped_val):
        if len(refresh_times) == 1 and len(refresh_times) != len(clamped_idx):
            self.clamped_idx = np.expand_dims(clamped_idx, 0)
            self.clamped_val = np.expand_dims(clamped_val, 0)
        else:
            self.clamped_idx = clamped_idx
            self.clamped_val = clamped_val
        self.refresh_times = refresh_times

    def __call__(self, t):
        try:
            i = np.where(np.isclose(self.refresh_times, t))[0][0]
        except IndexError:
            print('No matching clamping time stamp; this should not happen.')
            return float('inf'), [], []

        binary_val = np.round(self.clamped_val[i])

        if i < len(self.refresh_times) - 1:
            dt = self.refresh_times[i + 1] - t
        else:
            dt = float('inf')
        return dt, self.clamped_idx[i], binary_val.astype(float)


class Clamp_window(object):
    def __init__(self, interval, clamp_img, win_size):
        self.interval = interval
        self.clamp_img = clamp_img
        self.win_size = win_size

    def __call__(self, t):
        end = min(int(t / self.interval), self.clamp_img.shape[1])
        clamped_idx = get_windowed_image_index(
            self.clamp_img.shape, end, self.win_size)
        # binarized version
        clamped_val = np.round(self.clamp_img.flatten()[clamped_idx])

        return self.interval, clamped_idx, clamped_val


def sigma_fct(x, x0, alpha):
    return 1/(1 + np.exp(-(x - x0) / alpha))

if __name__ == '__main__':
    # calibrate first if necessary
    # calibration(tutorial_params, tutorial_noise, 'Calibrations/tutorial_calib')

    # check HCS
    # vmem_dist('mihai_calib.json')

    # clamped sampling: Pong
    if len(sys.argv) != 5:
        print('Please specify the arguments:'
              ' pong/gauss, win_size, start_idx, chunk_size')
        sys.exit()

    pot_str = sys.argv[1]
    win_size = int(sys.argv[2])
    start = int(sys.argv[3])
    chunk_size = int(sys.argv[4])

    sim_dt = .01
    img_shape = (36, 48)
    n_pixels = np.prod(img_shape)

    # load stuff
    data_name = pot_str + '_var_start{}x{}'.format(*img_shape)
    _, _, test_set = load_images(data_name)
    rbm = load_rbm(data_name + '_crbm')
    w_rbm = rbm.w
    b = np.concatenate((rbm.vbias, rbm.hbias))
    nv, nh = rbm.n_visible, rbm.n_hidden

    # Bring weights and biases into right form
    w = np.concatenate((np.concatenate((np.zeros((nv, nv)), w_rbm), axis=1),
                       np.concatenate((w_rbm.T, np.zeros((nh, nh))), axis=1)),
                       axis=0)

    calib_file = 'dodo_calib.json'
    bm = initialise_network('calibrations/' + calib_file, w, b)

    # clamp sliding window
    clamp_duration = 100.
    clamp_fct = Clamp_window(clamp_duration, np.zeros(img_shape), win_size)
    duration = clamp_duration * (img_shape[1] + 1)

    # run simulations for each image in the chunk
    # store samples as bools to save disk space
    end = min(start + chunk_size, len(test_set[0]))
    samples = np.zeros((end - start, int(duration/sampling_interval), nv + nh)
                       ).astype(bool)
    save_file = pot_str + \
        '_win{}_all_chunk{:03d}'.format(win_size, start // chunk_size)

    for i, test_img in enumerate(test_set[0][start:end]):
        # this works only with ClampWindow. Fixed clamping only for testing
        clamp_fct.clamp_img = test_img.reshape(img_shape)
        samples[i] = sample_network(bm, duration, dt=sim_dt, tso_params=1,
                                    clamp_fct=clamp_fct, seed=7741092)

    save_file = 'test'
    np.savez_compressed(make_data_folder() + save_file,
                        samples=samples,
                        data_idx=np.arange(start, end),
                        win_size=win_size,
                        samples_per_frame=int(clamp_duration/sampling_interval)
                        )

    # ======== testing ========
    # # Load rbm and data
    # # minimal rbm example for debugging
    # nv = 3
    # nh = 2
    # dim = nv + nh
    # w_rbm = .5*np.random.randn(nv, nh)
    # b = np.zeros(dim)
    # save_file = 'toyrbm_samples'

    # test_img = np.ones(nv)
    # # fixed clamped image part
    # clamped_mask = np.zeros(nv + nh)
    # clamped_mask[:nv] = 1
    # clamped_idx = np.nonzero(clamped_mask == 1)[0]
    # clamped_val = test_img[clamped_idx]
    # refresh_times = np.array([0])
    # clamp_fct = clamp_anything

    # -- OR --
    # # MNIST
    # import gzip
    # img_shape = (28, 28)
    # n_pixels = np.prod(img_shape)
    # with open('../gibbs_rbm/saved_rbms/mnist_disc_rbm.pkl', 'rb') as f:
    #     rbm = cPickle.load(f)
    # f = gzip.open('../datasets/mnist.pkl.gz', 'rb')
    # _, _, test_set = np.load(f)
    # f.close()
    # save_file = 'mnist_samples'

    # # clamping
    # test_img = test_set[0][41]
    # # test_img = np.ones(n_pixels)
    # pxls_x = img_shape[1]

    # # fixed clamped image part
    # clamped_mask = np.zeros(img_shape)
    # clamped_mask[:, :pxls_x//1] = 1
    # clamped_mask = clamped_mask.flatten()
    # clamped_idx = np.nonzero(clamped_mask == 1)[0]
    # clamped_val = test_img[clamped_idx]
    # refresh_times = np.array([0])
    # clamp_fct = Clamp_anything(refresh_times, clamped_idx, clamped_val)

    # # sample with clamped labels --- only for crbms
    # refresh_times = np.arange(0, 1e3, 100.)
    # clamped_idx = [np.arange(rbm.n_visible - rbm.n_labels, rbm.n_visible)]*10
    # clamped_val = [0] * 10
    # for i, c in enumerate(clamped_val):
    #     clamped_val[i] = np.zeros(10)
    #     clamped_val[i][i] = 1
    # clamp_fct = clamp_anything

    # w_rbm = rbm.w
    # b = np.concatenate((rbm.vbias, rbm.hbias))
    # nv, nh = rbm.n_visible, rbm.n_hidden

    # # Bring weights and biases into right form
    # w = np.concatenate((np.concatenate((np.zeros((nv, nv)), w_rbm), axis=1),
    #                    np.concatenate((w_rbm.T, np.zeros((nh, nh))), axis=1)),
    #                    axis=0)

    # calib_file = 'dodo_calib.json'
    # bm = initialise_network(calib_file, w, b)

    # # fixed clamped image part
    # clamped_mask = np.zeros(img_shape)
    # clamped_mask[:, :4] = 1
    # clamped_mask = clamped_mask.flatten()
    # clamped_idx = np.nonzero(clamped_mask == 1)[0]
    # refresh_times = np.array([0])
    # clamped_val = test_img[clamped_idx]
    # clamp_fct = Clamp_anything(refresh_times, clamped_idx, clamped_val)
    # duration = 1e4

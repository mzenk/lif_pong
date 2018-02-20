#! /usr/bin/env python
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import yaml
from lif_pong.utils.data_mgmt import load_images, get_rbm_dict
from lif_pong.sampling.lif_clamped_sampling import Clamp_window
from lif_pong.neuron_simulations.stp_theory import r_theo_interpolated
import lif_pong.training.rbm as rbm_pkg
import sbs


def get_clamp_offsets(duration, clamp_fct, winsize=None, thresh=.5):
    t = 0
    flat_img = clamp_fct.clamp_img.flatten()
    exc_offsets = np.ones_like(flat_img)*duration
    inh_offsets = np.ones_like(flat_img)*duration
    last_idx = []
    while t < duration:
        delta_t, curr_idx, curr_val = clamp_fct(t)
        t_next = min(t + delta_t, duration)
        newly_clamped = np.in1d(np.arange(len(flat_img)),
                                np.setdiff1d(curr_idx, last_idx))
        exc_offsets[np.logical_and(newly_clamped, flat_img > thresh)] = t
        inh_offsets[np.logical_and(newly_clamped, flat_img < thresh)] = t
        
        last_idx = curr_idx.copy()
        t = t_next
    return exc_offsets, inh_offsets


def get_frames_for_img(img, vis_bias, n_samples, sbs_dict, clamp_dict):
    '''
        * define clamping function
        * get offsets from clamping function (for each neuron, times at which clamping starts)
        * for each sampling step t:
            rarray = r_theo_interpolated(t, offset_matrix, ...)
            weights = rarray * U * weight_matrix
            frames.append(act_fct(weights/alpha_w))
        * save frames (should have same format as sample frames from network simulations)
        * visualize with lif_inspect_samples side-by-side (maybe take other colormap)

        what about unclamped neurons?
        maybe mask weigh_matrix, so that unclamped neurons have zero weight

        what about more complex clamping patterns (i.e. units siwtch on more than once)?
        neglect for now; would have to make offset time-dependent

        how to handle excitatory/inhibitory weights?
        probably need to do above twice: offsets for exc. and inh.;
        act_fct(eweights/apha_w + iweights/alpha_w)

        empirical alternative:
        make normal experiment, but set RBM-synapses inactive (e.g. by U=0).
    '''
    assert len(img.shape) == 2
    n_pixels = img.size
    sampling_interval = sbs_dict['sampling_interval']
    clamp_dt_spike = clamp_dict['dt_spike']

    # get activation function and neuron parameters from calibration
    neuron_calib = sbs_dict['calib_file']
    sampler_config = sbs.db.SamplerConfiguration.load(neuron_calib)
    sampler = sbs.samplers.LIFsampler(sampler_config, sim_name='pyNN.nest')
    neuron_params = sampler.get_pynn_parameters()
    gl = neuron_params['cm']/neuron_params['tau_m']
    tau_syn = neuron_params['tau_syn_E']
    assert neuron_params['tau_syn_E'] == neuron_params['tau_syn_I']
    v_p05 = sampler.calibration.fit.v_p05
    alpha = sampler.calibration.fit.alpha
    # accumulation correction contained in shape parameter alpha_w
    alpha_w = alpha * gl*clamp_dt_spike/tau_syn

    # clamping stuff
    if 'winsize' in clamp_dict.keys():
        winsize = clamp_dict['winsize']
    else:
        winsize = img.shape[1]
    clamp_tso_params = clamp_dict['tso_params']
    try:
        clamp_tso_params.pop('tau_fac')
    except KeyError:
        pass

    # apply corrections to weight as in lif_clamped_sampling
    if 'weight' in clamp_tso_params.keys():
        weight = clamp_tso_params.pop('weight')/clamp_tso_params['U']
        exc_weights = weight
        inh_weights = -weight
    else:
        # width of sigma function is roughly 4
        bshift = 4
        exc_weights = alpha_w*(bshift - vis_bias)/clamp_tso_params['U']
        inh_weights = alpha_w*(-bshift - vis_bias)/clamp_tso_params['U']    
        if clamp_tso_params['U'] == 1 and clamp_tso_params['tau_rec'] == tau_syn:
            # for renewing synapses we usually want to clamp such that the
            # stationary weights yield the desired clamping. correction:
            exc_weights /= (1 - np.exp(-clamp_dt_spike/tau_syn))
            inh_weights /= (1 - np.exp(-clamp_dt_spike/tau_syn))
        print(exc_weights.max(), exc_weights.min())

    clamp_interval = sampling_interval*n_samples
    clamp_fct = Clamp_window(clamp_interval, img, win_size=winsize)

    duration = clamp_interval*(img.shape[1] + 1)
    thresh = .5
    exc_offsets, inh_offsets = get_clamp_offsets(
            duration, clamp_fct, winsize=winsize, thresh=thresh)

    p_on = np.zeros((int(duration/sampling_interval), n_pixels))
    for i, t in enumerate(np.arange(0, duration, sampling_interval)):
        delta_t, clamp_idx, clamp_val = clamp_fct(t)
        exc_mask = np.in1d(np.arange(n_pixels), clamp_idx[clamp_val > thresh])
        inh_mask = np.in1d(np.arange(n_pixels), clamp_idx[clamp_val < thresh])

        rarr_exc = r_theo_interpolated(t, exc_offsets, clamp_dt_spike,
                                       **clamp_tso_params)
        rarr_inh = r_theo_interpolated(t, inh_offsets, clamp_dt_spike,
                                       **clamp_tso_params)

        # improve if necessary to support weight matrix
        eff_weights = exc_weights*exc_mask*rarr_exc*clamp_tso_params['U'] + \
            inh_weights*inh_mask*rarr_inh*clamp_tso_params['U']
        p_on[i] = sigma(vis_bias + eff_weights/alpha_w)

    print(p_on.max())
    return p_on


def main(data_set, rbm, general_dict, sbs_dict, clamp_dict):
    clamp_dict['dt_spike'] = 1.
    n_samples = general_dict['n_samples']
    img_shape = tuple(general_dict['img_shape'])
    n_pixels = np.prod(img_shape)
    start = general_dict['start_idx']
    end = start + general_dict['chunksize']
    vis_bias = rbm.vbias[:n_pixels]
    img = data_set[0][start].reshape(img_shape)   # change later

    frames = []
    for img in data_set[0][start:end].reshape((-1,) + img_shape):
        frames.append(
            get_frames_for_img(img, vis_bias, n_samples, sbs_dict, clamp_dict))
    np.savez_compressed('test', samples=np.array(frames))


def sigma(x):
    return 1./(1 + np.exp(-x))


def inv_sigma(p):
    return np.log(p/(1 - p))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Wrong number of arguments. Please provide a yaml-config file.')
        sys.exit()
    with open(sys.argv[1]) as configfile:
        config = yaml.load(configfile)

    general_dict = config.pop('general')
    sbs_dict = config.pop('sbs')
    clamp_dict = config.pop('clamping')

    # load data
    _, _, test_set = load_images(general_dict['data_name'])
    rbm = rbm_pkg.load(get_rbm_dict(general_dict['rbm_name']))
    main(test_set, rbm, general_dict, sbs_dict, clamp_dict)

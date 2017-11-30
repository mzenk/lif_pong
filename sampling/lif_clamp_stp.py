from __future__ import division
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import sys
import numpy as np
import lif_clamped_sampling as lifsampl
from utils.data_mgmt import make_data_folder, load_images, load_rbm
from rbm import RBM, CRBM


def lif_tso_clamping_expt(test_imgs, img_shape, rbm, calib_file, sbs_kwargs,
                          clamp_kwargs, n_samples=20):
    # Bring weights and biases into right form
    w, b = rbm.bm_params()
    sampling_interval = sbs_kwargs['sampling_interval']
    # clamp using TSO
    clamp_duration = n_samples * sampling_interval
    duration = clamp_duration * (img_shape[1] + 1)

    bm = lifsampl.initialise_network(
        calib_file, w, b, tso_params=sbs_kwargs['tso_params'])

    # add all necessary kwargs to one dictionary
    kwargs = {k: sbs_kwargs[k] for k in
              ('dt', 'sim_setup_kwargs', 'burn_in_time')}
    for k in clamp_kwargs.keys():
        kwargs[k] = clamp_kwargs[k]
    results = []
    for img in test_imgs:
        kwargs['clamp_fct'] = \
            lifsampl.Clamp_window(clamp_duration, img.reshape(img_shape))
        bm.spike_data = lifsampl.gather_network_spikes_clamped_sf(
            bm, duration, rbm.n_inputs, **kwargs)
        results.append(bm.get_sample_states(sampling_interval))
    return np.array(results)


def test(test_imgs, img_shape, rbm, calib_file, sbs_kwargs,
         clamp_kwargs, n_samples=20):
    # Bring weights and biases into right form
    w, b = rbm.bm_params()
    sampling_interval = sbs_kwargs['sampling_interval']
    # clamp using TSO
    duration = n_samples * sampling_interval
    # clamp all but labels
    clamped_mask = np.ones(img_shape)
    clamped_mask = clamped_mask.flatten()
    clamped_idx = np.nonzero(clamped_mask == 1)[0]
    refresh_times = [0.]

    # sample_clamped = partial(lifsampl.sample_network_clamped,
    #                          calib_file, w, b, duration, **sbs_kwargs)
    bm = lifsampl.initialise_network(
        calib_file, w, b, tso_params=sbs_kwargs['tso_params'])

    # add all necessary kwargs to one dictionary
    kwargs = {k: sbs_kwargs[k] for k in
              ('dt', 'sim_setup_kwargs', 'burn_in_time')}
    for k in clamp_kwargs.keys():
        kwargs[k] = clamp_kwargs[k]
    results = []
    for img in test_imgs:
        kwargs['clamp_fct'] = \
            lifsampl.Clamp_anything(refresh_times, clamped_idx, img)
        # bm.spike_data = lifsampl.gather_network_spikes_clamped_bn(
        #     bm, duration, rbm.n_inputs, **kwargs)
        bm.spike_data = lifsampl.gather_network_spikes_clamped_sf(
            bm, duration, rbm.n_inputs, **kwargs)
        results.append(bm.get_sample_states(sampling_interval))
    return np.array(results)


if __name__ == '__main__':
    # clamped sampling: Pong
    if len(sys.argv) < 4:
        print('Please specify the arguments:'
              ' pong/gauss, start_idx, chunk_size, [name_modifier]')
        sys.exit()

    # parameters that can be changed via command line
    pot_str = sys.argv[1]
    start = int(sys.argv[2])
    chunk_size = int(sys.argv[3])
    end = start + chunk_size
    if len(sys.argv) == 5:
        modifier = '_' + str(sys.argv[4])
    else:
        modifier = ''
    save_file = pot_str + \
        '{}_chunk{:03d}'.format(modifier, start // chunk_size)

    # simulation parameters
    n_samples = 20
    seed = 7741092
    calib_file = 'calibrations/dodo_calib.json'

    mixing_tso_params = {
        "U": .01,
        "tau_rec": 280.,
        "tau_fac": 0.
    }

    renewing_tso_params = {
        "U": 1.,
        "tau_rec": 10.,
        "tau_fac": 0.
    }

    # parameters used for clamping with TSO-synapses
    clamp_tso_params = {
        "U": .002,
        "tau_rec": 2500.,
        "tau_fac": 0.
    }
    wp_fit_params = {}
    with np.load('../neuron_simulations/calibrations/dt1ms_calib.npz') as d:
        for k in d.keys():
            wp_fit_params[k] = d[k]
    clamp_kwargs = {
        'clamp_tso_params': clamp_tso_params,
        'wp_fit_params': wp_fit_params
    }

    sim_setup_kwargs = {
        'rng_seeds_seed': seed
    }
    sbs_kwargs = {
        'dt': .1,
        'burn_in_time': 500.,
        'sim_setup_kwargs': sim_setup_kwargs,
        'sampling_interval': 10.,   # samples are taken every tau_refrac [ms]
        "tso_params": mixing_tso_params
    }

    # load stuff
    img_shape = (36, 48)
    n_pixels = np.prod(img_shape)
    data_name = pot_str + '_var_start{}x{}'.format(*img_shape)
    _, _, test_set = load_images(data_name)
    end = min(end, len(test_set[0]))
    rbm = load_rbm(data_name + '_crbm')

    samples = lif_tso_clamping_expt(
        test_set[0][start:end], img_shape, rbm, calib_file, sbs_kwargs,
        clamp_kwargs, n_samples=n_samples)

    # # testing
    # img_shape = (2, 2)
    # rbm = CRBM(4, 5, 2)
    # test_set = (np.array(([0, 1, 0, 1], [1, 0, 0, 1]), dtype=float), 0)
    # start = 0
    # end = len(test_set[0])
    # save_file = 'test'

    # n_samples = 1000
    # samples = test(
    #     test_set[0][start:end], img_shape, rbm, calib_file, sbs_kwargs,
    #     clamp_kwargs, n_samples=n_samples)

    # import timeit
    # setup = 'from __main__ import test, test_set, start, end, img_shape, rbm, calib_file, sbs_kwargs, clamp_kwargs, n_samples'
    # print(timeit.Timer('test(test_set[0][start:end], img_shape, rbm, calib_file, sbs_kwargs, clamp_kwargs, n_samples=n_samples)',
    #                setup=setup).timeit(number=100))

    np.savez_compressed(make_data_folder() + save_file,
                        samples=samples.astype(bool),
                        data_idx=np.arange(start, end),
                        clamp_tso_params=clamp_tso_params,
                        samples_per_frame=n_samples
                        )

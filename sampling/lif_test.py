from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import lif_clamped_sampling as lifsampl
from lif_pong.utils.data_mgmt import make_data_folder, load_images, get_rbm_dict, get_data_path
from lif_pong.utils import get_windowed_image_index
import lif_pong.training.rbm as rbm_pkg


def lif_unclamp(test_imgs, rbm, calib_file, sbs_kwargs, n_samples=50):
    # Bring weights and biases into right form
    w, b = rbm.bm_params()
    sampling_interval = sbs_kwargs['sampling_interval']
    duration = n_samples * sampling_interval
    # clamp all but labels
    clamped_mask = np.ones(img_shape)
    clamped_mask = clamped_mask.flatten()
    clamped_idx = np.nonzero(clamped_mask == 1)[0]
    refresh_times = np.array([0, .25*duration])

    bm = lifsampl.initialise_network(
        calib_file, w, b, tso_params=sbs_kwargs['tso_params'])
    samples = []
    kwargs = {k: sbs_kwargs[k] for k in ('dt', 'sim_setup_kwargs',
                                         'burn_in_time')}
    for img in test_imgs:
        clamp_fct = lifsampl.Clamp_anything(
            refresh_times, [clamped_idx, []], [img, []])
        bm.spike_data = lifsampl.gather_network_spikes_clamped(
            bm, duration, clamp_fct=clamp_fct, **kwargs)
        samples.append(bm.get_sample_states(sampling_interval))
    return np.array(samples)


def lif_clamp_beginning(test_imgs, rbm, calib_file, sbs_kwargs, n_samples=20):
    # Bring weights and biases into right form
    w, b = rbm.bm_params()
    sampling_interval = sbs_kwargs['sampling_interval']
    # clamp all but labels
    clamped_mask = np.ones(img_shape)
    clamped_mask[:, int(.33 * img_shape[1]):] = 0
    clamped_idx = np.nonzero(clamped_mask.flatten())[0]
    refresh_times = np.array([0])
    duration = n_samples * sampling_interval

    bm = lifsampl.initialise_network(
        calib_file, w, b, tso_params=sbs_kwargs['tso_params'])
    kwargs = {k: sbs_kwargs[k] for k in ('dt', 'sim_setup_kwargs',
                                         'burn_in_time')}
    results = []
    for img in test_imgs:
        # clamp_fct = lifsampl.Clamp_anything(
        #     refresh_times, clamped_idx, img[clamped_idx])
        clamp_fct = None
        bm.spike_data = lifsampl.gather_network_spikes_clamped(
            bm, duration, clamp_fct=clamp_fct, **kwargs)
        results.append(bm.get_sample_states(sampling_interval))
    return np.array(results)


def test_clamping(calib_file, sbs_kwargs, n_samples=20):
    # minimal rbm example for debugging
    nv = 4
    nh = 2
    dim = nv + nh
    np.random.seed(42)
    w_rbm = .5*np.random.randn(nv, nh) * 0
    b = np.zeros(dim)
    w = np.concatenate((np.concatenate((np.zeros((nv, nv)), w_rbm), axis=1),
                        np.concatenate((w_rbm.T, np.zeros((nh, nh))), axis=1)),
                       axis=0)

    sampling_interval = sbs_kwargs['sampling_interval']

    duration = n_samples * sampling_interval
    # fixed clamped image part
    clamped_mask = np.ones(nv)
    clamped_idx = np.nonzero(clamped_mask.flatten())[0]
    refresh_times = np.array([0])
    clamped_val = np.linspace(0, 1, 100)

    bm = lifsampl.initialise_network(
        calib_file, w, b, tso_params='renewing')
    kwargs = {k: sbs_kwargs[k] for k in ('dt', 'sim_setup_kwargs',
                                         'burn_in_time')}
    p_on = []
    for v in clamped_val:
        clamp_fct = lifsampl.Clamp_anything(
            refresh_times, clamped_idx, np.array([0, 0, 0, v]))
        bm.spike_data = lifsampl.gather_network_spikes_clamped(
            bm, duration, clamp_fct=clamp_fct, **kwargs)
        samples = bm.get_sample_states(sampling_interval)
        p_on.append(np.mean(samples[:, 3]))
    return clamped_val, p_on


if __name__ == '__main__':
    # pot_str = 'pong'
    # start = 0
    # end = 2

    # n_samples = 500
    # seed = 7741092
    # calib_file = 'calibrations/dodo_calib.json'
    # mixing_tso_params = {
    #     "U": .01,
    #     "tau_rec": 280.,
    #     "tau_fac": 0.
    # }

    # sbs_kwargs = {
    #     'dt': .1,
    #     'burn_in_time': 300.,
    #     'sim_setup_kwargs': {'rng_seeds_seed': seed},
    #     'sampling_interval': 10.,
    #     "tso_params": mixing_tso_params
    # }

    # img_shape = (36, 48)
    # data_name = pot_str + '_var_start{}x{}'.format(*img_shape)
    # _, _, test_set = load_images(data_name)
    # rbm = rbm_pkg.load(get_rbm_dict(data_name + '_crbm'))

    # save_file = 'test'
    # samples = lif_clamp_beginning(
    #     test_set[0][start:end], rbm, calib_file, sbs_kwargs,
    #     n_samples=n_samples)

    # np.savez_compressed(make_data_folder() + save_file,
    #                     samples=samples.astype(bool),
    #                     data_idx=np.arange(start, end))

    # plt.figure()
    # pxl_value, p_on = test_clamping(calib_file, sbs_kwargs, n_samples)
    # plt.plot(pxl_value, p_on)
    # plt.plot(pxl_value, pxl_value, 'g--')
    # plt.savefig('test.png')

    # compute isl
    from training import compute_isl as isl
    # # Pong
    # img_shape = (36, 48)
    # n_pixels = np.prod(img_shape)
    # n_labels = img_shape[0] // 3
    # data_name = 'pong_var_start{}x{}'.format(*img_shape)
    # _, _, test_set = load_images(data_name)
    # MNIST
    import gzip
    img_shape = (28, 28)
    n_pixels = np.prod(img_shape)
    with gzip.open('../shared_data/datasets/mnist.pkl.gz', 'rb') as f:
        _, _, test_set = np.load(f)

    test_vis = test_set[0][:2000, :n_pixels]
    sample_file = get_data_path('lif_dreaming') + \
        'pong_samples.npz'

    with np.load(sample_file) as d:
        # samples.shape: ([n_instances], n_samples, n_units)
        samples = d['samples'].astype(float).squeeze()
        vis_samples = samples[:, :n_pixels]
        n_samples = samples.shape[0]
        print('Loaded sample array with shape {}'.format(samples.shape))

        # compute isl
        dm = isl.ISL_density_model()
        dm.fit(vis_samples, quick=True)
        print(dm.avg_loglik(test_vis))
        # ...

from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import lif_clamped_sampling as lifsampl
from utils.data_mgmt import make_data_folder, load_images, load_rbm
from rbm import RBM, CRBM
import multiprocessing as mp
from functools import partial


def lif_classify(test_imgs, rbm, calib_file, sbs_kwargs, n_samples=50):
    # Bring weights and biases into right form
    w, b = rbm.bm_params()
    sampling_interval = sbs_kwargs['sampling_interval']
    # clamp all but labels
    clamped_mask = np.ones(img_shape)
    clamped_mask = clamped_mask.flatten()
    clamped_idx = np.nonzero(clamped_mask == 1)[0]
    refresh_times = np.array([0])

    duration = n_samples * sampling_interval

    pool = mp.Pool(processes=8)
    sample_clamped = partial(lifsampl.sample_network_clamped,
                             calib_file, w, b, duration, **sbs_kwargs)
    results = []
    for img in test_imgs:
        clamp_fct = lifsampl.Clamp_anything(refresh_times, clamped_idx, img)
        results.append(pool.apply_async(
            sample_clamped, kwds={'clamp_fct': clamp_fct}))
    pool.close()
    pool.join()
    samples = np.array([r.get() for r in results])

    lab_samples = samples[..., rbm.n_visible - rbm.n_labels:rbm.n_visible]
    # return mean activities of label layer
    return lab_samples.sum(axis=1), samples


def lif_classify_serial(test_imgs, rbm, calib_file, sbs_kwargs, n_samples=50):
    # Bring weights and biases into right form
    w, b = rbm.bm_params()
    sampling_interval = sbs_kwargs['sampling_interval']
    # clamp all but labels
    clamped_mask = np.ones(img_shape)
    clamped_mask = clamped_mask.flatten()
    clamped_idx = np.nonzero(clamped_mask == 1)[0]
    refresh_times = np.array([0])

    duration = n_samples * sampling_interval
    bm = lifsampl.initialise_network(
        calib_file, w, b, tso_params=sbs_kwargs['tso_params'])
    samples = []
    kwargs = {k: sbs_kwargs[k] for k in ('dt', 'sim_setup_kwargs',
                                         'burn_in_time')}
    for img in test_imgs:
        clamp_fct = lifsampl.Clamp_anything(refresh_times, clamped_idx, img)
        bm.spike_data = lifsampl.gather_network_spikes_clamped(
            bm, duration, clamp_fct=clamp_fct, **kwargs)
        samples.append(bm.get_sample_states(sampling_interval))
    samples = np.array(samples)
    lab_samples = samples[..., rbm.n_visible - rbm.n_labels:rbm.n_visible]
    # return mean activities of label layer
    return lab_samples.sum(axis=1), samples

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Please specify the arguments: '
              'pong/gauss/mnist, start_idx, chunk_size')
        sys.exit()

    pot_str = sys.argv[1]
    start = int(sys.argv[2])
    chunk_size = int(sys.argv[3])
    end = start + chunk_size
    # simulation parameters
    n_samples = 100
    seed = 7741092
    save_file = pot_str + '_classif_{}samples'.format(n_samples)
    calib_file = 'dodo_calib.json'
    mixing_tso_params = {
        "U": .01,
        "tau_rec": 280.,
        "tau_fac": 0.
    }

    sim_setup_kwargs = {
        'rng_seeds_seed': seed
    }
    sbs_kwargs = {
        'dt': .1,
        'burn_in_time': 300.,
        'sim_setup_kwargs': sim_setup_kwargs,
        'sampling_interval': 10.,
        "tso_params": 'renewing'
    }

    # load stuff
    if pot_str == 'mnist':
        import gzip
        img_shape = (28, 28)
        with gzip.open('../shared_data/datasets/mnist.pkl.gz', 'rb') as f:
            _, _, test_set = np.load(f)
        rbm = load_rbm('mnist_disc_rbm')
        test_targets = test_set[1][start:min(end, len(test_set[0]))]
    else:
        img_shape = (36, 48)
        data_name = pot_str + '_var_start{}x{}'.format(*img_shape)
        _, _, test_set = load_images(data_name)
        rbm = load_rbm(data_name + '_crbm')
        test_targets = np.argmax(test_set[1][start:min(end, len(test_set[0]))],
                                 axis=1)

    n_pixels = np.prod(img_shape)

    label_mean, samples = lif_classify_serial(
        test_set[0][start:end], rbm, 'calibrations/' + calib_file, sbs_kwargs,
        n_samples)

    np.savez_compressed(make_data_folder() + save_file,
                        samples=samples.astype(bool),
                        data_idx=np.arange(start, end),
                        n_samples=n_samples
                        )
    labels = np.argmax(label_mean, axis=1)
    print('Correct predictions: {}'.format((labels == test_targets).mean()))

    # # I would like to use the reset mechanism so that I don't have to setup the
    # # network over and over again but somehow it doesn't work (clamping is not changed)
    # # lifsampl.setup_simulation(sim_dt, sim_setup_kwargs)
    # # # connect pyNN neurons
    # # lifsampl.make_network_connections(bm, duration, burn_in_time=burn_in)
    # # lifsampl.simulate_network(bm, duration, dt=sim_dt, reset=(i != 0),
    # #                               burn_in_time=burn_in_time, clamp_fct=clamp_fct)
    # #     samples[i] = bm.get_sample_states(sampling_interval)
    # # lifsampl.end_simulation()

    # np.savez_compressed(make_data_folder() + save_file,
    #                     samples=samples,
    #                     data_idx=np.arange(start, end),
    #                     n_samples=n_samples
    #                     )

    # # compute classification rate
    # labels = np.argmax(samples.sum(axis=1), axis=1)
    # print('Correct predictions: {}'.format((labels == test_targets).mean()))

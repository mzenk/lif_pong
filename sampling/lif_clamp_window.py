from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import lif_clamped_sampling as lifsampl
from lif_pong.utils.data_mgmt import make_data_folder, load_images, get_rbm_dict
import lif_pong.training.rbm as rbm_pkg


def lif_window_expt(win_size, test_imgs, img_shape, rbm, calib_file,
                    sbs_kwargs, n_samples=20):
    # Bring weights and biases into right form
    w, b = rbm.bm_params()
    sampling_interval = sbs_kwargs['sampling_interval']
    # clamp sliding window
    clamp_duration = n_samples * sampling_interval
    duration = clamp_duration * (img_shape[1] + 1)

    # sample_clamped = partial(lifsampl.sample_network_clamped,
    #                          calib_file, w, b, duration, **sbs_kwargs)
    bm = lifsampl.initialise_network(
        calib_file, w, b, tso_params=sbs_kwargs['tso_params'])
    kwargs = {k: sbs_kwargs[k] for k in ('dt', 'sim_setup_kwargs',
                                         'burn_in_time')}
    results = []
    for img in test_imgs:
        clamp_fct = lifsampl.Clamp_window(
            clamp_duration, img.reshape(img_shape), win_size)
        bm.spike_data = lifsampl.gather_network_spikes_clamped(
            bm, duration, clamp_fct=clamp_fct, **kwargs)
        results.append(bm.get_sample_states(sampling_interval))
    return np.array(results)


def test(test_imgs, img_shape, rbm, calib_file, sbs_kwargs,
         n_samples=20):
    # Bring weights and biases into right form
    w, b = rbm.bm_params()
    sampling_interval = sbs_kwargs['sampling_interval']
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
    kwargs = {k: sbs_kwargs[k] for k in ('dt', 'sim_setup_kwargs',
                                         'burn_in_time')}
    results = []
    for img in test_imgs:
        clamp_fct = lifsampl.Clamp_anything(refresh_times, clamped_idx, img)
        bm.spike_data = lifsampl.gather_network_spikes_clamped(
            bm, duration, clamp_fct=clamp_fct, **kwargs)
        results.append(bm.get_sample_states(sampling_interval))
    return np.array(results)


# clamped sampling: Pong
if len(sys.argv) < 5:
    print('Please specify the arguments:'
          ' pong/gauss, start_idx, chunk_size, win_size, [name_modifier]')
    sys.exit()

# oarameters that can be changed via command line
pot_str = sys.argv[1]
start = int(sys.argv[2])
chunk_size = int(sys.argv[3])
win_size = int(sys.argv[4])
end = start + chunk_size
if len(sys.argv) == 6:
    modifier = '_' + str(sys.argv[5])
else:
    modifier = ''
save_file = pot_str + \
    '_win{}{}_chunk{:03d}'.format(win_size, modifier, start // chunk_size)

# simulation parameters
n_samples = 20
seed = 7741092
calib_file = 'calibrations/dodo_calib.json'

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
rbm = rbm_pkg.load(get_rbm_dict(data_name + '_crbm'))

samples = lif_window_expt(
    win_size, test_set[0][start:end], img_shape, rbm, calib_file, sbs_kwargs,
    n_samples=n_samples)

# # testing
# img_shape = (2, 2)
# rbm = rbm_pkg.CRBM(4, 5, 2)
# test_set = (np.array(([0, 1, 0, 1], [1, 0, 0, 1]), dtype=float), 0)
# start = 0
# end = len(test_set[0])
# n_samples = 100
# save_file = 'test'

# # time measurement
# import timeit
# setup = 'from __main__ import test, test_set, start, end, img_shape, rbm, calib_file, sbs_kwargs, n_samples'
# print(timeit.Timer('test(test_set[0][start:end], img_shape, rbm, calib_file, sbs_kwargs, n_samples=n_samples)',
#                    setup=setup).timeit(number=100))

# samples = test(
#     test_set[0][start:end], img_shape, rbm, calib_file, sbs_kwargs,
#     n_samples=n_samples)

np.savez_compressed(make_data_folder() + save_file,
                    samples=samples.astype(bool),
                    data_idx=np.arange(start, end),
                    win_size=win_size,
                    samples_per_frame=n_samples
                    )

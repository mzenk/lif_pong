from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import lif_clamped_sampling as lifsampl
from utils.data_mgmt import make_data_folder, load_images, load_rbm
from rbm import RBM, CRBM
import multiprocessing as mp
from functools import partial


def lif_window_expt(win_size, test_imgs, rbm, calib_file, sbs_kwargs,
                    n_samples=20):
    # Bring weights and biases into right form
    w, b = rbm.bm_params()
    sampling_interval = sbs_kwargs['sampling_interval']
    # clamp sliding window
    clamp_duration = n_samples * sampling_interval
    duration = clamp_duration * (img_shape[1] + 1)

    pool = mp.Pool(processes=8)
    sample_clamped = partial(lifsampl.sample_network_clamped,
                             calib_file, w, b, duration, **sbs_kwargs)
    results = []
    for img in test_imgs:
        clamp_fct = lifsampl.Clamp_window(
            clamp_duration, img.reshape(img_shape), win_size)
        results.append(pool.apply_async(
            sample_clamped, kwds={'clamp_fct': clamp_fct}))
    pool.close()
    pool.join()
    samples = np.array([r.get() for r in results])
    return samples


def lif_window_expt_serial(win_size, test_imgs, rbm, calib_file, sbs_kwargs,
                           n_samples=20):
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
    tso_clamp = {
        "U": 1.,
        "tau_rec": 10.,
        "tau_fac": 0.
    }
    results = []
    for img in test_imgs:
        clamp_fct = lifsampl.Clamp_window(
            clamp_duration, img.reshape(img_shape), win_size)
        # results.append(sample_clamped(clamp_fct=clamp_fct))
        bm.spike_data = lifsampl.gather_network_spikes_clamped(
            bm, duration, clamp_fct=clamp_fct, **kwargs)
        # bm.spike_data = lifsampl.gather_network_spikes_clamped_bn(
        #     bm, duration, rbm.n_inputs, clamp_fct=clamp_fct,
        #     clamp_tso_params=tso_clamp, **kwargs)
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
rbm = load_rbm(data_name + '_crbm_post')

samples = lif_window_expt_serial(
    win_size, test_set[0][start:end], rbm, calib_file, sbs_kwargs,
    n_samples=n_samples)

np.savez_compressed(make_data_folder() + save_file,
                    samples=samples.astype(bool),
                    data_idx=np.arange(start, end),
                    win_size=win_size,
                    samples_per_frame=n_samples
                    )

# # ================= older version ================
# # Bring weights and biases into right form
# sampling_interval = sbs_kwargs['sampling_interval']
# burn_in_time = sbs_kwargs['burn_in_time']
# sim_dt = sbs_kwargs['dt']
# nv, nh = rbm.n_visible, rbm.n_hidden
# w, b = rbm.bm_params()
# bm = lifsampl.initialise_network(calib_file, w, b,
#                                  tso_params=mixing_tso_params)

# # clamp sliding window
# clamp_duration = n_samples * sampling_interval
# clamp_fct = lifsampl.Clamp_window(clamp_duration, np.zeros(img_shape),
#                                   win_size)
# duration = clamp_duration * (img_shape[1] + 1)

# # ====== actual simulation ======
# # setup simulation with seed
# sim_setup_kwargs = {
#     'rng_seeds_seed': seed
# }

# # run simulations for each image in the chunk
# # store samples as bools to save disk space
# end = min(start + chunk_size, len(test_set[0]))
# samples = np.zeros((end - start, int(duration/sampling_interval), nv + nh)
#                    ).astype(bool)
# for i, test_img in enumerate(test_set[0][start:end]):
#     clamp_fct.clamp_img = test_img.reshape(img_shape)
#     bm.spike_data = lifsampl.gather_network_spikes_clamped(
#         bm, duration, dt=sim_dt, burn_in_time=burn_in_time,
#         sim_setup_kwargs=sim_setup_kwargs, clamp_fct=clamp_fct)
#     samples[i] = bm.get_sample_states(sampling_interval)

# np.savez_compressed(make_data_folder() + save_file,
#                     samples=samples,
#                     data_idx=np.arange(start, end),
#                     win_size=win_size,
#                     samples_per_frame=int(clamp_duration/sampling_interval)
#                     )

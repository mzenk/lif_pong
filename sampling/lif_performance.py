from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import lif_clamped_sampling as lifsampl
from lif_pong.utils.data_mgmt import make_data_folder, load_images, get_rbm_dict
from lif_pong.utils import get_windowed_image_index
import lif_pong.training.rbm as rbm_pkg
import multiprocessing as mp
from functools import partial


def lif_window_quick(test_imgs, rbm, calib_file, sbs_kwargs, winsize=-1,
                     n_samples=20):
    # Bring weights and biases into right form
    w, b = rbm.bm_params()
    sampling_interval = sbs_kwargs['sampling_interval']
    # # clamp only a few window positions, more in the first half of the field
    # winpos = range(2, img_shape[1]//2, 2) + \
    #     range(img_shape[1]//2, img_shape[1], 4)
    # alternatively, very small region
    winpos = (np.linspace(.2, .3, 5) * img_shape[1]).astype(int)
    refresh_times = sampling_interval*n_samples * np.arange(len(winpos))
    clamped_idx = [get_windowed_image_index(img_shape, p) for p in winpos]
    duration = len(refresh_times) * sampling_interval*n_samples

    bm = lifsampl.initialise_network(
        calib_file, w, b, tso_params=sbs_kwargs['tso_params'])
    kwargs = {k: sbs_kwargs[k] for k in ('dt', 'sim_setup_kwargs',
                                         'burn_in_time')}
    results = []
    for img in test_imgs:
        clamped_val = [img[idx] for idx in clamped_idx]
        clamp_fct = lifsampl.Clamp_anything(
            refresh_times, clamped_idx, clamped_val)
        bm.spike_data = lifsampl.gather_network_spikes_clamped(
            bm, duration, clamp_fct=clamp_fct, **kwargs)
        results.append(bm.get_sample_states(sampling_interval))
    return np.array(results), np.array(winpos)


if __name__ == '__main__':
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
        'sampling_interval': 10.,
        "tso_params": mixing_tso_params
    }

    img_shape = (36, 48)
    data_name = pot_str + '_var_start{}x{}'.format(*img_shape)
    _, _, test_set = load_images(data_name)
    rbm = rbm_pkg.load(get_rbm_dict(data_name + '_crbm'))

    samples, winpos = lif_window_quick(
        test_set[0][start:end], rbm, calib_file, sbs_kwargs,
        n_samples=n_samples)

    np.savez_compressed(make_data_folder() + save_file,
                        samples=samples.astype(bool),
                        data_idx=np.arange(start, end),
                        win_size=win_size,
                        winpos=winpos,
                        samples_per_frame=n_samples)

# script for analyzing sampled data
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from training.rbm import RBM, CRBM
from utils.data_mgmt import make_figure_folder, load_images, load_rbm, get_data_path, make_data_folder


# # =============================
# # Investigate clamping problems

# # MNIST
# img_shape = (28, 28)
# import gzip
# f = gzip.open('../shared_data/datasets/mnist.pkl.gz', 'rb')
# _, _, test_set = np.load(f)
# f.close()
# rbm = load_rbm('mnist_disc_rbm')
# nv = rbm.n_visible
# nl = rbm.n_labels
# sample_file = get_data_path('lif_classification') + \
#     'mnist_classif_500samples.npz'

# n_pixels = np.prod(img_shape)


# with np.load(sample_file) as d:
#     # samples.shape: ([n_instances], n_samples, n_units)
#     samples = d['samples'].astype(float)
#     if len(samples.shape) == 3:
#         # take only one of the instances
#         samples = samples[0]
#     print('Loaded sample array with shape {}'.format(samples.shape))

# vis_samples = samples[..., :n_pixels]
# hid_samples = samples[..., nv:]
# sample_imgs = vis_samples.reshape(vis_samples.shape[0], *img_shape)

# test_img = np.ones(img_shape)
# counter = 0
# for i, s in enumerate(sample_imgs[1:]):
#     if not np.all(np.isclose(s, test_img)):
#         # if counter > 20:
#         #     print('More than 20 wrong...')
#         #     break
#         # plt.figure()
#         # plt.imshow(s - test_img, interpolation='Nearest', cmap='gray')
#         # plt.colorbar()
#         # plt.savefig(make_figure_folder() + 'wrong' + str(i) + '.png')
#         # plt.close()
#         counter += 1
# print('{} of {} samples were clamped incorrectly'.format(
#     counter, len(sample_imgs)))

# ============================
# Bias neurons + sbs
import lif_clamped_sampling as lifsampl


def test_classification(test_imgs, rbm, calib_file, sbs_kwargs, n_samples=50):
    # Bring weights and biases into right form
    w, b = rbm.bm_params()
    sampling_interval = sbs_kwargs['sampling_interval']
    # clamp all but labels
    clamped_mask = np.ones(img_shape)
    clamped_mask = clamped_mask.flatten()
    clamped_idx = np.nonzero(clamped_mask == 1)[0]
    refresh_times = [0.]

    duration = n_samples * sampling_interval
    bm = lifsampl.initialise_network(
        calib_file, w, b, tso_params=sbs_kwargs['tso_params'])
    samples = []
    kwargs = {k: sbs_kwargs[k] for k in ('dt', 'sim_setup_kwargs',
                                         'burn_in_time')}
    tso_clamp = {
        "U": 1.,
        "tau_rec": 10.,
        "tau_fac": 0.
    }
    for img in test_imgs:
        clamp_fct = lifsampl.Clamp_anything(refresh_times, clamped_idx, img)
        bm.spike_data = lifsampl.gather_network_spikes_clamped_bn(
            bm, duration, rbm.n_inputs, clamp_fct=clamp_fct,
            clamp_tso_params=tso_clamp, **kwargs)
        # bm.spike_data = lifsampl.gather_network_spikes_clamped(
        #     bm, duration, clamp_fct=clamp_fct, **kwargs)
        samples.append(bm.get_sample_states(sampling_interval))
    samples = np.array(samples)
    lab_samples = samples[..., rbm.n_visible - rbm.n_labels:rbm.n_visible]
    # return mean activities of label layer
    return lab_samples.sum(axis=1), samples

# simulation parameters
n_samples = 300
seed = 7741092
save_file = 'test_bias_neurons'
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
    'burn_in_time': 300.,
    'sim_setup_kwargs': sim_setup_kwargs,
    'sampling_interval': 10.,
    "tso_params": 'renewing'
}

# load stuff
import gzip
img_shape = (28, 28)
n_pixels = np.prod(img_shape)
with gzip.open('../shared_data/datasets/mnist.pkl.gz', 'rb') as f:
    _, _, test_set = np.load(f)
rbm = load_rbm('mnist_disc_rbm')
start = 0
end = 2
test_targets = test_set[1][start:end]

img_shape = (2, 2)
rbm = CRBM(4, 5, 2)
test_set = (np.array(([0, 1, 0, 1], [1, 0, 0, 1]), dtype=float), 0)

label_mean, samples = test_classification(
    test_set[0][start:end], rbm, calib_file, sbs_kwargs, n_samples)

np.savez_compressed(make_data_folder() + save_file,
                    samples=samples.astype(bool),
                    data_idx=0,
                    n_samples=n_samples
                    )
labels = np.argmax(label_mean, axis=1)
print('Correct predictions: {}'.format((labels == test_targets).mean()))

from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import lif_clamped_sampling as lifsampl
from lif_pong.utils.data_mgmt import make_data_folder, load_images, load_rbm
from lif_pong.training.rbm import RBM, CRBM

# ==========
# das hier kommt in lif_clamped_sampling.py

def setup_and_create(
        network, duration, dt=0.1, burn_in_time=0., create_kwargs=None,
        sim_setup_kwargs=None, initial_vmem=None):
    if sim_setup_kwargs is None:
        sim_setup_kwargs = {}

    sim.setup(timestep=dt, **sim_setup_kwargs)

    if create_kwargs is None:
        create_kwargs = {}
    network.create(duration=duration + burn_in_time, **create_kwargs)

    network.population.record("spikes")
    if initial_vmem is not None:
        network.population.initialize(v=initial_vmem)


def run_simulation(network, duration, dt=0.1, burn_in_time=0., clamp_fct=None):
    """
        clamp_fct: method handle that returns clamped indices and values;
        is called by the ClampCallback object, which then adjusts the biases
        accordingly
    """
    sim.reset()
    population = network.population
    log.info("Gathering spike data...")

    callbacks = get_callbacks(sim, {
            "duration": duration,
            "offset": burn_in_time,
            })

    t_start = time.time()
    if burn_in_time > 0.:
        log.info("Burning in samplers for {} ms".format(burn_in_time))
        sim.run(burn_in_time)
        eta_from_burnin(t_start, burn_in_time, duration)

    # add clamping functionality
    callbacks.append(
        ClampCallback(network, clamp_fct, duration, offset=burn_in_time))
    log.info("Starting data gathering run.")
    sim.run(duration, callbacks=callbacks)

    if isinstance(population, sim.common.BasePopulation):
        spiketrains = population.get_data("spikes").segments[0].spiketrains
    else:
        spiketrains = np.vstack(
                [pop.get_data("spikes").segments[0].spiketrains[0]
                 for pop in population])

    # we need to ignore the burn in time
    clean_spiketrains = []
    for st in spiketrains:
        clean_spiketrains.append(np.array(st[st > burn_in_time])-burn_in_time)

    return_data = {
            "spiketrains": clean_spiketrains,
            "duration": duration,
            "dt": dt,
            }

    return return_data


def end():
    sim.end()

# ==========


def lif_classify_fast(test_imgs, rbm, calib_file, sbs_kwargs, n_samples=50):
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
    kwargs = {k: sbs_kwargs[k] for k in ('dt', 'sim_setup_kwargs',
                                         'burn_in_time')}
    lifsampl.setup_and_create(bm, duration, **kwargs)

    samples = []
    del kwargs['sim_setup_kwargs']
    for img in test_imgs:
        clamp_fct = lifsampl.Clamp_anything(refresh_times, clamped_idx, img)
        bm.spike_data = lifsampl.run_simulation(
            bm, duration, clamp_fct=clamp_fct, **kwargs)
        samples.append(bm.get_sample_states(sampling_interval))
    lifsampl.end()
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
    n_samples = 10
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
        img_shape = (18, 24)
        data_name = pot_str + '_fixed_start{}x{}'.format(*img_shape)
        _, _, test_set = load_images(data_name)
        rbm = load_rbm(data_name + '_crbm')
        test_targets = np.argmax(test_set[1][start:min(end, len(test_set[0]))],
                                 axis=1)

    n_pixels = np.prod(img_shape)

    label_mean, samples = lif_classify_fast(
        test_set[0][start:end], rbm, 'calibrations/' + calib_file, sbs_kwargs,
        n_samples)

    np.savez_compressed(make_data_folder() + save_file,
                        samples=samples.astype(bool),
                        data_idx=np.arange(start, end),
                        n_samples=n_samples
                        )
    labels = np.argmax(label_mean, axis=1)
    print('Correct predictions: {}'.format((labels == test_targets).mean()))
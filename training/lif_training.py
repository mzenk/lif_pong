from __future__ import division
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import numpy as np
import multiprocessing as mp
import logging
import sampling.lif_clamped_sampling as lifsampl
from utils.data_mgmt import make_data_folder, load_images, load_rbm
from rbm import RBM, CRBM
from functools import partial
from utils import bin_to_dec, compute_dkl
import matplotlib.pyplot as plt
import time
import cPickle

# This calibration file is used for all simulations (global variable)
calib_file = '../sampling/calibrations/dodo_calib.json'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.FileHandler('training.log')
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s : %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)


# This method computes the gradient of a minibatch.
def compute_grad(rbm, batch, n_steps, n_vis, n_hid_samples=10,
                 sbs_kwargs=None):
        if sbs_kwargs is None:
            sbs_kwargs = {
                'dt': .1,
                'burn_in_time': 500.,
                'sim_setup_kwargs': None,
                'sampling_interval': 10.
            }
        sampling_interval = sbs_kwargs['sampling_interval']
        w, b = rbm.bm_params()

        # data must be binary for LIF networks
        # bin_batch = (np.random.rand(*batch.shape) < batch)*1.
        bin_batch = np.round(batch)

        # obtain samples for data average
        # clamp all but labels
        clamped_idx = np.arange(n_vis)
        refresh_times = np.array([0])
        clamped_val = np.random.rand(len(clamped_idx))
        clamp_fct = lifsampl.Clamp_anything(refresh_times, clamped_idx,
                                            clamped_val)

        # run simulations for each image in the batch
        # average hidden samples before computing corelation
        pos_samples = np.zeros(
            (len(batch), n_hid_samples, rbm.n_visible + rbm.n_hidden))
        duration = n_hid_samples * sampling_interval

        pool = mp.Pool(processes=len(batch))
        sample_clamped = partial(lifsampl.sample_network_clamped,
                                 calib_file, w, b, duration, **sbs_kwargs)
        logger.info('Collecting positive samples')
        results = []
        for img in bin_batch:
            clamp_fct.set_clamped_val(img)
            results.append(pool.apply_async(
                sample_clamped, kwds={'clamp_fct': clamp_fct}))
        pool.close()
        pool.join()
        pos_samples = np.array([r.get() for r in results])

        pos_hid_samples = pos_samples[..., n_vis:]
        pos_hid_probs = pos_hid_samples.mean(axis=1)

        # obtain samples for model average
        logger.info('Collecting negative samples')
        duration = n_steps * sampling_interval
        neg_samples = lifsampl.sample_network(
            calib_file, w, b, duration, **sbs_kwargs)
        neg_vis_samples = neg_samples[:, :n_vis]
        neg_hid_samples = neg_samples[:, n_vis:]
        assert len(neg_vis_samples) == n_steps

        # Compute gradients
        pos_grad_w = bin_batch.T.dot(pos_hid_probs) / len(batch)
        neg_grad_w = neg_vis_samples.T.dot(neg_hid_samples) / n_steps
        grad_w = pos_grad_w - neg_grad_w
        grad_bh = np.mean(pos_hid_probs, axis=0) - \
            np.mean(neg_hid_samples, axis=0)
        grad_bv = np.mean(bin_batch, axis=0) - np.mean(neg_vis_samples, axis=0)

        return (grad_w, grad_bv, grad_bh)


# This method computes the gradient of a minibatch.
def compute_grad_cdn(rbm, batch, n_steps, n_vis, n_hid_samples=10,
                     sbs_kwargs=None):
        if sbs_kwargs is None:
            sbs_kwargs = {
                'dt': .1,
                'burn_in_time': 500.,
                'sim_setup_kwargs': None,
                'sampling_interval': 10.
            }
        sampling_interval = sbs_kwargs['sampling_interval']
        w, b = rbm.bm_params()

        # data must be binary for LIF networks
        # bin_batch = (np.random.rand(*batch.shape) < batch)*1.
        bin_batch = np.round(batch)

        pos_duration = n_hid_samples * sampling_interval
        neg_duration = n_steps * sampling_interval
        refresh_times = np.array([0., pos_duration])
        clamped_idx = [np.arange(n_vis), []]
        clamped_val = [np.random.rand(len(clamped_idx)), []]
        clamp_fct = lifsampl.Clamp_anything(refresh_times, clamped_idx,
                                            clamped_val)

        # run simulations for each image in the batch
        # average hidden samples before computing corelation
        samples = np.zeros((len(batch), n_hid_samples + n_steps,
                            rbm.n_visible + rbm.n_hidden))
        logger.info('Collecting samples')
        pool = mp.Pool(processes=len(batch))
        sample_clamped = partial(lifsampl.sample_network_clamped,
                                 calib_file, w, b, pos_duration + neg_duration,
                                 **sbs_kwargs)
        results = []
        for img in bin_batch:
            clamp_fct.set_clamped_val([img, []])
            results.append(pool.apply_async(
                sample_clamped, kwds={'clamp_fct': clamp_fct}))
        pool.close()
        pool.join()
        samples = np.array([r.get() for r in results])

        pos_hid_samples = samples[:, :n_hid_samples, n_vis:]
        pos_hid_probs = pos_hid_samples.mean(axis=1)
        neg_vis_samples = samples[:, -1, :n_vis]
        neg_hid_samples = samples[:, -1, n_vis:]

        # Compute gradients
        pos_grad_w = bin_batch.T.dot(pos_hid_probs) / len(batch)
        neg_grad_w = neg_vis_samples.T.dot(neg_hid_samples) / n_steps
        grad_w = pos_grad_w - neg_grad_w
        grad_bh = np.mean(pos_hid_probs, axis=0) - \
            np.mean(neg_hid_samples, axis=0)
        grad_bv = np.mean(bin_batch, axis=0) - np.mean(neg_vis_samples, axis=0)

        return (grad_w, grad_bv, grad_bh)


# This method computes the gradient of a minibatch.
def compute_grad_serial(rbm, batch, n_steps, n_vis, n_hid_samples=10,
                        sbs_kwargs=None):
        if sbs_kwargs is None:
            sbs_kwargs = {
                'dt': .1,
                'burn_in_time': 500.,
                'sim_setup_kwargs': None,
                'sampling_interval': 10.
            }
        sampling_interval = sbs_kwargs['sampling_interval']
        w, b = rbm.bm_params()

        # data must be binary for LIF networks
        # bin_batch = (np.random.rand(*batch.shape) < batch)*1.
        bin_batch = np.round(batch)
        # ---> now, clamped values need not be binary any more. Try out!

        # obtain samples for data average
        # clamp all but labels
        clamped_idx = np.arange(n_vis)
        refresh_times = np.array([0])
        clamped_val = np.random.rand(len(clamped_idx))
        clamp_fct = lifsampl.Clamp_anything(refresh_times, clamped_idx,
                                            clamped_val)

        # run simulations for each image in the batch
        # average hidden samples before computing corelation
        pos_samples = np.zeros(
            (len(batch), n_hid_samples, rbm.n_visible + rbm.n_hidden))
        duration = n_hid_samples * sampling_interval

        logger.info('Collecting positive samples')
        for i, img in enumerate(bin_batch):
            clamp_fct.set_clamped_val(img[clamped_idx])
            pos_samples[i] = lifsampl.sample_network_clamped(
                calib_file, w, b, duration, clamp_fct=clamp_fct, **sbs_kwargs)
        pos_hid_samples = pos_samples[..., n_vis:]
        pos_hid_probs = pos_hid_samples.mean(axis=1)

        pos_hid_samples = pos_samples[..., n_vis:]
        pos_hid_probs = pos_hid_samples.mean(axis=1)

        # obtain samples for model average
        logger.info('Collecting negative samples')
        duration = n_steps * sampling_interval
        neg_samples = lifsampl.sample_network(
            calib_file, w, b, duration, **sbs_kwargs)
        neg_vis_samples = neg_samples[:, :n_vis]
        neg_hid_samples = neg_samples[:, n_vis:]
        assert len(neg_vis_samples) == n_steps

        # Compute gradients
        pos_grad_w = bin_batch.T.dot(pos_hid_probs) / len(batch)
        neg_grad_w = neg_vis_samples.T.dot(neg_hid_samples) / n_steps
        grad_w = pos_grad_w - neg_grad_w
        grad_bh = np.mean(pos_hid_probs, axis=0) - \
            np.mean(neg_hid_samples, axis=0)
        grad_bv = np.mean(bin_batch, axis=0) - np.mean(neg_vis_samples, axis=0)

        return (grad_w, grad_bv, grad_bh)


# alternatively: CD with training data as initial vis. state
def compute_grad_cdn_serial(rbm, batch, n_steps, n_vis, n_hid_samples=10,
                            sbs_kwargs=None):
        if sbs_kwargs is None:
            sbs_kwargs = {
                'dt': .1,
                'burn_in_time': 500.,
                'sim_setup_kwargs': None,
                'sampling_interval': 10.
            }
        sampling_interval = sbs_kwargs['sampling_interval']
        w, b = rbm.bm_params()

        # data must be binary for LIF networks
        # bin_batch = (np.random.rand(*batch.shape) < batch)*1.
        bin_batch = np.round(batch)

        pos_duration = n_hid_samples * sampling_interval
        neg_duration = n_steps * sampling_interval
        refresh_times = np.array([0., pos_duration])
        clamped_idx = [np.arange(n_vis), []]
        clamped_val = [np.random.rand(len(clamped_idx)), []]
        clamp_fct = lifsampl.Clamp_anything(refresh_times, clamped_idx,
                                            clamped_val)

        # run simulations for each image in the batch
        # average hidden samples before computing corelation
        samples = np.zeros((len(batch), n_hid_samples + n_steps,
                            rbm.n_visible + rbm.n_hidden))
        logger.info('Collecting samples')
        for i, img in enumerate(bin_batch):
            clamp_fct.set_clamped_val([img[clamped_idx[0]], []])
            samples[i] = lifsampl.sample_network_clamped(
                calib_file, w, b, pos_duration + neg_duration,
                clamp_fct=clamp_fct, **sbs_kwargs)

        pos_hid_samples = samples[:, :n_hid_samples, n_vis:]
        pos_hid_probs = pos_hid_samples.mean(axis=1)
        neg_vis_samples = samples[:, -1, :n_vis]
        neg_hid_samples = samples[:, -1, n_vis:]

        # Compute gradients
        pos_grad_w = bin_batch.T.dot(pos_hid_probs) / len(batch)
        neg_grad_w = neg_vis_samples.T.dot(neg_hid_samples) / n_steps
        grad_w = pos_grad_w - neg_grad_w
        grad_bh = np.mean(pos_hid_probs, axis=0) - \
            np.mean(neg_hid_samples, axis=0)
        grad_bv = np.mean(bin_batch, axis=0) - np.mean(neg_vis_samples, axis=0)

        return (grad_w, grad_bv, grad_bh)


def train(rbm, train_data, n_epochs=1, batch_size=10, lrate=.01,
          cd_steps=1, valid_set=None, momentum=0, weight_cost=1e-5,
          sbs_kwargs=None):
        assert train_data.shape[1] == rbm.n_visible
        # initializations
        n_instances = train_data.shape[0]
        n_batches = int(np.ceil(n_instances/batch_size))
        shuffled_data = train_data[np.random.permutation(n_instances), :]
        w_incr = np.zeros_like(rbm.w)
        vb_incr = np.zeros_like(rbm.vbias)
        hb_incr = np.zeros_like(rbm.hbias)
        initial_lrate = lrate

        # log monitoring quantities in this file
        logger.info('---------------------------------------------')
        logger.info('Training LIF-RBM for {} epochs with {} batches.'
                    ' Parameters:'.format(n_epochs, n_batches))
        logger.info('#hidden: {}; Batch size: {}; Learning_rate: {}; '
                    'CD-steps: {}'.format(rbm.n_hidden, batch_size, lrate,
                                          cd_steps))
        t_start = time.time()
        # increment_history = []
        for epoch_index in range(n_epochs):
            logger.info('Training epoch {}'.format(epoch_index + 1))
            # if momentum != 0 and epoch_index > 5:
            #     # momentum = min(momentum + .1, .9)
            #     momentum = .9

            for batch_index in range(n_batches):
                update_step = batch_index + n_batches * epoch_index
                t_step = time.time()
                # Other lrate schedules are possible
                lrate = initial_lrate * 2000 / (2000 + update_step)

                # pick mini-batch randomly; actually
                # http://leon.bottou.org/publications/pdf/tricks-2012.pdf
                # says that shuffling and sequential picking is also ok
                batch = shuffled_data[batch_index*batch_size:
                                      min((batch_index + 1)*batch_size,
                                          n_instances), :]

                # compute "gradient" on batch
                gradients = compute_grad(
                    rbm, batch, cd_steps, rbm.n_visible, sbs_kwargs=sbs_kwargs)

                # update parameters including momentum
                w_incr = momentum * w_incr + \
                    lrate * (gradients[0] - weight_cost * rbm.w)
                vb_incr = momentum * vb_incr + lrate * gradients[1]
                hb_incr = momentum * hb_incr + lrate * gradients[2]
                rbm.w += w_incr
                rbm.vbias += vb_incr
                rbm.hbias += hb_incr
                logger.info('Finished training step {} of {} in {:.1f}min)'
                         ''.format(update_step, n_batches*n_epochs,
                                   (time.time() - t_step)/60))

                # # Run monitoring
                # increment_history.append(w_incr/rbm.w)
                # log_file.write('{}\n'.format(update_step))
        t_train = time.time() - t_start
        logger.info('Training took {:.1f}min ({:.1f}s per training step)'
                 ''.format(t_train/60, t_train/n_epochs/n_batches))
        # np.save('increments', increment_history)


if __name__ == '__main__':

    img_shape = (18, 24)
    pot_str = 'pong'
    # simulation parameters
    seed = 7741092
    sim_setup_kwargs = {
        'rng_seeds_seed': seed
    }
    mixing_tso_params = {
        "U": .01,
        "tau_rec": 280.,
        "tau_fac": 0.
    }
    sbs_kwargs = {
        'dt': .1,
        'burn_in_time': 300.,
        'sim_setup_kwargs': sim_setup_kwargs,
        'sampling_interval': 10.,
        "tso_params": 'renewing'
    }

    # # load rbm and data
    # data_name = pot_str + '_fixed_start{}x{}'.format(*img_shape)
    # train_set, _, _ = load_images(data_name)
    # assert len(train_set[1].shape) == 2
    # train_wlabel = np.concatenate((train_set[0][:20], train_set[1][:20]), axis=1)

    # rbm = load_rbm(data_name + '_crbm')
    # nv, nh, nl = rbm.n_visible, rbm.n_hidden, rbm.n_labels

    seed = 1234567
    np.random.seed(seed)
    nv = 3
    nh = 2
    w_small = 2*(np.random.beta(1.5, 1.5, (nv, nh)) - .5)
    w_small = np.zeros((nv, nh))
    bv = np.random.randn(nv)
    bh = np.random.randn(nh)

    # generate samples from true distribution and train bm
    target_rbm = RBM(nv, nh, w_small, bv, bh, numpy_seed=seed)
    tmp = target_rbm.draw_samples(int(2e3), binary=True)
    train_samples = tmp[1000:, :nv]

    rbm = RBM(nv, nh)
    train(rbm, train_samples, n_epochs=3, lrate=.01,
          cd_steps=1000, momentum=0.5, weight_cost=1e-4, sbs_kwargs=sbs_kwargs)

    # # Save rbm for later use
    # with open('../shared_data/saved_rbms/post_lif.pkl', 'wb') as output:
    #     cPickle.dump(rbm, output, cPickle.HIGHEST_PROTOCOL)

    w, b = rbm.bm_params()
    # run bm and compare histograms
    samples = lifsampl.sample_network(calib_file, w, b, 1e5, **sbs_kwargs)

    logger.info('Make histograms...')
    decimals = bin_to_dec(samples[:, :nv])
    decimals_train = bin_to_dec(train_samples)
    plt.figure()
    h1 = plt.hist(decimals, bins=np.arange(0, 2**nv + 1, 1), normed=True,
                  alpha=0.5, label='trained', color='g', align='mid',
                  rwidth=0.5, log=True)
    h2 = plt.hist(decimals_train, bins=np.arange(0, 2**nv + 1, 1), normed=True,
                  alpha=0.5, label='groundtruth', color='b', align='mid',
                  rwidth=0.5, log=True)
    plt.legend(loc='upper left')
    plt.savefig('test.png')
    plt.close()

    p_target = np.histogram(decimals_train, bins=np.arange(0, 2**nv + 1, 1),
                            normed=True)[0]
    logger.info("DKL = " + str(compute_dkl(samples[:, :nv], p_target)))

    # # plot increment
    # increments = np.load('increments.npy')
    # plt.figure()
    # plt.hist(increments.flatten(), bins='auto')
    # plt.savefig('incr.png')

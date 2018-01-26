#! /usr/bin/env python
from __future__ import division
from __future__ import print_function
import numpy as np
from rbm import RBM, CRBM
from rbm import load as load_rbm
import sys
import os
import yaml
import cPickle
import logging
from lif_pong.utils.data_mgmt import load_images
from lif_pong.utils import to_1_of_c
import lif_pong.sampling.gibbs_window_expt as winexpt


def run_window_expt(rbm, data_set):
    n_instances = len(data_set[0])
    chunksize = 50
    # iterate over chunks and sum the analysis quantities
    cum_prederr_sum = 0
    cum_prederr_sqsum = 0
    # move sim.yaml because it would be overwritten
    try:
        os.rename('sim.yaml', 'sim.yaml.backup')
    except OSError:
        print('No sim.yaml found in folder.', file=sys.stderr)

    counter = 0
    for i in range(0, n_instances, chunksize):
        d = {
            'data_name': general_dict['data_name'],
            'seed': 828384,
            'n_samples': 20,
            'winsize': general_dict['img_shape'][1],
            'img_shape': general_dict['img_shape'],
            'binary': True,
            'gather_data': True,
            'burn_in': 100,
            'start_idx': i,
            'chunksize': chunksize
        }
        # save a fake sim.yaml
        simdict = {'general': d, 'identifier': {'dummy': 42}}
        with open('sim.yaml', 'w') as f:
            f.write(yaml.dump(simdict))

        try:
            # Samples-file would be reloaded after the first chunk -> put away
            os.remove('samples.npz')
        except OSError:
            if i > 0:
                print('Error: No samples.npz found. Check if result is valid.',
                      file=sys.stderr)
        analysis_dict = winexpt.main(data_set, rbm, d, {})

        cum_prederr_sum += analysis_dict['cum_prederr_sum']
        cum_prederr_sqsum += analysis_dict['cum_prederr_sqsum']
        counter += analysis_dict['n_instances']
    # clean up
    os.remove('prediction.npz')
    os.remove('agent_performance.npz')
    os.remove('analysis')
    os.remove('wrong_cases')
    os.remove('sim.yaml')
    # restore sim.yaml
    try:
        os.rename('sim.yaml.backup', 'sim.yaml')
    except OSError:
        print('No backup found in folder.', file=sys.stderr)

    assert counter == n_instances
    cum_prederr_std = np.sqrt(cum_prederr_sqsum/n_instances -
                              (cum_prederr_sum/n_instances)**2)
    return cum_prederr_sum/n_instances, cum_prederr_std


def analyse_quality(rbm, train_set, valid_set):
    train_data = train_set[0]
    valid_data = valid_set[0]
    if len(train_set[1].shape) == 1:
        train_labels = train_set[1]
        valid_labels = valid_set[1]
    else:
        train_labels = np.argmax(train_set[1], axis=1)
        valid_labels = np.argmax(valid_set[1], axis=1)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    log_name = 'analysis.log'
    ch = logging.FileHandler(log_name)
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s : %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    result_dict = {}
    if hasattr(rbm, 'n_labels'):
        # compute classification rates
        train_rate = (rbm.classify(train_data) == train_labels).mean()
        valid_rate = (rbm.classify(valid_data) == valid_labels).mean()
        logger.info('Classification rate on train/valid set: '
                    '{:.3f} / {:.3f}'.format(train_rate, valid_rate))
        result_dict.update({'classrate_valid': float(valid_rate),
                            'classrate_train': float(train_rate)})

    # compute ais
    if hasattr(rbm, 'n_labels'):
        train_data = np.concatenate(
            (train_set[0], to_1_of_c(train_labels, rbm.n_labels)), axis=1)
        valid_data = np.concatenate(
            (valid_set[0], to_1_of_c(valid_labels, rbm.n_labels)), axis=1)
    logger.info('Starting with AIS run...')
    loglik_valid, loglik_train = \
        rbm.run_ais(valid_data, logger, train_data=train_data, n_runs=100)

    result_dict.update({'loglik_valid': float(loglik_valid),
                        'loglik_train': float(loglik_train)})

    # evaluate on prediction task
    logger.info('Starting Window experiment evaluation on validation set...')
    cum_prederr_mean, cum_prederr_std = run_window_expt(rbm, valid_set)
    result_dict.update({'cum_prederr_mean_valid': float(cum_prederr_mean),
                        'cum_prederr_std_valid': float(cum_prederr_std)})
    logger.info('Finished analysis')

    return result_dict


def main(data, img_shape, rbm_params, hyper_params, log_params,
         identifier_dict, save_file):
    n_pixels = np.prod(img_shape)
    train_set, valid_set, _ = data
    assert n_pixels == train_set[0].shape[1]

    try:
        with open(save_file) as f:
            rbm_dict = cPickle.load(f)
        my_rbm = load_rbm(rbm_dict)
        print('Loaded RBM from file.')
    except IOError:
        # train RBM
        if rbm_params['n_labels'] == 0:
            print('Training generative RBM on  {}'
                  ' instances...'.format(train_set[0].shape[0]))
            rbm_params.pop('n_labels')

            # discard labels
            train_data = train_set[0]
            valid_data = valid_set[0]

            # initialize biases like in Hinton's guide
            pj = np.average(train_data, axis=0)
            pj[pj == 0] = 1e-5
            pj[pj == 1] = 1 - 1e-5
            bias_init = np.log(pj / (1 - pj))

            my_rbm = RBM(n_pixels, vbias=bias_init, **rbm_params)
        else:
            print('Training discriminative RBM on  {}'
                  ' instances...'.format(train_set[0].shape[0]))

            # transform labels into one-hot representation if necessary
            if len(train_set[1].shape) == 1:
                train_targets = to_1_of_c(train_set[1], rbm_params['n_labels'])
                valid_targets = to_1_of_c(valid_set[1], rbm_params['n_labels'])
            else:
                train_targets = train_set[1]
                valid_targets = valid_set[1]

            train_data = np.concatenate((train_set[0], train_targets), axis=1)
            valid_data = np.concatenate((valid_set[0], valid_targets), axis=1)

            # initialize biases like in Hinton's guide
            pj = np.average(train_data, axis=0)
            pj[pj == 0] = 1e-5
            pj[pj == 1] = 1 - 1e-5
            bias_init = np.log(pj / (1 - pj))

            my_rbm = CRBM(n_inputs=n_pixels, vbias=bias_init, **rbm_params)

        kwargs = hyper_params.copy()
        kwargs.update(log_params)
        my_rbm.train(train_data, valid_set=valid_data, **kwargs)
        my_rbm.save(save_file)

    # make an analysis file here (e.g. valid set classification rate)
    analysis_dict = identifier_dict.copy()
    result_dict = analyse_quality(my_rbm, train_set, valid_set)
    analysis_dict.update(result_dict)

    with open('analysis', 'w') as f:
        f.write(yaml.dump(analysis_dict))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Wrong number of arguments. Please provide a yaml-config file.')
        sys.exit()
    with open(sys.argv[1]) as configfile:
        config = yaml.load(configfile)

    general_dict = config.pop('general')
    rbm_dict = config.pop('rbm')
    hyper_dict = config.pop('hyper')
    log_dict = config.pop('log')
    identifier_dict = config.pop('identifier')
    data = load_images(general_dict['data_name'])
    img_shape = tuple(general_dict['img_shape'])
    save_file = general_dict['save_file']

    main(data, img_shape, rbm_dict, hyper_dict, log_dict, identifier_dict,
         save_file)

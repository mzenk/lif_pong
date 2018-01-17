#! /usr/bin/env python
from __future__ import division
from __future__ import print_function
import numpy as np
from rbm import RBM, CRBM
import sys
import yaml
import logging
from lif_pong.utils.data_mgmt import load_images
from lif_pong.utils import to_1_of_c


def analyse_quality(rbm, train_set, test_set):
    train_data = train_set[0]
    test_data = test_set[0]
    if len(train_set[1].shape) == 1:
        train_labels = train_set[1]
        test_labels = test_set[1]
    else:
        train_labels = np.argmax(train_set[1])
        test_labels = np.argmax(test_set[1])

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
        test_rate = (rbm.classify(test_data) == train_labels).mean()
        logger.info('Error rate on train/test set: {:.2f} / {:.2f}'.format(
                     train_rate, test_rate))
        vislab_train = np.concatenate(
            (train_set[0], to_1_of_c(train_labels, rbm.n_labels)), axis=1)
        vislab_test = np.concatenate(
            (test_set[0], to_1_of_c(test_labels, rbm.n_labels)), axis=1)
        result_dict.update({'classrate_test': test_rate,
                            'classrate_train': train_rate})

    # compute ais
    if hasattr(rbm, 'n_labels'):
        train_data = vislab_train
        test_data = vislab_test
    logger.info('Starting with AIS run...')
    loglik_test, loglik_train = \
        rbm.run_ais(test_data, logger, train_data=train_data, n_runs=10)

    result_dict.update({'loglik_test': loglik_test,
                        'loglik_train': loglik_train})
    return result_dict


def main(data, img_shape, rbm_params, hyper_params, log_params,
         identifier_dict, save_file):
    n_pixels = np.prod(img_shape)
    train_set, valid_set, test_set = data
    assert n_pixels == train_set[0].shape[1]
    if rbm_params['n_labels'] == 0:
        print('Training generative RBM on Pong...')
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
        print('Training discriminative RBM on Pong on  {}'
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

    # make an analysis file here (e.g. test set classification rate)
    analysis_dict = identifier_dict.copy()
    result_dict = analyse_quality(my_rbm, train_set, test_set)
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

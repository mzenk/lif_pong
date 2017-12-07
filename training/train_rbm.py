from __future__ import division
from __future__ import print_function
import numpy as np
from rbm import RBM, CRBM
import sys
import yaml
from lif_pong.utils.data_mgmt import load_images
from lif_pong.utils import to_1_of_c


def main(data, img_shape, rbm_params, hyper_params, save_file, log_file):
    n_pixels = np.prod(img_shape)
    train_set, valid_set, test_set = data
    assert n_pixels == train_set[0].shape[1]
    if rbm_params['n_labels'] == 0:
        print('Training generative RBM on Pong...')
        rbm_params.pop('n_labels')

        # discard labels
        train_set = train_set[0]
        valid_set = valid_set[0]

        # initialize biases like in Hinton's guide
        pj = np.average(train_set, axis=0)
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

        train_set = np.concatenate((train_set[0], train_targets), axis=1)
        valid_set = np.concatenate((valid_set[0], valid_targets), axis=1)

        # initialize biases like in Hinton's guide
        pj = np.average(train_set, axis=0)
        pj[pj == 0] = 1e-5
        pj[pj == 1] = 1 - 1e-5
        bias_init = np.log(pj / (1 - pj))

        my_rbm = CRBM(n_inputs=n_pixels, vbias=bias_init, **rbm_params)

    my_rbm.train(train_set, valid_set=valid_set, log_name=log_file,
                 **hyper_params)

    my_rbm.save(save_file)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Wrong number of arguments. Please provide a yaml-config file.')
        sys.exit()
    with open(sys.argv[1]) as configfile:
        config = yaml.load(configfile)

    general_dict = config.pop('general')
    rbm_dict = config.pop('rbm')
    hyper_dict = config.pop('hyper')
    data = load_images(general_dict['data_name'])
    img_shape = general_dict['img_shape']
    save_file = general_dict['save_file']
    log_file = general_dict['log_file']

    main(data, img_shape, rbm_dict, hyper_dict, save_file, log_file)

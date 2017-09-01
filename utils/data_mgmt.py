import os
from inspect import stack
import numpy as np
import cPickle


# utilities for data management
def make_data_folder(name=None, shared_data=False):
    # get name of calling script
    caller_frame = stack()[1]
    caller_name = caller_frame[0].f_globals.get('__file__', None)
    script_name = os.path.basename(caller_name.split('.')[0])

    # if data is used across project save in shared_data folder
    if shared_data:
        home = os.path.expanduser('~')
        data_path = home + '/Projects/Pong/shared_data/'
    else:
        data_path = 'data/'

    if name is None:
        data_path += '{}_data/'.format(script_name)
    else:
        assert type(name) is str
        data_path += name + '/'

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    return data_path


def make_figure_folder():
    # get name of calling script
    caller_frame = stack()[1]
    caller_name = caller_frame[0].f_globals.get('__file__', None)
    script_name = os.path.basename(caller_name.split('.')[0])

    # if data is used across project save in shared_data folder
    figure_path = 'figures/{}/'.format(script_name)

    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    return figure_path


def get_data_path(source_script_name):
    data_path = 'data/{}_data/'.format(source_script_name)
    if not os.path.exists(data_path):
        raise OSError('No such file path: ' + data_path)
    return data_path


def load_images(data_name):
    path = os.path.expanduser('~') + '/Projects/Pong/shared_data/datasets/'
    with np.load(path + data_name + '.npz') as d:
        train_set, valid_set, test_set = d[d.keys()[0]]
    return train_set, valid_set, test_set


def load_rbm(data_name):
    path = os.path.expanduser('~') + '/Projects/Pong/shared_data/saved_rbms/'
    with open(path + data_name + '.pkl', 'rb') as f:
        rbm = cPickle.load(f)
    return rbm

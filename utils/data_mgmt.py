import os
from inspect import stack
import socket
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
        if 'nemo' in socket.gethostname():
            data_path = os.path.expanduser('~/git_repos/lif_pong/shared_data/')
        else:
            data_path = os.path.expanduser('~/Projects/lif_pong/shared_data/')
    else:
        data_path = 'data/'

    if name is None:
        data_path = os.path.join(data_path, '{}_data'.format(script_name))
    else:
        assert type(name) is str
        data_path = os.path.join(data_path, name)

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


def load_images(data_name, path=False):
    if path:
        with np.load(data_name) as d:
            if len(d.keys()) == 1:
                train_set, valid_set, test_set = d[d.keys()[0]]
            else:
                train_set = (d['train_data'], d['train_labels'])
                valid_set = (d['valid_data'], d['valid_labels'])
                test_set = (d['test_data'], d['test_labels'])
        return train_set, valid_set, test_set

    path = ''
    if socket.gethostname() == 'asdf':
        path = os.path.expanduser('~/mnt/hel_mnt/shared_data/datasets')
    elif 'nemo' in socket.gethostname():
        path = os.path.expanduser('~/git_repos/lif_pong/shared_data/datasets')
    else:
        path = os.path.expanduser('~/Projects/lif_pong/shared_data/datasets')
    with np.load(os.path.join(path, data_name + '.npz')) as d:
        if len(d.keys()) == 1:
            train_set, valid_set, test_set = d[d.keys()[0]]
        else:
            train_set = (d['train_data'], d['train_labels'])
            valid_set = (d['valid_data'], d['valid_labels'])
            test_set = (d['test_data'], d['test_labels'])
    return train_set, valid_set, test_set


# The previous load_rbm method can be problematic if no relative import is used, cf.
# https://stackoverflow.com/questions/13398462/unpickling-python-objects-with-a-changed-module-path
# Hence, I switched to a dictionary-based storing of RBMs, where a shortcut to the shared
# data path is provided by the following method and the loading done in the rbm module.
def get_rbm_dict(rbm_name):
    path = ''
    if socket.gethostname() == 'asdf':
        path = os.path.expanduser('~/mnt/hel_mnt/shared_data/saved_rbms')
    elif 'nemo' in socket.gethostname():
        path = os.path.expanduser('~/git_repos/lif_pong/shared_data/saved_rbms')
    else:
        path = os.path.expanduser('~/Projects/lif_pong/shared_data/saved_rbms')
    with open(os.path.join(path, rbm_name + '.pkl'), 'rb') as f:
        try:
            rbm_dict = cPickle.load(f)
        except ImportError:
            print('Cannot unpickle the object because it is a non-primitive type')
            return None
    return rbm_dict

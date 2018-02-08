#! /usr/bin/env python
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import sys
import yaml
from trajectory import Gaussian_trajectory, Const_trajectory
from scipy.ndimage import convolve1d
from lif_pong.utils.data_mgmt import make_data_folder
from lif_pong.utils import average_pool
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def get_angle_range(pos, width, height):
    lower = 180./np.pi * np.arctan(-(height + pos)/width)
    upper = 180./np.pi * np.arctan((2*height - pos)/width)
    return lower, upper


# this method draws states such that only one reflection can occurr for Pong
def draw_init_states(num, width, height, init_pos=None):
    if init_pos is None:
        init_pos = np.random.rand(num)*height
    else:
        assert num == len(init_pos)
    init_states = []
    lower, upper = get_angle_range(init_pos, width, height)
    init_angle = np.random.rand(num)*(upper - lower) + lower
    init_states = np.vstack((init_pos, init_angle)).T
    return init_states


def generate_data(num_train, num_valid, num_test, grid, fixed_start=False,
                  pot_dict=None, kink_dict=None, linewidth=1., dist_exponent=1.,
                  save_name=None, seed=7491055):
    # c = potential scale; h = grid spacing for pixeled image
    # actually, distinguishing grid and field may be unnecessary since we don't
    # care about physics here, i.e. realistic length scales
    # what does matters is the scale of the potential and the field
    grid = np.array(grid)
    h = 1./grid[0]
    field = grid * h
    v0 = 1.
    num_tot = num_train + num_valid + num_test

    # set default for missing values; type checking would be good here
    if pot_dict is None:
        pot_dict = {'type': 'pong'}
    if pot_dict['type'] == 'pong':
        if 'gradient' not in pot_dict.keys():
            print('Key \'gradient\' not found in dictionary. Using default.',
                  file=sys.stderr)
            pot_dict['gradient'] = [0, 0]
    elif pot_dict['type'] == 'gauss':
        for k in ['amplitude', 'mu', 'sigma_xx', 'sigma_xy', 'sigma_yy']:
            if k not in pot_dict.keys():
                print('Key \'{}\' not found in dictionary.'
                      ' Using default dictionary.'.format(k),
                      file=sys.stderr)
                pot_dict['amplitude'] = .8
                pot_dict['mu'] = [.5, .5]
                pot_dict['sigma_xx'] = 0.32
                pot_dict['sigma_xy'] = 0.
                pot_dict['sigma_yy'] = 0.25
                break
        amplitude = pot_dict['amplitude'] * .5*v0**2
        mu = field * np.array(pot_dict['mu'])
        cov_mat = np.array([[pot_dict['sigma_xx'], pot_dict['sigma_xy']],
                            [pot_dict['sigma_xy'], pot_dict['sigma_yy']]])**2 * \
            np.outer(field, field)

    # generate sample data set
    np.random.seed(seed)
    if fixed_start:
        starts = np.repeat([field[1]*.5], num_tot)
        init_states = draw_init_states(num_tot, field[0], field[1], starts)
    else:
        init_states = draw_init_states(num_tot, field[0], field[1])

    data = []
    impact_points = np.zeros(num_tot)
    print('Generating trajectories...')
    for i, s in enumerate(init_states):
        init_pos = s[0]
        init_angle =s[1]
        if pot_dict['type'] == 'pong':
            traj = Const_trajectory(pot_dict['gradient'], grid, h,
                                    np.array([0, init_pos]), init_angle, v0,
                                    kink_dict=kink_dict)
        if pot_dict['type'] == 'gauss':
            traj = Gaussian_trajectory(amplitude, mu, cov_mat, grid, h,
                                       np.array([0, init_pos]), init_angle, v0,
                                       kink_dict=kink_dict)
        traj.integrate()
        traj_pxls = traj.to_image(linewidth, dist_exponent)
        img_shape = traj_pxls.shape
        data.append(traj_pxls.flatten())
        impact_points[i] = traj.trace[-1, 1]

    data = np.array(data)
    # shuffle data so that successive samples do not have same start
    np.random.shuffle(data)

    # add label layer
    last_col = data.reshape((-1, img_shape[0], img_shape[1]))[..., -1]
    reflected = np.all(last_col == 0, axis=1)
    print('{} of {} balls were reflected.'.format(reflected.sum(),
                                                  data.shape[0]))
    # last_col = last_col[~reflected]
    # labels can be overlapping or not
    labwidth = 3
    if last_col.shape[1] % labwidth != 0:
        print('Setting label width = 3, although number of pixels is not'
              ' a multiple of 3.', file=sys.stderr)
    labels = average_pool(last_col, labwidth, labwidth)
    # label prob should be normalized
    z = np.sum(labels, axis=1)
    z[z == 0] = 1
    labels /= np.expand_dims(z, 1)

    if save_name is not None:
        train_data = data[:num_train]
        train_labels = labels[:num_train]
        valid_data = data[num_train:-num_test]
        valid_labels = labels[num_train:-num_test]
        test_data = data[-num_test:]
        test_labels = labels[-num_test:]

        np.savez_compressed(save_name + '_{}x{}'.format(*img_shape),
                            train_data=train_data, train_labels=train_labels,
                            valid_data=valid_data, valid_labels=valid_labels,
                            test_data=test_data, test_labels=test_labels)
        print('Saved data set with {0} samples and image shape {2}x{3}.'
              '(#labels = {1})'.format(
              data.shape[0], labels.shape[1], *img_shape))

    # some statistics
    plt.figure()
    counts, bins, _ = plt.hist(init_states[:, 1], bins='auto')
    plt.xlabel('Initial angle')
    plt.ylabel('#')
    plt.savefig('angle_dist.png')

    plt.figure()
    counts, bins, _ = plt.hist(init_states[:, 0], bins='auto')
    plt.plot([field[1]/2]*2, [0, counts.max()], 'r-')
    plt.xlim([min(bins.min(), 0), max(bins.max(), field[1])])
    plt.xlabel('Impact point')
    plt.ylabel('#')
    plt.savefig('start_points.png')

    plt.figure()
    counts, bins, _ = plt.hist(impact_points, bins='auto')
    plt.plot([field[1]/2]*2, [0, counts.max()], 'r-')
    plt.xlim([min(bins.min(), 0), max(bins.max(), field[1])])
    plt.xlabel('Impact point')
    plt.ylabel('#')
    # from scipy.stats import gaussian_kde
    # kernel = gaussian_kde(impact_points)
    # plt.plot(bins, kernel(bins)*impact_points.size, 'm.')
    plt.savefig('end_points.png')

    plt.figure()
    plt.imshow(data.mean(axis=0).reshape(img_shape),
               interpolation='Nearest', cmap='gray')
    plt.savefig('mean_image.png')
    # save histogram dat because it is not available afterwards
    np.savez(save_name + '_stats', init_pos=init_states[:, 0],
             init_angles=init_states[:, 1], end_pos=impact_points)

    return data, labels


def test():
    grid = np.array([48, 36])
    h = 1./grid[0]
    field = grid * h
    v0 = 1.

    # # general test
    # start_pos = np.array([0, field[1]/2])
    # ang = 30.
    # test = Const_trajectory(np.array([0., 0.]), grid, h, start_pos, ang, v0)
    # # amplitude = .4
    # # mu = field * [.5, .5]
    # # cov_mat = np.diag([.3, .2] * field)**2
    # # test = Gaussian_trajectory(
    # #     amplitude, mu, cov_mat, grid, h, start_pos, ang, v0)

    # test.integrate()
    # fig = plt.figure()
    # # test.draw_trajectory(fig, potential=True)
    # # imshow has the origin at the top left
    # linewidth = 5
    # pxls = test.to_image(linewidth)
    # print(pxls.shape)
    # plt.imshow(pxls, interpolation='Nearest', cmap='Blues', vmin=0, vmax=1,
    #            extent=(0, field[0], 0, field[1]))
    # plt.colorbar()
    # plt.savefig('test_trajectory.png')

    # test initialisation
    start_pos = np.linspace(0, field[1], 5)
    angles_lower, angles_upper = get_angle_range(start_pos, field[0], field[1])
    fig, ax = plt.subplots()
    for pos, al, au in zip(start_pos, angles_lower, angles_upper):
        # draw line for lower and upper limit for angle
        test = Const_trajectory(np.array([0., 0.]), grid, h, [0, pos], al, v0)
        test.integrate()
        test.draw_trajectory(ax, potential=True, color='C0')

        test = Const_trajectory(np.array([0., 0.]), grid, h, [0, pos], au, v0)
        test.integrate()
        test.draw_trajectory(ax, potential=True, color='C1')
    fig.savefig('test_init.png')


def main(data_dict, pot_dict, kink_dict):
    num_train = data_dict.pop('num_train')
    num_valid = data_dict.pop('num_valid')
    num_test = data_dict.pop('num_test')
    grid = data_dict.pop('grid')
    generate_data(num_train, num_valid, num_test, grid, pot_dict=pot_dict,
                  kink_dict=kink_dict, **data_dict)


if __name__ == '__main__':
    # more flexible version
    if len(sys.argv) != 2:
        print('Wrong number of arguments. Please provide a yaml-config file.')
        sys.exit()
    with open(sys.argv[1]) as configfile:
        config = yaml.load(configfile)

    data_dict = config.pop('data')
    if 'potential' not in config.keys():
        pot_dict = None
    else:
        pot_dict = config.pop('potential')
    if 'kink' not in config.keys():
        kink_dict = None
    else:
        kink_dict = config.pop('kink')
    main(data_dict, pot_dict, kink_dict)

    # # Pong
    # generate_data_old(10, 5, 5, [48, 36],
    #                   pot_str='pong', linewidth=5., save_name='test')

    # test()

#! /usr/bin/env python
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import sys
import yaml
from trajectory import Gaussian_trajectory, Const_trajectory, get_angle_range
from scipy.ndimage import convolve1d
from lif_pong.utils.data_mgmt import make_data_folder
from lif_pong.utils import average_pool
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# more general method in trajectory module
# def get_angle_range(pos, width, height):
#     lower = 180./np.pi * np.arctan(-(height + pos)/width)
#     upper = 180./np.pi * np.arctan((2*height - pos)/width)
#     return lower, upper


# this method draws states such that only one reflection can occurr for Pong
def draw_init_states(num, width, height, init_posy=None):
    if init_posy is None:
        init_posy = np.random.rand(num)*height
    else:
        assert num == len(init_posy)
    init_posx = np.zeros_like(init_posy)
    init_states = []
    lower, upper = get_angle_range(init_posx - .5*width, init_posy - .5*height,
                                   width, height)
    init_angle = (np.random.rand(num)*(upper - lower) + lower) * 180./np.pi
    init_states = np.vstack((init_posy, init_angle)).T
    return init_states


def generate_data(num_train, num_valid, num_test, grid, fixed_start=False,
                  pot_dict=None, kink_dict=None, linewidth=1., dist_exponent=1.,
                  save_name=None, seed=7491055):
    # c = potential scale; h = grid spacing for pixeled image
    # actually, distinguishing grid and field may be unnecessary since we don't
    # care about physics here, i.e. realistic length scales
    # what does matters is the scale of the potential and the field
    grid = np.array(grid)
    h = 10./grid[0]
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
    impact_points = []
    if kink_dict is not None:
        nokink_lastcols = []
        if 'clip_angle' in kink_dict.keys():
            clip_angle = kink_dict['clip_angle']
        else:
            clip_angle = False

    print('Generating trajectories...')
    for i, s in enumerate(init_states):
        init_pos = s[0]
        init_angle = s[1]
        if pot_dict['type'] == 'pong':
            traj = Const_trajectory(pot_dict['gradient'], grid, h,
                                    np.array([0, init_pos]), init_angle, v0,
                                    kink_dict=kink_dict)
        if pot_dict['type'] == 'gauss':
            traj = Gaussian_trajectory(amplitude, mu, cov_mat, grid, h,
                                       np.array([0, init_pos]), init_angle, v0,
                                       kink_dict=kink_dict)

        clipped_angle = traj.integrate()
        if clipped_angle and not clip_angle:
            continue
        traj_pxls = traj.to_image(linewidth, dist_exponent)
        img_shape = traj_pxls.shape
        data.append(traj_pxls.flatten())
        impact_points.append(traj.trace[-1, 1])

        # with kink we need to run integration another time to get the target
        # before the kink
        if kink_dict is not None:
            traj.reset()
            traj.kink_dict = None
            traj.integrate()
            nokink_pxls = traj.to_image(linewidth, dist_exponent)
            nokink_lastcols.append(nokink_pxls[:, -1])

    data = np.array(data)
    impact_points = np.array(impact_points)
    # shuffle data so that successive samples do not have same start
    shuffled_idx = np.random.permutation(len(data))
    data = data[shuffled_idx]
    init_states = init_states[shuffled_idx]
    impact_points = impact_points[shuffled_idx]
    if kink_dict is not None:
        nokink_lastcols = np.array(nokink_lastcols)[shuffled_idx]

    # add label layer
    last_col = data.reshape((-1, img_shape[0], img_shape[1]))[..., -1]
    reflected = np.all(last_col == 0, axis=1)
    print('{} of {} balls were reflected.'.format(reflected.sum(),
                                                  data.shape[0]))
    # remove reflected samples
    if reflected.sum() > 0:
        last_col = last_col[np.logical_not(reflected)]
        data = data[np.logical_not(reflected)]
        init_states = init_states[np.logical_not(reflected)]
        impact_points = impact_points[np.logical_not(reflected)]
        if kink_dict is not None:
            nokink_lastcols = nokink_lastcols[np.logical_not(reflected)]

    with open('reflected', 'w') as f:
        f.write(yaml.dump(np.nonzero(reflected)[0].tolist()))
    # last_col = last_col[~reflected]
    # labels can be overlapping or not
    labwidth = 4
    if last_col.shape[1] % labwidth != 0:
        print('Number of pixels is not a multiple of 4. Setting label width = 3',
              file=sys.stderr)
        labwidth = 3
    labels = average_pool(last_col, labwidth, labwidth)
    # label prob should be normalized
    z = np.sum(labels, axis=1)
    z[z == 0] = 1
    labels /= np.expand_dims(z, 1)

    if save_name is not None:
        num_train = min(num_train, len(data))
        num_test = min(num_test, len(data))
        train_data = data[:num_train]
        train_labels = labels[:num_train]
        valid_data = data[num_train:-num_test]
        valid_labels = labels[num_train:-num_test]
        test_data = data[-num_test:]
        test_labels = labels[-num_test:]

        if kink_dict is None:
            np.savez_compressed(save_name + '_{}x{}'.format(*img_shape),
                                train_data=train_data, train_labels=train_labels,
                                valid_data=valid_data, valid_labels=valid_labels,
                                test_data=test_data, test_labels=test_labels)
        else:
            np.savez_compressed(save_name + '_{}x{}'.format(*img_shape),
                                train_data=train_data, train_labels=train_labels,
                                valid_data=valid_data, valid_labels=valid_labels,
                                test_data=test_data, test_labels=test_labels,
                                nokink_lastcol=nokink_lastcols,
                                kink_pos=kink_dict['pos'])
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
    # save the data used for statistics; separate statistics for train/valid/test
    # can be obtained later by using n_train/n_valid/n_test
    np.savez(save_name + '_stats', init_pos=init_states[:, 0],
             init_angles=init_states[:, 1], end_pos=impact_points)

    return data, labels


def test():
    grid = np.array([48, 40])
    h = 10./grid[0]
    field = grid * h
    v0 = 1.

    # # general test
    # start_pos = np.array([0, 5.08])
    # ang = 22.6
    # ampls = np.linspace(0.2, .8, 5)
    # # amplitude = .4
    # # mu = field * [.5, .5]
    # # cov_mat = np.diag([.3, .2] * field)**2
    # # test = Gaussian_trajectory(
    # #     amplitude, mu, cov_mat, grid, h, start_pos, ang, v0)

    # fig, ax = plt.subplots()
    # for ampl in ampls:
    #     kink_dict = {'pos': ampl, 'ampl': 5., 'sigma': 0.}
    #     test = Const_trajectory(np.array([0., 0.]), grid, h, start_pos, ang, v0,
    #                             kink_dict=kink_dict)
    #     test.integrate()
    #     test.draw_trajectory(ax, potential=False)
    # # imshow has the origin at the top left
    # # linewidth = 5
    # # pxls = test.to_image(linewidth)
    # # print(pxls.shape)
    # # plt.imshow(pxls, interpolation='Nearest', cmap='Blues', vmin=0, vmax=1,
    # #            extent=(0, field[0], 0, field[1]))
    # # plt.colorbar()
    # plt.savefig('test_trajectory.png')

    # # test initialisation
    # start_posy = np.linspace(0, field[1], 5)
    # start_posx = np.zeros_like(start_posy, dtype=float)
    # centered_posx = start_posx - .5*field[0]
    # centered_posy = start_posy - .5*field[1]
    # min_angles, max_angles = get_angle_range(
    #     centered_posx, centered_posy, field[0], field[1])
    # min_angles *= 180./np.pi
    # max_angles *= 180./np.pi
    # fig, ax = plt.subplots()
    # for posx, posy, al, au in zip(start_posx, start_posy, min_angles, max_angles):
    #     # draw line for lower and upper angle limit
    #     test = Const_trajectory(np.array([0., 0.]), grid, h, [posx, posy], al, v0)
    #     test.integrate()
    #     test.draw_trajectory(ax, potential=True, color='C0')

    #     test = Const_trajectory(np.array([0., 0.]), grid, h, [posx, posy], au, v0)
    #     test.integrate()
    #     test.draw_trajectory(ax, potential=True, color='C1')
    # fig.savefig('test_init.png')

    # test angle restriction -> double reflections?
    # with knick double reflections are possible
    init_states = draw_init_states(1000, field[0], field[1])
    for i, s in enumerate(init_states):
        starty = s[0]
        startangle = s[1]
        print('{} percent done...'.format(100*i/len(init_states)))
        test = Const_trajectory(np.array([0., 0.]), grid, h, [0, starty],
                                startangle, v0)
        test.integrate()


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

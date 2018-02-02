from __future__ import division
from __future__ import print_function
import numpy as np
import os
import sys
# import yaml
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


def generate_data(num_train, num_valid, num_test, grid, pot_str='pong',
                  fixed_start=False, kink_dict=None, linewidth=1., fname=None):
    # c = potential scale; h = grid spacing for pixeled image
    # actually, distinguishing grid and field may be unnecessary since we don't
    # care about physics here, i.e. realistic length scales
    # what does matters is the scale of the potential and the field
    grid = np.array(grid)
    h = 1./grid[0]
    field = grid * h
    v0 = 1.
    num_tot = num_train + num_valid + num_test

    if pot_str == 'gauss':
        # potential parameters
        amplitude = .4
        mu = field * [.5, .5]
        # the units are wrong for the cov. matrix but values were set by eye
        cov_mat = np.diag([.1, .05] * field)

    # generate sample data set
    np.random.seed(7491055)
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
        if pot_str == 'pong':
            # No forces
            traj = Const_trajectory(np.array([0., 0.]), grid, h,
                                    np.array([0, init_pos]), init_angle, v0,
                                    kink_dict=kink_dict)
        if pot_str == 'gauss':
            # Hill in the centre
            traj = Gaussian_trajectory(amplitude, mu, cov_mat, grid, h,
                                       np.array([0, init_pos]), init_angle, v0,
                                       kink_dict=kink_dict)
        traj.integrate()
        traj_pxls = traj.to_image(linewidth)
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
    labwidth = 0
    for l in range(3, img_shape[0] + 1):
        if img_shape[0] % l == 0:
            labwidth = l
            break
    if labwidth > 5:
        print('Label width {} is rather large.'.format(labwidth), file=sys.stderr)
    labels = average_pool(last_col, labwidth, labwidth)
    # label prob should be normalized
    z = np.sum(labels, axis=1)
    z[z == 0] = 1
    labels /= np.expand_dims(z, 1)

    if fname is not None:
        train_data = data[:num_train]
        train_labels = labels[:num_train]
        valid_data = data[num_train:-num_test]
        valid_labels = labels[num_train:-num_test]
        test_data = data[-num_test:]
        test_labels = labels[-num_test:]

        savename = os.path.join(make_data_folder('datasets', True),
                                fname + '{}x{}'.format(*img_shape))
        np.savez_compressed(savename,
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

    # save histogram dat because it is not available afterwards
    np.savez(os.path.join(make_data_folder(), fname + '_stats'),
             init_pos=init_states[:, 0], init_angles=init_states[:, 1],
             end_pos=impact_points)

    plt.figure()
    plt.imshow(data.mean(axis=0).reshape(img_shape),
               interpolation='Nearest', cmap='gray')
    plt.savefig('mean_image.png')
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


def main(grid_dict, pot_dict, kink_dict):
    raise NotImplementedError


if __name__ == '__main__':
    # # more flexible version
    # if len(sys.argv) != 2:
    #     print('Wrong number of arguments. Please provide a yaml-config file.')
    #     sys.exit()
    # with open(sys.argv[1]) as configfile:
    #     config = yaml.load(configfile)

    # grid_dict = config.pop('grid')
    # pot_dict = config.pop('potential')
    # kink_dict = config.pop('kink')
    # main(grid_dict, pot_dict, kink_dict)

    # Pong
    generate_data(10000, 5, 5, [48, 36],
                  pot_str='pong', linewidth=5., fname='test')

    # test()

    # # Knick
    # # first dataset had random pos between 1/3 and 2/3 and ampl=.5, sigma=.2
    # knick_ampls = np.arange(.1, .9, .1)
    # knick_pos = np.arange(.2, .9, .1)
    # for a in knick_ampls:
    #     for p in knick_pos:
    #         d = {'pos': p, 'ampl': a, 'sigma': 0.1}
    #         generate_data([48, 36], pot_str='pong', kink_dict=d,
    #                       fname='knick_pos{:.1f}_ampl{:.1f}_var_start36x48'
    #                       ''.format(d['pos'], d['ampl']))

    # # check balancing of dataset
    # plt.figure()
    # abundances = np.insert(np.histogram(labels, bins=n_labels)[0], 0,
    #                        reflected.sum())
    # plt.bar(np.arange(-1, n_labels), abundances, width=1)
    # plt.savefig('label_abundance.png')

    # initial_histo, x, y = np.histogram2d(np.repeat(starts, nangles),
    #                                      angles.flatten(), bins=[grid[0], 50])

    # xgrid, ygrid = np.meshgrid(x, y, indexing='ij')
    # plt.figure()
    # plt.pcolormesh(xgrid, ygrid, initial_histo, cmap='gray')
    # # initial_histo / np.expand_dims(.1 + np.sum(initial_histo, axis=1), 1),

    # plt.axis([x.min(), x.max(), y.min(), y.max()])
    # plt.colorbar()
    # plt.savefig('combined_histo.png')

    # plt.figure()
    # plt.bar(np.arange(initial_histo.shape[0]), np.sum(initial_histo, axis=1),
    #         width=1)
    # plt.savefig('start_histo.png')

    # plt.figure()
    # plt.bar(np.arange(-initial_histo.shape[1]/2, initial_histo.shape[1]/2),
    #         np.sum(initial_histo, axis=0), width=1)
    # plt.savefig('angle_histo.png')

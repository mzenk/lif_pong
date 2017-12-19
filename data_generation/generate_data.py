from __future__ import division
from __future__ import print_function
import numpy as np
from trajectory import Gaussian_trajectory, Const_trajectory
from scipy.ndimage import convolve1d
from lif_pong.utils.data_mgmt import make_data_folder
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def pool_vector(vec, width, stride, mode='default'):
    kernel = np.ones(width)
    # kernel = np.convolve(kernel, kernel, mode='full')
    # sigma = .5
    # kernel = np.exp(-(np.arange(width) - width/2)**2 / sigma**2 / 2)
    kernel /= np.sum(kernel)
    filtered = convolve1d(vec, kernel, mode='constant', axis=1)
    # if no boundary effects should be visible
    if mode == 'valid':
        return filtered[:, width//2:-(width//2):stride]
    # startpoint is chosen such that the remainder of the division is
    # distributed evenly between left and right boundary
    start = ((vec.shape[1] - 1) % stride) // 2
    return filtered[:, start::stride]


def generate_data(grid, pot_str='pong', fixed_start=False, kink_dict=None,
                  fname=None):
    # c = potential scale; h = grid spacing for pixeled image
    # actually, distinguishing grid and field may be unnecessary since we don't
    # care about physics here, i.e. realistic length scales
    # what does matters is the scale of the potential and the field
    grid = np.array(grid)
    h = 1./grid[0]
    field = grid * h
    v0 = 1.

    if pot_str == 'gauss':
        # potential parameters
        amplitude = .4
        mu = field * [.5, .5]
        cov_mat = np.diag([.1, .05] * field)

    # generate sample data set
    np.random.seed(7491055)
    max_angle = 50.
    if fixed_start:
        nstarts = 1
        nangles = 2*1000
        starts = np.array([[field[1]*.5]])
    else:
        nangles = 200
        nstarts = 100
        starts = field[1]*np.random.beta(1.5, 1.5, nstarts)

    # draw for each start position nangles angles
    angles = max_angle * 2*(np.random.rand(nstarts, nangles) - .5)

    data = np.zeros((angles.size, np.prod(grid)))
    impact_points = np.zeros(angles.size)
    n = 0
    print('Generating trajectories...')
    for i, s in enumerate(starts):
        for a in angles[i]:
            if pot_str == 'pong':
                # No forces
                traj = Const_trajectory(np.array([0., 0.]),
                                        grid, h, np.array([0, s]), a, v0,
                                        kink_dict=kink_dict)
            if pot_str == 'gauss':
                # Hill in the centre
                traj = Gaussian_trajectory(amplitude, mu, cov_mat,
                                           grid, h, np.array([0, s]), a, v0,
                                           kink_dict=kink_dict)
            traj.integrate(write_pixels=fname is not None)
            data[n] = traj.pixels.flatten()
            # smooth if desired
            # tmp = gaussian_filter(traj.pixels, sigma=.7).flatten()
            # data[n] = tmp / np.max(tmp)
            impact_points[n] = traj.trace[-1, 1]
            n += 1

    counts, bins, _ = plt.hist(impact_points, bins=100)
    plt.plot([field[1]/2]*2, [0, counts.max()], 'r-')
    plt.xlabel('Impact point')
    plt.ylabel('#')
    # from scipy.stats import gaussian_kde
    # kernel = gaussian_kde(impact_points)
    # plt.plot(bins, kernel(bins)*impact_points.size, 'm.')
    plt.savefig('impact_points.png')
    # traj.draw_trajectory(potential=False)

    # shuffle data so that successive samples do not have same start
    np.random.shuffle(data)

    # add label layer
    last_col = data.reshape((data.shape[0], grid[1], grid[0]))[:, :, -1]
    reflected = np.all(last_col == 0, axis=1)
    print('{} of {} balls were reflected.'.format(reflected.sum(),
                                                  data.shape[0]))
    # last_col = last_col[~reflected]
    # labels can be overlapping or not
    labels = pool_vector(last_col, 3, 3, mode='valid')
    # label prob should be normalized
    z = np.sum(labels, axis=1)
    z[z == 0] = 1
    labels /= np.expand_dims(z, 1)
    # n_labels = labels.shape[1]
    # binarization if desired
    # labels = np.argmax(pooled_last_col, axis=1)

    # # test ----
    # n_col = 1
    # data = data.reshape((nstarts * nangles, grid[1], grid[0]))
    # data = data[:, :, -n_col:].reshape((nstarts * nangles, grid[1] * n_col))
    # # ----
    if fname is not None:
        size_train = data.shape[0]//4
        size_test = data.shape[0]//2
        train_set = data[:size_train]
        train_labels = labels[:size_train]
        valid_set = data[size_train: -size_test]
        valid_labels = labels[size_train: -size_test]
        test_set = data[-size_test:]
        test_labels = labels[-size_test:]

        np.savez_compressed(make_data_folder('datasets', True) + fname,
                            ((train_set, train_labels),
                             (valid_set, valid_labels),
                             (test_set, test_labels)))
    return data, labels


if __name__ == '__main__':
    kds = [{'pos': .7, 'ampl': .3, 'sigma': 0.1},
           {'pos': .6, 'ampl': .3, 'sigma': 0.1},
           {'pos': .5, 'ampl': .3, 'sigma': 0.1},
           {'pos': .4, 'ampl': .3, 'sigma': 0.1},
           {'pos': .3, 'ampl': .3, 'sigma': 0.1}]
    for i, d in enumerate(kds):
        generate_data([48, 36], pot_str='pong', kink_dict=d,
                      fname='knick{:02d}_var_start36x48'.format(i))

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

    # # testing
    # grid = np.array([48, 36])
    # h = 1./grid[0]
    # field = grid * h
    # v0 = .5

    # start_pos = np.array([0, field[1]/2])
    # ang = 30.
    # # coul_ampl = 1.
    # # coul_args = (coul_ampl, [field[0]/3, field[1]/2], 2*coul_ampl/v0**2)
    # # const_args = ([0., 1.],)
    # # amplitude = .4
    # # mu = field * [.5, .5]
    # # cov_mat = np.diag([.1, .05] * field)
    # # test = Gaussian_trajectory(grid, h, start_pos, ang, v0, amplitude, mu, cov_mat)
    # test = Const_trajectory(grid, h, start_pos, ang, v0, np.array([0, 0]))
    # test.integrate(write_pixels=True)
    # fig = plt.figure()
    # test.draw_trajectory(fig, potential=True)
    # # imshow has the origin at the top left
    # plt.imshow(test.pixels, interpolation='Nearest', cmap='Blues',
    #            origin='lower', extent=(0, field[0], 0, field[1]), alpha=0.5)
    # plt.savefig('test_trajectory.png')

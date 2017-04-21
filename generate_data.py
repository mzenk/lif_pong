from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.integrate import ode
from abc import ABCMeta, abstractmethod
from scipy.ndimage.filters import gaussian_filter
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


# abstract class for trajectory generation
class Trajectory:
    __metaclass__ = ABCMeta

    def __init__(self, grid_size, h, pos, angle, v0):
        # grid size: (n_pxls_y, n_pxls_x) y|_x
        # pos, angle, v0: starting position, angle, velocity
        self.pos = pos
        self.v0 = v0
        self.angle = angle / 180. * np.pi
        self.grid_size = np.array(grid_size)
        self.h = h
        # field size (x_range, y_range)
        self.field_size = h * self.grid_size[::-1]
        self.pixels = np.zeros(grid_size)
        # list for the entire trace
        self.trace = np.empty(1)

    @abstractmethod
    def force_fct(self, t, r):
        pass

    @abstractmethod
    def pot_fct(self, x, y):
        pass

    def integrate(self):
        vx = self.v0 * np.cos(self.angle)
        vy = self.v0 * np.sin(self.angle)
        size = self.field_size

        # use ode solver for integration
        r = ode(self.force_fct).set_integrator('vode', method='adams')
        r.set_initial_value([self.pos[0], self.pos[1], vx, vy], 0.)
        # This does not really make sense because the arguments are set by init

        t1 = 10*(size[0] + size[1])/2/self.v0
        # time scale: somehow related to field size and velocity; maybe adjust
        # program would be better/faster if dt larger, but dt needs to be
        # small enough to work with reflecting boundaries
        dt = .001*(size[0] + size[1])/2/self.v0
        curr_pos = np.zeros((1 + int(t1 / dt), 2))
        i = 0
        self.add_to_image(self.pos)
        while r.successful() and r.t < t1:
            r.integrate(r.t+dt)

            # stop if particle leaves field -> maybe add epsilon term to avoid
            # overshoot
            if r.y[0] > size[0] or r.y[0] < 0:
                break
            curr_pos[i, :] = r.y[:2]
            self.add_to_image(r.y[:2])

            # reflect if particle hits top or bottom
            if r.y[1] > size[1] or r.y[1] < 0:
                r.set_initial_value(r.y*np.array([1, 1, 1, -1]), r.t)
            i += 1

        self.trace = curr_pos[~np.all(curr_pos == 0, axis=1), :]
        # normalize pixels
        self.pixels /= np.max(self.pixels)
        return 0

    def draw_trajectory(self, potential=False):
        if self.trace.shape == (1,):
            print("integrate first!")
            return
        if potential:
            # overlay potential heatmap
            gridx, gridy = np.meshgrid(np.linspace(0, self.field_size[0], 70),
                                       np.linspace(0, self.field_size[1], 70))
            plt.imshow(self.pot_fct(gridx, gridy),
                       interpolation='Nearest',
                       extent=(0, self.field_size[0], 0, self.field_size[1]),
                       cmap='gray', origin='lower')
        plt.plot(self.trace[:, 0], self.trace[:, 1], '.', markersize=1)
        plt.xlim(0, self.field_size[0])
        plt.ylim(0, self.field_size[1])
        plt.savefig("trajectory.png")

    def add_to_image(self, point):
        # image coordinates are swapped natural coordinates
        point = point[::-1]
        # calculate floating point indices; pixel centers are at (n + .5)
        fp_ind = point/self.h - .5
        # special cases at the boundaries
        fp_ind[fp_ind < 0] = 0
        fp_ind[0] = min(fp_ind[0], self.grid_size[0] - 1)
        fp_ind[1] = min(fp_ind[1], self.grid_size[1] - 1)

        # distribute point influence over the four closest pixels
        lower = np.floor(fp_ind).astype(int)
        upper = np.ceil(fp_ind).astype(int)

        lower_frac = 1 - (fp_ind - lower)
        upper_frac = 1 - lower_frac
        self.pixels[lower[0], lower[1]] += np.prod(lower_frac)
        self.pixels[lower[0], upper[1]] += lower_frac[0] * upper_frac[1]
        self.pixels[upper[0], lower[1]] += lower_frac[1] * upper_frac[0]
        self.pixels[upper[0], upper[1]] += np.prod(upper_frac)

        # simpler scheme:
        # self.pixels[tuple(np.round(fp_ind).astype(int))] += 1


# class for r^-1 potential
class Coulomb_trajectory(Trajectory):
    def __init__(self, grid_size, h, pos, angle, v0,
                 amplitude, location, epsilon):
        # force parameters: amplitude, location and epsilon (if no infinite
        # potential wanted)
        super(Coulomb_trajectory, self).__init__(grid_size, h, pos, angle, v0)
        self.loc = np.array(location)
        self.amplitude = amplitude
        self.epsilon = epsilon

    def pot_fct(self, x, y):
        if type(x) is np.ndarray:
            r = np.linalg.norm(np.dstack((x, y)).reshape(x.size, 2) - self.loc,
                               axis=1)
            return (self.amplitude / (r + self.epsilon)).reshape(x.shape)
        else:
            r = np.linalg.norm(np.array([x, y]) - self.loc)
            return self.amplitude / (r + self.epsilon)

    def force_fct(self, t, y):
        r = np.linalg.norm(y[:2] - self.loc)
        grad = -self.amplitude / (r + self.epsilon)**3 * (y[:2] - self.loc)
        return [y[2], y[3], -grad[0], -grad[1]]


# class for constant force field
class Const_trajectory(Trajectory):
    def __init__(self, grid_size, h, pos, angle, v0, gradient):
        super(Const_trajectory, self).__init__(grid_size, h, pos, angle, v0)
        self.grad = gradient

    def pot_fct(self, x, y):
        return self.grad[0]*x + self.grad[1]*y

    def force_fct(self, t, y):
        return [y[2], y[3], -self.grad[0], -self.grad[1]]


# class for 2d-Gaussian hill
class Gaussian_trajectory(Trajectory):
    def __init__(self, grid_size, h, pos, angle, v0,
                 amplitude, mu, cov_mat):
        super(Gaussian_trajectory, self).__init__(grid_size, h, pos, angle, v0)
        self.amplitude = amplitude
        # store the inverse covariance matrix
        self.inv_covmat = np.linalg.inv(cov_mat)
        self.mu = mu

    def pot_fct(self, x, y):
        if type(x) is np.ndarray:
            centered_pos = np.dstack((x, y)).reshape(x.size, 2) - self.mu
            pot = self.amplitude * \
                np.exp(-0.5 * np.sum(centered_pos *
                                     centered_pos.dot(self.inv_covmat),
                                     axis=1))
            return pot.reshape(x.shape)
        else:
            centered_pos = np.array([x, y]) - self.mu
            return self.amplitude * \
                np.exp(-0.5 *
                       centered_pos.dot(self.inv_covmat.dot(centered_pos)))

    def force_fct(self, t, y):
        centered_pos = y[:2] - self.mu
        grad = -self.pot_fct(y[0], y[1]) * self.inv_covmat.dot(centered_pos)
        return [y[2], y[3], -grad[0], -grad[1]]


if __name__ == '__main__':
    # c = potential scale; h = grid spacing for pixeled image
    # actually, distinguishing grid and field may be unnecessary since we don't
    # care about physics here, i.e. realistic length scales
    # what does matters is the scale of the potential and the field

    grid = np.array([36, 48])
    fixed_start = True
    pot_str = 'gauss'

    h = 3./grid[0]
    field = grid[::-1] * h
    v0 = 1.

    # potential parameters
    amplitude = .5
    mu = field / 2
    cov_mat = np.diag([.7, .3])
    # generate sample data set
    np.random.seed(8563902)
    if fixed_start:
        nstarts = 1
        nangles = int(100/.3)*2
        starts = np.array([[field[0]*.5]])
        fname = pot_str + '_fixed_start{}x{}'.format(grid[0], grid[1])
    else:
        nangles = int(100/.3)
        nstarts = int(24 * 3 * 2)
        starts = field[0]*np.random.beta(1.5, 1.5, nstarts)
        fname = pot_str + '_var_start{}x{}'.format(grid[0], grid[1])
    # draw for each start position nangles angles
    angles = 50 * 2*(np.random.rand(nstarts, nangles) - .5)

    data = np.zeros((angles.size, np.prod(grid)))
    n = 0
    for i, s in enumerate(starts):
        for a in angles[i, :]:
            # # No forces
            # traj = Const_trajectory(grid, h, np.array([0, s]),
            #                         a, v0, np.array([0., 0.]))
            # Hill in the centre
            traj = Gaussian_trajectory(grid, h, np.array([0, s]), a, v0,
                                       amplitude, mu, cov_mat)
            traj.integrate()

            # smoothing + normalizing may not be necessary
            # tmp = gaussian_filter(traj.pixels, sigma=.7).flatten()
            # data[n] = tmp / np.max(tmp)
            data[n] = traj.pixels.flatten()
            n += 1

    # shuffle data so that successive samples do not have same start
    np.random.shuffle(data)

    # add label layer (#labels depends on size of paddle)
    n_labels = grid[0]//3
    last_col = data.reshape((data.shape[0], grid[0], grid[1]))[:, :, -1]
    # length of a column must be divisible by n_labels
    pooled_last_col = np.sum(last_col.reshape((last_col.shape[0], n_labels,
                                               last_col.shape[1] // n_labels)),
                             axis=2)

    # if integer labels desired
    labels = np.argmax(pooled_last_col, axis=1)

    # # test ----
    # n_col = 1
    # data = data.reshape((nstarts * nangles, grid[0], grid[1]))
    # data = data[:, :, -n_col:].reshape((nstarts * nangles, grid[0] * n_col))
    # # ----

    size_train = data.shape[0]//2
    size_test = data.shape[0]//4
    train_set = data[:size_train]
    train_labels = labels[:size_train]
    valid_set = data[size_train: -size_test]
    valid_labels = labels[size_train: -size_test]
    test_set = data[-size_test:]
    test_labels = labels[-size_test:]

    np.savez_compressed('datasets/' + fname,
                        ((train_set, train_labels),
                         (valid_set, valid_labels),
                         (test_set, test_labels)))

    # # inspect the data
    # data = (data, labels)

    # from util import tile_raster_images
    # # inspect pixels
    # samples = tile_raster_images(data[0][:9, :],
    #                              img_shape=(grid[0], grid[1]),
    #                              tile_shape=(3, 3),
    #                              tile_spacing=(1, 1),
    #                              scale_rows_to_unit_interval=True,
    #                              output_pixel_vals=False)

    # plt.figure()
    # plt.imshow(samples, interpolation='Nearest', cmap='gray')
    # plt.savefig('imgs.png')

    # check balancing of dataset
    # plt.figure()
    # plt.hist(labels, bins=n_labels)
    # plt.savefig('label_abundance.png')

    # initial_histo, x, y = np.histogram2d(np.repeat(starts, nangles),
    #                                      angles.flatten(), bins=[grid[0], 100])

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
    # start_pos = np.array([0, field[1]/2])
    # ang = 0.
    # v0 = 1.
    # coul_ampl = 1.
    # coul_args = (coul_ampl, [field[0]/3, field[1]/2], 2*coul_ampl/v0**2)
    # const_args = ([0., 1.],)
    # cov_mat = np.array([[.7, 0.], [0., .2]])
    # gauss_args = (.5*v0**2, [field[0]/2, field[1]/2], cov_mat)
    # test = Gaussian_trajectory(grid, h, start_pos, ang, v0, *gauss_args)
    # test.integrate()
    # test.draw_trajectory(potential=True)
    # # imshow has the origin at the top left
    # plt.imshow(test.pixels, interpolation='Nearest', cmap='Blues',
    #            origin='lower', extent=(0, field[0], 0, field[1]), alpha=0.5)
    # plt.show()

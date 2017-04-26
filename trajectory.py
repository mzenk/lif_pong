from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.integrate import ode
from abc import ABCMeta, abstractmethod
# from scipy.ndimage.filters import gaussian_filter
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


# abstract class for trajectory generation
class Trajectory:
    __metaclass__ = ABCMeta

    def __init__(self, grid_size, grid_spacing, pos, angle, v0):
        # grid size: (n_pxls_x, n_pxls_y) y|_x
        # pos, angle, v0: starting position, angle, velocity
        self.pos = pos
        self.v0 = v0
        self.angle = angle / 180. * np.pi
        self.grid_spacing = grid_spacing
        # field size (x_range, y_range) -> maybe center field later
        self.field_size = grid_spacing * np.array(grid_size)
        # for compatibility with plt.imshow, the pixels are saved as (ny, nx)
        self.pixels = np.zeros(grid_size[::-1])
        # list for the entire trace
        self.trace = np.empty(1)

    @abstractmethod
    def force_fct(self, t, r):
        pass

    @abstractmethod
    def pot_fct(self, x, y):
        pass

    def integrate(self, write_pixels=True):
        vx = self.v0 * np.cos(self.angle)
        vy = self.v0 * np.sin(self.angle)
        size = self.field_size

        # use ode solver for integration
        r = ode(self.force_fct).set_integrator('vode', method='adams')
        r.set_initial_value([self.pos[0], self.pos[1], vx, vy], 0.)

        # time scale: somehow related to field size and velocity; maybe adjust
        # program would be better/faster if dt larger, but dt needs to be
        # small enough to work with reflecting boundaries
        t1 = 10*(size[0] + size[1])/2/self.v0
        dt = .001*(size[0] + size[1])/2/self.v0
        curr_pos = np.zeros((1 + int(t1 / dt), 2))
        i = 0
        self.add_to_image(self.pos)
        while r.t < t1:
            r.integrate(r.t+dt)

            # stop if particle leaves field -> maybe add epsilon term to avoid
            # overshoot
            if r.y[0] > size[0] or r.y[0] < 0:
                break
            curr_pos[i] = r.y[:2]
            if write_pixels:
                self.add_to_image(r.y[:2])

            # reflect if particle hits top or bottom
            if r.y[1] > size[1] or r.y[1] < 0:
                r.set_initial_value(r.y*np.array([1, 1, 1, -1]), r.t)
            i += 1

        # # ------------
        # # other, simpler ode-solvers
        # # use ode solver for integration
        # t1 = 10*(size[0] + size[1])/2/self.v0
        # # time scale: somehow related to field size and velocity; maybe adjust
        # # program would be better/faster if dt larger, but dt needs to be
        # # small enough to work with reflecting boundaries
        # dt = .001*(size[0] + size[1])/2/self.v0
        # curr_pos = np.zeros((1 + int(t1 / dt), 2))
        # i = 0
        # t = 0
        # y = np.array([self.pos[0], self.pos[1], vx, vy], dtype=np.float64)
        # self.add_to_image(self.pos)
        # while True:
        #     # Euler -> fast but inaccurate (still sufficient?)
        #     # y += self.force_fct(t, y)*dt
        #     # # Leapfrog -> too slow
        #     # x = y[:2]
        #     # v = y[2:]
        #     # x += v*dt
        #     # v += self.force_fct(t, np.concatenate((x, v)))[2:]*dt

        #     # stop if particle leaves field -> maybe add epsilon term to avoid
        #     # overshoot
        #     if y[0] > size[0] or y[0] < 0:
        #         break
        #     if write_pixels:
        #         self.add_to_image(y[:2])

        #     # reflect if particle hits top or bottom
        #     if y[1] > size[1] or y[1] < 0:
        #         y *= [1, 1, 1, -1]
        #     i += 1
        #     t += dt
        #     curr_pos[i] = y[:2]
        # # ------------

        self.trace = curr_pos[~np.all(curr_pos == 0, axis=1), :]
        # normalize pixels
        self.pixels /= np.max(self.pixels)
        return 0

    def draw_trajectory(self, potential=False):
        if self.trace.shape == (1,):
            print("integrate first!")
            return

        fig = plt.figure()
        ax = fig.add_subplot(111)
        if potential:
            # overlay potential heatmap
            gridx, gridy = np.meshgrid(np.linspace(0, self.field_size[0], 70),
                                       np.linspace(0, self.field_size[1], 70))
            cax = ax.imshow(self.pot_fct(gridx, gridy),
                            interpolation='Nearest',
                            extent=(0, self.field_size[0],
                                    0, self.field_size[1]),
                            cmap='gray', origin='lower')
            fig.colorbar(cax)
        ax.plot(self.trace[:, 0], self.trace[:, 1], '.', markersize=1)
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), self.field_size[0],
                                                  self.field_size[1],
                                                  fill=False))
        ax.set_xlim(0 - self.field_size[0]*.1, self.field_size[0]*1.1)
        ax.set_ylim(0 - self.field_size[1]*.1, self.field_size[1]*1.1)
        fig.savefig("trajectory.png")

    def add_to_image(self, point):
        # image coordinates are swapped natural coordinates
        # calculate floating point indices; pixel centers are at (n + .5)
        fp_ind = point[::-1]/self.grid_spacing - .5
        # special cases at the boundaries
        fp_ind[fp_ind < 0] = 0
        fp_ind[0] = min(fp_ind[0], self.pixels.shape[0] - 1)
        fp_ind[1] = min(fp_ind[1], self.pixels.shape[1] - 1)

        # distribute point influence over the four closest pixels
        lower = np.floor(fp_ind).astype(int)
        upper = np.ceil(fp_ind).astype(int)

        lower_frac = 1 - (fp_ind - lower)
        upper_frac = 1 - lower_frac
        self.pixels[lower[0], lower[1]] += lower_frac[0] * lower_frac[1]
        self.pixels[lower[0], upper[1]] += lower_frac[0] * upper_frac[1]
        self.pixels[upper[0], lower[1]] += lower_frac[1] * upper_frac[0]
        self.pixels[upper[0], upper[1]] += upper_frac[0] * upper_frac[1]


# class for r^-1 potential
class Coulomb_trajectory(Trajectory):
    def __init__(self, grid_size, grid_spacing, pos, angle, v0,
                 amplitude, location, epsilon):
        # force parameters: amplitude, location and epsilon (if no infinite
        # potential wanted)
        super(Coulomb_trajectory, self).__init__(grid_size, grid_spacing, pos,
                                                 angle, v0)
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
        return np.array([y[2], y[3], -grad[0], -grad[1]])


# class for constant force field
class Const_trajectory(Trajectory):
    def __init__(self, grid_size, grid_spacing, pos, angle, v0, gradient):
        super(Const_trajectory, self).__init__(grid_size, grid_spacing, pos,
                                               angle, v0)
        self.grad = gradient

    def pot_fct(self, x, y):
        return self.grad[0]*x + self.grad[1]*y

    def force_fct(self, t, y):
        return np.array([y[2], y[3], -self.grad[0], -self.grad[1]])


# class for 2d-Gaussian hill
class Gaussian_trajectory(Trajectory):
    def __init__(self, grid_size, grid_spacing, pos, angle, v0,
                 amplitude, mu, cov_mat):
        super(Gaussian_trajectory, self).__init__(grid_size, grid_spacing, pos,
                                                  angle, v0)
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
        return np.array([y[2], y[3], -grad[0], -grad[1]])

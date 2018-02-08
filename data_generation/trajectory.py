from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.integrate import ode
from abc import ABCMeta, abstractmethod
# from scipy.ndimage.filters import gaussian_filter
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def gauss1d(x, mu, sigma):
        mu = np.repeat(np.expand_dims(mu, 1), x.shape[0], axis=1)
        return np.exp(-.5 * (x - mu)**2 / sigma**2)/np.sqrt(2)/sigma


# abstract class for trajectory generation
class Trajectory:
    __metaclass__ = ABCMeta

    def __init__(self, grid_size, grid_spacing, pos, angle, v0,
                 kink_dict=None):
        # grid size: (n_pxls_x, n_pxls_y) y|_x
        # pos, angle, v0: starting position, angle, velocity
        self.pos = np.array(pos)
        self.v0 = np.array(v0)
        self.angle = angle / 180. * np.pi
        self.grid_spacing = grid_spacing
        # field size (x_range, y_range) -> maybe center field later
        self.field_size = grid_spacing * np.array(grid_size)
        # for compatibility with plt.imshow, the pixels are saved as (ny, nx)
        self.pixels = np.zeros(grid_size[::-1])
        # list for the entire trace
        self.trace = np.empty(1)
        if kink_dict is not None:
            assert sorted(kink_dict.keys()) == sorted(['pos', 'ampl', 'sigma']), \
                'Got keys {}'.format(kink_dict.keys())
        self.kink_dict = kink_dict
        if self.kink_dict is not None:
            # kink_pos is given relative to field-xrange
            self.kink_dict['pos'] *= self.field_size[0]

    @abstractmethod
    def force_fct(self, t, r):
        pass

    @abstractmethod
    def pot_fct(self, x, y):
        pass

    def integrate(self, write_pixels=False):
        vx = self.v0 * np.cos(self.angle)
        vy = self.v0 * np.sin(self.angle)
        size = self.field_size

        if self.kink_dict is not None:
            noise_injected = False

        # use ode solver for integration
        r = ode(self.force_fct).set_integrator('vode', method='adams')
        r.set_initial_value([self.pos[0], self.pos[1], vx, vy], 0.)

        # time scale: somehow related to field size and velocity; maybe adjust
        # program would be better/faster if dt larger, but dt needs to be
        # small enough to work with reflecting boundaries
        t1 = 10*(size[0] + size[1])/2/self.v0
        dt = .001*size[1]/self.v0
        # particle can only travel max. dt*v0 further before reflection
        pos_list = []
        self.add_to_image(self.pos)
        while r.t < t1:
            r.integrate(r.t+dt)

            # stop if particle leaves field -> maybe add epsilon term to avoid
            # overshoot
            if r.y[0] > size[0] or r.y[0] < 0:
                break
            pos_list.append(r.y[:2])
            if write_pixels:
                self.add_to_image(r.y[:2])

            # inject random y-momentum
            if self.kink_dict is not None and r.y[0] > self.kink_dict['pos'] \
                    and not noise_injected:
                # good values for 'amp' and 'sigma': 0.5 and 0.2
                randmom = self.kink_dict['ampl'] * self.v0 * \
                    ((-1)**np.random.randint(2) + self.kink_dict['sigma'] *
                     np.random.randn())
                v_new = r.y[2:] + np.array([0, randmom])
                # normalize again so that pixels are still bright enough
                v_new /= np.linalg.norm(v_new) * self.v0
                r.set_initial_value(np.hstack((r.y[:2], v_new)), r.t)
                # shift kink_pos so that only one kink occurs
                noise_injected = True

            # reflect if particle hits top or bottom
            if r.y[1] > size[1] or r.y[1] < 0:
                r.set_initial_value(r.y*np.array([1, 1, 1, -1]), r.t)

        self.trace = np.array(pos_list)
        # normalize pixels
        self.pixels /= np.max(self.pixels)

    def draw_trajectory(self, ax, potential=False, color=None):
        if self.trace.shape == (1,):
            print("integrate first!")
            return

        if potential:
            # overlay potential heatmap
            gridx, gridy = np.meshgrid(np.linspace(0, self.field_size[0], 70),
                                       np.linspace(0, self.field_size[1], 70))
            ax.imshow(self.pot_fct(gridx, gridy), interpolation='Nearest',
                      extent=(0, self.field_size[0], 0, self.field_size[1]),
                      cmap='gray', origin='lower')
        ax.plot(self.trace[:, 0], self.trace[:, 1], '.', markersize=1, color=color)
        ax.add_patch(matplotlib.patches.Rectangle((0, 0), self.field_size[0],
                                                  self.field_size[1],
                                                  fill=False))
        ax.set_xlim(0 - self.field_size[0]*.1, self.field_size[0]*1.1)
        ax.set_ylim(0 - self.field_size[1]*.1, self.field_size[1]*1.1)

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

        # # alternative: gauss; most general solution would be a convolution
        # of a delta peak at the point with an arbitrary kernel
        # # points is a (nx2) array
        # points = np.array(point)
        # if len(points.shape) == 1:
        #     points = np.expand_dims(points, 0)
        # # switch coordinates for image

        # # create gauss array
        # x = np.arange(0, self.field_size[0], self.grid_spacing) \
        #     + .5*self.grid_spacing
        # y = np.arange(0, self.field_size[1], self.grid_spacing) \
        #     + .5*self.grid_spacing
        # gx = gauss1d(x, points[:, 0], self.grid_spacing)
        # gy = gauss1d(y, points[:, 1], self.grid_spacing)

        # # combine them to grid
        # self.pixels += gy.T.dot(gx)

    def to_image(self, linewidth=1., dist_exponent=1.):
        # need to add pixels on top/bottom to fit the linewidth inside
        # if linewidth/2 is not an integer, the excess pieces are cut away
        extra_y = int(.5*linewidth)
        gridx = np.arange(.5*self.grid_spacing, self.field_size[0],
                          self.grid_spacing)
        gridy = np.arange(.5*self.grid_spacing - extra_y*self.grid_spacing,
                          self.field_size[1] + extra_y*self.grid_spacing,
                          self.grid_spacing)

        # pixels have usual image axis convention
        pixelarr = np.zeros((len(gridy), len(gridx)))
        # very inefficient solution but one that works
        for i, y in enumerate(gridy):
            for j, x in enumerate(gridx):
                min_dist = np.min(np.linalg.norm(self.trace - [x, y], axis=1))
                # print(min_dist)
                pixelarr[i, j] = soft_assignment(
                    min_dist, .5*linewidth*self.grid_spacing, dist_exponent)

        return pixelarr[::-1]


def hard_assignment(dist, threshold):
    return 1.*(dist <= threshold)


def soft_assignment(dist, threshold, dist_exponent):
    if dist > threshold:
        return 0
    else:
        return np.power((threshold - dist)/threshold, dist_exponent)


# class for r^-1 potential
class Coulomb_trajectory(Trajectory):
    def __init__(self, amplitude, location, epsilon, *args, **kwargs):
        # force parameters: amplitude, location and epsilon (if no infinite
        # potential wanted)
        super(Coulomb_trajectory, self).__init__(*args, **kwargs)
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
    def __init__(self, gradient, *args, **kwargs):
        super(Const_trajectory, self).__init__(*args, **kwargs)
        self.grad = np.array(gradient)

    def pot_fct(self, x, y):
        return self.grad[0]*x + self.grad[1]*y

    def force_fct(self, t, y):
        return np.array([y[2], y[3], -self.grad[0], -self.grad[1]])


# class for 2d-Gaussian hill
class Gaussian_trajectory(Trajectory):
    def __init__(self, amplitude, mu, cov_mat, *args, **kwargs):
        super(Gaussian_trajectory, self).__init__(*args, **kwargs)
        self.amplitude = amplitude
        # store the inverse covariance matrix
        self.inv_covmat = np.linalg.inv(cov_mat)
        self.mu = np.array(mu)

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

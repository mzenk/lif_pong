from __future__ import division
from __future__ import print_function
import numpy as np
from trajectory import Gaussian_trajectory, Const_trajectory
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# c = potential scale; h = grid spacing for pixeled image
# actually, distinguishing grid and field may be unnecessary since we don't
# care about physics here, i.e. realistic length scales
# what does matters is the scale of the potential and the field
save_data = True
grid = np.array([48, 36])
fixed_start = False
pot_str = 'gauss'

h = 1./grid[0]
field = grid * h
v0 = 1.
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
    fname = pot_str + '_fixed_start{1}x{0}'.format(grid[0], grid[1])
else:
    nangles = 400
    nstarts = 70
    starts = field[1]*np.random.beta(1.5, 1.5, nstarts)
    fname = pot_str + '_var_start{1}x{0}'.format(grid[0], grid[1])
# draw for each start position nangles angles
angles = max_angle * 2*(np.random.rand(nstarts, nangles) - .5)

data = np.zeros((angles.size, np.prod(grid)))
impact_points = np.zeros(angles.size)
n = 0
for i, s in enumerate(starts):
    for a in angles[i]:
        # # No forces
        # traj = Const_trajectory(grid, h, np.array([0, s]),
        #                         a, v0, np.array([0., 0.]))
        # Hill in the centre
        traj = Gaussian_trajectory(grid, h, np.array([0, s]), a, v0,
                                   amplitude, mu, cov_mat)
        traj.integrate(write_pixels=save_data)
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

# add label layer (#labels depends on size of paddle)
n_labels = grid[0]//3
last_col = data.reshape((data.shape[0], grid[0], grid[1]))[:, :, -1]
reflected = np.all(last_col == 0, axis=1)
print('{} of {} balls were reflected.'.format(reflected.sum(),
                                              data.shape[0]))
# last_col = last_col[~reflected]
# length of a column must be divisible by n_labels
pooled_last_col = np.sum(last_col.reshape((last_col.shape[0], n_labels,
                                           last_col.shape[1] // n_labels)),
                         axis=2)
labels = np.argmax(pooled_last_col, axis=1)

# # test ----
# n_col = 1
# data = data.reshape((nstarts * nangles, grid[0], grid[1]))
# data = data[:, :, -n_col:].reshape((nstarts * nangles, grid[0] * n_col))
# # ----
if save_data:
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
# start_pos = np.array([0, field[1]/2])
# ang = 30.
# # coul_ampl = 1.
# # coul_args = (coul_ampl, [field[0]/3, field[1]/2], 2*coul_ampl/v0**2)
# # const_args = ([0., 1.],)
# amplitude = .4
# mu = field * [.5, .5]
# cov_mat = np.diag([.1, .05] * field)
# test = Gaussian_trajectory(grid, h, start_pos, ang, v0, amplitude, mu, cov_mat)
# test.integrate(write_pixels=False)
# test.draw_trajectory(potential=True)
# # imshow has the origin at the top left
# # plt.imshow(test.pixels, interpolation='Nearest', cmap='Blues',
# #            origin='lower', extent=(0, field[0], 0, field[1]), alpha=0.5)
# # plt.savefig('test_trajectory.png')

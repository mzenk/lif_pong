from __future__ import division
from __future__ import print_function
import numpy as np
from trajectory import Gaussian_trajectory, Const_trajectory, get_angle_range
import matplotlib
import matplotlib.pyplot as plt


def visualize_discretization():
    field = np.array([4*10/3, 10])
    h = field[0]/48
    grid = (field/h).astype(int)
    v0 = 1.

    # general test
    start_pos = np.array([0, .5*field[1]])
    ang = 24.

    fig, ax = plt.subplots()
    test = Const_trajectory(np.array([0., 0.]), grid, h, start_pos, ang, v0)
    test.integrate()
    test.draw_trajectory(ax, potential=False, color='C1', framecolor='C1')

    linewidth = 5
    pxls = test.to_image(linewidth)
    print(pxls.shape)
    ximg = np.arange(0, field[0] + h, h)
    yimg = np.arange(- int(.5*linewidth)*h,
                     field[1] + .5*h + int(.5*linewidth)*h, h)
    X, Y = np.meshgrid(ximg, yimg)

    ax.pcolormesh(X, Y, pxls[::-1], cmap='gray_r', vmin=0, vmax=1)
    ax.add_patch(matplotlib.patches.Rectangle(
        (ximg.min(), yimg.min()), ximg.max() - ximg.min(), yimg.max() - yimg.min(),
        fill=False, lw=2))

    ax.set_xlim([ximg.min()-.01, ximg.max() + .01])
    ax.set_ylim([yimg.min()-.05, yimg.max() + .01])
    plt.tight_layout()
    plt.axis('off')
    plt.savefig('discretization.pdf')


def visualize_init_angles():
    field = np.array([4*10/3, 10])
    h = field[0]/48
    grid = (field/h).astype(int)
    v0 = 1.

    start_posy = np.linspace(0, field[1], 5)
    start_posx = np.zeros_like(start_posy, dtype=float)
    centered_posx = start_posx - .5*field[0]
    centered_posy = start_posy - .5*field[1]
    min_angles, max_angles = get_angle_range(
        centered_posx, centered_posy, field[0], field[1])
    min_angles *= 180./np.pi
    max_angles *= 180./np.pi
    fig, ax = plt.subplots()
    for posx, posy, al, au in \
            zip(start_posx, start_posy, min_angles, max_angles):
        # draw line for lower and upper angle limit
        test = Const_trajectory(
            np.array([0., 0.]), grid, h, [posx, posy], al, v0)
        test.integrate()
        test.draw_trajectory(ax, potential=False, color='C0')

        test = Const_trajectory(
            np.array([0., 0.]), grid, h, [posx, posy], au, v0)
        test.integrate()
        test.draw_trajectory(ax, potential=False, color='C1')
    plt.xlim([-.01, field[0] + .01])
    plt.ylim([-.05, field[1] + .05])
    plt.tight_layout()
    plt.axis('off')
    fig.savefig('init_angles.pdf')


def visualize_angle_range():
    # visualize angle range
    width = 48.
    height = 40.
    nx = 100
    x = np.linspace(0, width, nx)
    y = np.linspace(0, height, int(nx*height/width))
    x_centered = x - x.max()/2
    y_centered = y - y.max()/2
    theo_max = 180/np.pi*np.arctan(2*y.max()/x.max())

    xgrid, ygrid = np.meshgrid(x_centered, y_centered)
    min_angles, max_angles = get_angle_range(xgrid, ygrid, width, height)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.set_title('Min. angle')
    ax2.set_title('Max. angle')
    im1 = ax1.imshow(min_angles*180/np.pi, interpolation='Nearest', cmap='viridis',
                     vmin=-theo_max, vmax=theo_max, origin='lower')
    im2 = ax2.imshow(max_angles*180/np.pi, interpolation='Nearest', cmap='viridis',
                     vmin=-theo_max, vmax=theo_max, origin='lower')
    # im2 = ax2.imshow(xgrid + ygrid, origin='lower', cmap='viridis')
    plt.colorbar(im1, ax=ax1, orientation='horizontal')
    plt.colorbar(im2, ax=ax2, orientation='horizontal')
    fig.savefig('angle_range.png')


if __name__ == '__main__':
    visualize_init_angles()

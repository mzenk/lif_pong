from scipy.integrate import ode
import matplotlib.pyplot as plt
import numpy as np


# y = [posx,posy,vx,vy]
a = 10./180.*np.pi
y0, t0 = [0., .5, np.cos(a), np.sin(a)], 0.


# 1/r potential
def invsq_pot(t, y, arg):
    epsilon = 0.01
    r = np.sqrt((y[0] - arg[0])**2 + (y[1] - arg[1])**2)
    return [y[2], y[3], arg[2]*(y[0] - arg[0])/(r + epsilon)**3,
            arg[2]*(y[1] - arg[1])/(r + epsilon)**3]


def coul_pot_fct(x, y, arg):
    return arg[2] / (np.sqrt((x - arg[0])**2 + (y - arg[1])**2) + 1.)


# constant/no potential
def const_pot(t, y, grad):
    return [y[2], y[3], grad[0], grad[1]]

r = ode(invsq_pot).set_integrator('vode', method='adams')
r.set_initial_value(y0, t0)
r.set_f_params([0., 1.])
coul_args = [.5, .5, .05]
r.set_f_params(coul_args)
t1 = 5.
dt = .01
pos = np.zeros((int(t1/dt) + 1, 2))
vel = np.zeros((int(t1/dt) + 1, 2))
i = 0
while r.successful() and r.t < t1:
    r.integrate(r.t+dt)
    pos[i, :] = r.y[:2]
    vel[i, :] = r.y[2:]
    if r.y[1] > 1 or r.y[1] < 0:
        r.set_initial_value(r.y*np.array([1, 1, 1, -1]), r.t)
    i += 1

# overlay potential heatmap
gridx, gridy = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
plt.imshow(coul_pot_fct(gridx, gridy[::-1, :], coul_args), interpolation=None,
           extent=(0, 1, 0, 1), cmap='gray')
# plt.plot(np.arange(0, t1 + dt, dt), y[:,0])
plt.plot(pos[:, 0], pos[:, 1], 'b.')
plt.show()

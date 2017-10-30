from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_ru_envelope(t_spike, U=1., tau_rec=0., tau_fac=0.):
    t_interspike = np.diff(t_spike)
    r_val = np.zeros(len(t_interspike) + 1)
    u_val = np.zeros(len(t_interspike) + 1)
    r_val[0] = 1.
    u_val[0] = U
    # need to add entry for first spike (special case)
    for n, dt in enumerate(t_interspike):
        u_val[n+1] = update_u(dt, u_val[n], U, tau_fac)
        r_val[n+1] = update_r(dt, r_val[n], u_val[n], tau_rec)
    return r_val, u_val


def update_r(dt, r_prev, U, tau_rec):
    if tau_rec == 0:
        return 1
    return 1 - (1 - (1 - U) * r_prev) * np.exp(-dt/tau_rec)


def update_u(dt, u_prev, U0, tau_fac):
    if tau_fac == 0:
        return U0
    return U0 + (1 - U0) * u_prev * np.exp(-dt/tau_fac)


if __name__ == '__main__':
    # test
    tso_params = {
        "U": .01,
        "tau_rec": 200.,
        "tau_fac": 300.,
    }

    tau_rec = 300.
    tau_fac = 0.
    U = .01

    spike_times = np.arange(0., 500, 10.)
    for tau_fac in np.linspace(100., 1000, 5):
        r, u = compute_ru_envelope(spike_times, U, tau_rec, tau_fac)
        # plt.plot(spike_times, r, '.')
        plt.plot(spike_times, r*u, '.')
        # plt.plot(spike_times, u, '.')
    # plt.xlabel('time')
    # plt.ylabel('TSO-variables')
    plt.savefig('test.png')

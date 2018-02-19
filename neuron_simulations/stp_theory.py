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


def u_theory(n, dt_spike, U0, tau_fac):
    if tau_fac == 0:
        return U0 * 1**n
    q = (1 - U0) * np.exp(-dt_spike/tau_fac)
    return U0 * (1 - q**n)/(1 - q)


def r_theory(n, dt_spike, U0, tau_rec, tau_fac=0):
    if tau_fac != 0:
        print('Warning: Formula is not valid for tau_fac != 0')
    q = (1 - u_theory(n, dt_spike, U0, tau_fac)) * np.exp(-dt_spike/tau_rec)
    a0 = 1 - np.exp(-dt_spike/tau_rec)
    return a0/(1 - q) + (1 - a0/(1 - q)) * q**(n-1)


def r_theo_interpolated(t, offset, dt_spike, U, tau_rec):
    # t can be an array (x)or offset
    # in that case, R is calculated for many times or for many instances with
    # different offsets at time t
    n = (t - offset)/dt_spike
    q = (1 - U) * np.exp(-dt_spike/tau_rec)
    a0 = 1 - np.exp(-dt_spike/tau_rec)
    result = np.zeros_like(np.array(n))
    result[n > 0] = a0/(1 - q) + (1 - a0/(1 - q)) * q**(n-1)
    result[n < 0] = 1.
    return result


def u_stat(U0, tau_fac, dt_spike=10.):
    # warning for tau_fac == 0, but function should still work in that case
    return U0/(1 - (1-U0)*np.exp(-dt_spike/tau_fac))


def r_stat(u_stat, tau_rec, dt_spike=10.):
    return (1 - np.exp(-dt_spike/tau_rec)) / \
        (1 - (1 - u_stat)*np.exp(-dt_spike/tau_rec))


def w_stat(tau_fac, U0, tau_rec=300.):
    us = u_stat(U0, tau_fac)
    return r_stat(us, tau_rec) * us/U0


def decay_const(U0, tau_rec, dt_spike):
    return 1./tau_rec - np.log(1 - U0)/dt_spike


def param_plot(attenuation_factor, clamp_duration, spike_interval=1.):
    def lower_bound_u(tau_rec):
        return (1/attenuation_factor - 1)*(np.exp(spike_interval/tau_rec) - 1)

    def win_cond_u(tau_rec):
        return 1 - np.exp(spike_interval*(1/tau_rec - 1/clamp_duration))

    trecrange = np.linspace(clamp_duration, 6000., 100)

    plt.semilogy(trecrange, lower_bound_u(trecrange), 'C0',
                 label='lower bound')
    plt.semilogy(trecrange, win_cond_u(trecrange), 'C1',
                 label='Winsize cond.')
    plt.xlabel('tau_rec [ms]')
    plt.ylabel('U')
    # plt.legend()
    # plt.savefig('params.png')


if __name__ == '__main__':
    # time_scaling =  1.
    # tau_rec = 2500.
    # tau_fac = 0.
    # U = .002
    # duration = 2000.
    # spike_interval = 1.
    # if time_scaling != 1:
    #     tau_rec *= time_scaling
    #     tau_fac *= time_scaling
    #     spike_interval *= time_scaling
    # spike_times = np.arange(0., duration, spike_interval)

    # # plot single example
    # r, u = compute_ru_envelope(spike_times, U, tau_rec, tau_fac)
    # w_env = r*u / (u[0]*r[0])
    # plt.plot(spike_times, w_env, '.')
    # # xtheo = 1 + np.arange(len(spike_times))
    # # rtheo = r_theory(xtheo, spike_interval, U, tau_rec, tau_fac)
    # # utheo = u_theory(xtheo, spike_interval, U, tau_fac)
    # # plt.plot(spike_times, rtheo * utheo / (rtheo[0]*utheo[0]), ':')
    # plt.ylim(ymin=0)
    # plt.xlabel('time [ms]')
    # plt.ylabel('Weight envelope [w(first spike)]')
    # plt.savefig('test.png')

    # # plot trace for several parameters
    # n_params = 5
    # tr_range = np.linspace(100., 1000, n_params)
    # tf_range = np.linspace(0., 500, n_params)
    # u0_range = np.linspace(1e-2, .1, n_params)
    # colors = [plt.cm.cool(i) for i in np.linspace(0, 1, n_params)]
    # ax = plt.axes()
    # ax.set_prop_cycle('color', colors)
    # # ax2 = ax.twinx()
    # # ax2.set_prop_cycle('color', colors)
    # # for tau_fac in tf_range:
    # # for tau_rec in tr_range:
    # for U in u0_range:
    #     r, u = compute_ru_envelope(spike_times, U, tau_rec, tau_fac)
    #     ax.plot(spike_times, r*u / (r[0]*u[0]), '.')
    #     # ax.plot(spike_times, w_stat(tau_fac, U, tau_rec)*np.ones_like(spike_times), 'k')
    #     # ax2.plot(spike_times, r)
    #     # ax2.plot(spike_times, u)
    # ax.set_xlabel('time [ms]')
    # ax.set_ylabel('Weight envelope [w(first spike)]')
    # ax.set_ylim(ymin=0)
    # plt.savefig('u_comparison.pdf')

    # scan parameter region (stationary value)
    spike_interval = 1.
    n_points = 50
    start_tr = 1
    stop_tr = 4.5
    start_u0 = -4
    stop_u0 = 0
    tr_logrange = np.logspace(start_tr, stop_tr, n_points)
    u0_logrange = np.logspace(start_u0, stop_u0, n_points)
    tr_grid, u0_grid = np.meshgrid(tr_logrange, u0_logrange)
    # actually must shift in log space but i'm too stupid
    # vecx = np.logspace(start_tr, stop_tr, n_points + 1) / \
    #     (tr_logrange.max() - tr_logrange.min())**(.5/n_points)
    # vecy = np.logspace(start_u0, stop_u0, n_points + 1) / \
    #     (u0_logrange.max() - u0_logrange.min())**(.5/n_points)
    r_stat_grid = r_stat(u0_grid, tr_grid, dt_spike=spike_interval)
    decay_time_grid = 1/decay_const(u0_grid, tr_grid, dt_spike=spike_interval)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    im1 = ax1.pcolormesh(tr_logrange, u0_logrange, r_stat_grid, vmin=0, vmax=1)
    im2 = ax2.pcolormesh(tr_logrange, u0_logrange, decay_time_grid)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    ax1.set(xlabel='tau_rec', ylabel='U', title='Stationary R')
    ax2.set(xlabel='tau_rec', ylabel='U', title='Decay time constant')
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)
    plt.savefig('param_logscan.png')

    # # parameter choice
    # attenuation_factor = np.linspace(.01, .1, 5)
    # clamp_duration = np.linspace(500., 2000., 5)
    # # clamp_duration = [2000.]
    # for cd in clamp_duration:
    #     for att in attenuation_factor:
    #         param_plot(att, cd, spike_interval=1.)
    # plt.savefig('params.png')

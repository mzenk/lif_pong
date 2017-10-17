#! /usr/bin/env python

from __future__ import division
from __future__ import print_function
import commands
import numpy as np
import sys
import os
from inspect import isfunction, getargspec
import json
import training.compute_isl as isl
from utils.data_mgmt import load_rbm, load_images, make_data_folder, \
    make_figure_folder, get_data_path


def submit_job(wrap_cmd, sbatch_options=''):
    cmd = "sbatch -p simulation {} --wrap=\"{}\"".format(sbatch_options,
                                                         wrap_cmd)
    print("Submitting job with command: %s " % cmd)
    status, job_msg = commands.getstatusoutput(cmd)
    if status == 0:
        print(job_msg)
    else:
        print("Error submitting job.")
    return job_msg.split(' ')[-1]


def sweep_rec():
    # u_range = np.linspace(0, .05, 10)
    # t_rec_range = np.linspace(10., 800., 10)
    u_range = np.linspace(0, 1., 5)
    t_rec_range = np.linspace(10., 1000., 5)

    rbm_name = 'pong_var_start36x48_crbm'
    calib_file = 'calibrations/dodo_calib.json'
    n_samples = 2000
    jobs = []
    # sweep uxt_rec (tau_fac = 0)
    for i, u in enumerate(u_range):
        for j, t_rec in enumerate(t_rec_range):
            args = (rbm_name, n_samples, calib_file)
            tso_params = {
                "U": u,
                "tau_rec": t_rec,
                "tau_fac": 0.
            }
            save_name = make_data_folder('stp_param_sweep_data') + \
                '{:02d}_{:02d}u_rec'.format(i, j)
            kwargs = {
                'tso_params': tso_params,
                'seed': 7741092,
                'save_name': save_name
            }
            config_file = save_name + '.json'
            with open(config_file, 'w') as f:
                json.dump([args, kwargs], f)
            jobs.append(
                submit_job('python lif_dreaming.py {}'.format(config_file))
                )

    # start ISL-job
    dependency = "--kill-on-invalid-dep=yes" \
        " --dependency=afterok:" + ":".join(jobs)
    submit_job('python stp_param_sweep.py compute_isl', dependency)


def sweep_fac():
    # sweep uxt_fac (optimal (greedy) tau_rec)
    t_rec = 280.
    u_range = np.linspace(0, 1., 5)
    t_fac_range = np.linspace(10., 1000., 5)

    rbm_name = 'pong_var_start36x48_crbm'
    calib_file = 'calibrations/dodo_calib.json'
    n_samples = 2000
    jobs = []
    # sweep uxt_rec (tau_fac = 0)
    for i, u in enumerate(u_range):
        for j, t_fac in enumerate(t_fac_range):
            args = (rbm_name, n_samples, calib_file)
            tso_params = {
                "U": u,
                "tau_rec": t_rec,
                "tau_fac": t_fac
            }
            save_name = make_data_folder('stp_param_sweep_data') + \
                '{:02d}_{:02d}u_fac'.format(i, j)
            kwargs = {
                'tso_params': tso_params,
                'seed': 7741092,
                'save_name': save_name
            }
            config_file = save_name + '.json'
            with open(config_file, 'w') as f:
                json.dump([args, kwargs], f)
            jobs.append(
                submit_job('python lif_dreaming.py {}'.format(config_file))
                )

    # start ISL-job
    dependency = "--kill-on-invalid-dep=yes" \
        " --dependency=afterok:" + ":".join(jobs)
    submit_job('python stp_param_sweep.py compute_isl', dependency)


def compute_isl():
    img_shape = (36, 48)
    n_pixels = np.prod(img_shape)
    data_name = 'pong_var_start{}x{}'.format(*img_shape)
    _, _, test_set = load_images(data_name)
    test_vis = test_set[0][:2000]
    directory = get_data_path('stp_param_sweep')

    # Compute ISL for each file in data folder
    n_u = 5
    n_t = 5
    ll = np.zeros((n_u, n_t))
    tso = np.zeros((n_u, n_t, 3))
    for i in range(n_u):
        for j in range(n_t):
            filename = '{:02d}_{:02d}u_rec.npz'.format(i, j)
            curr_f = os.path.join(directory, filename)
            if not os.path.isfile(curr_f):
                ll[i, j] = np.nan
                tso[i, j, :] = np.nan
                continue

            print('Processing ' + curr_f)
            with np.load(curr_f) as d:
                vis_samples = d['samples'].astype(float)[:, :n_pixels]
                tso_params = d['tso_params']

            dm = isl.ISL_density_model()
            dm.fit(vis_samples, quick=True)
            ll[i, j] = dm.avg_loglik(test_vis)
            tso[i, j] = tso_params

    np.savez_compressed(directory + 'isl_test', logliks=ll, tso_params=tso)


def plot():
    directory = get_data_path('stp_param_sweep')
    isl_file = 'isl_test.npz'
    # load isl data
    with np.load(directory + isl_file) as d:
        logliks = d['logliks']
        tso_params = d['tso_params']
    # # 3D
    # import matplotlib as mpl
    # mpl.use('Agg')
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # x = tso_params[..., 0]
    # y = tso_params[..., 1]

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(x, y, logliks, cmap=plt.cm.coolwarm, linewidth=0)

    # 2D
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    img = ax.imshow(logliks, interpolation='Nearest', origin='lower', cmap=plt.cm.hot)
              # extent=(min(x), max(x), min(y), max(y)))
    plt.colorbar(img)

    fig.savefig(make_figure_folder() + 'test.png')

if __name__ == '__main__':
    local_globals = globals().keys()

    def is_noarg_function(f):
        "Test if f is valid function and has no arguments"
        func = globals()[f]
        if isfunction(func):
            argspec = getargspec(func)
            if len(argspec.args) == 0\
                    and argspec.varargs is None\
                    and argspec.keywords is None:
                return True
        return False

    def show_functions():
        functions.sort()
        for f in functions:
            print(f)
    functions = [f for f in local_globals if is_noarg_function(f)]
    if len(sys.argv) <= 1 or sys.argv[1] == "-h":
        show_functions()
    else:
        for launch in sys.argv[1:]:
            if launch in functions:
                run = globals()[launch]
                run()
            else:
                print(launch, "not part of functions:")
                show_functions()

from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import sys
import os
import yaml
import matplotlib.pyplot as plt
from lif_pong.utils.data_mgmt import make_figure_folder

def plot_for_param(param_name, df_avg, df_std):
    # plot for the specified parameter all quantities in (train, test)-pairs
    x_data = df_avg[param_name]
    # assume we have only the keys 
    keys = ['classrate', 'loglik']
    rate_fig, rate_ax = plt.subplots()
    ll_fig, ll_ax = plt.subplots()
    ax_dict = {}
    ax_dict[keys[0]] = rate_ax
    ax_dict[keys[1]] = ll_ax
    for k in keys:
        trainy_data = df_avg[k+'_train'].as_matrix()
        testy_data = df_avg[k+'_test'].as_matrix()
        trainy_err = df_std[k+'_train'].as_matrix()
        testy_err = df_std[k+'_test'].as_matrix()
        ax_dict[k].errorbar(x_data, trainy_data, yerr=trainy_err,
            fmt='.', alpha=0.7, label='train')
        ax_dict[k].errorbar(x_data, testy_data, yerr=testy_err,
            fmt='.', alpha=0.7, label='test')
        ax_dict[k].set_xlabel(param_name)
        ax_dict[k].set_ylabel(k)
        ax_dict[k].legend()

    rate_fig.savefig(os.path.join(make_figure_folder(),
        param_name + '_crates.png'))
    ll_fig.savefig(os.path.join(make_figure_folder(),
        param_name + '_logliks.png'))


if len(sys.argv) != 2:
    print('Wrong number of arguments. Please provide the experiment name')
    sys.exit()
expt_name = sys.argv[1]

# collectfolder = os.path.expanduser('~/mnt/bwnemo_mnt/workspace/experiment/collect')
# simfolder = os.path.expanduser('~/mnt/bwnemo_mnt/workspace/experiment/simulations')
collectfolder = os.path.expanduser('~/workspace/experiment/collect')
simfolder = os.path.expanduser('~/workspace/experiment/simulations')
# load simulation config file
with open(os.path.join(simfolder, '01_runs', expt_name)) as f:
    simdict = yaml.load(f)
    stubdict = simdict['stub']
# load data from "collect" folder into data frame
with open(os.path.join(collectfolder, expt_name)) as f:
    df = pd.DataFrame(yaml.load(f))

id_params = stubdict['identifier'].keys()

# print the 10 best parameters
# sort for test crate and ll
sorted_prederr = df.sort_values(by='cum_prederr_mean', ascending=True)
sorted_loglik = df.sort_values(by='loglik_test', ascending=False)
print('Top 10 sorted by window expt performance on test set:')
print(sorted_prederr.iloc[:10, :])
print('Top 10 sorted by test set LL-estimate:')
print(sorted_loglik.iloc[:10, :])

# plot average and std with respect to each parameter
for param in id_params:
    print('Group dataframe by: {}'.format(param))
    grouped = df.groupby(param, as_index=False)
    result_avg = grouped.aggregate(np.mean)
    result_std = grouped.aggregate(np.std)
    for p in id_params:
        if p != param:
            result_avg.pop(p)
            result_std.pop(p)

    plot_for_param(param, result_avg, result_std)



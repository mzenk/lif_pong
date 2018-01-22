from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import sys
import os
import yaml
import matplotlib.pyplot as plt
from lif_pong.utils.data_mgmt import make_figure_folder


def plot_infsuccess_pcolor(df, identifier, figname='paramsweep.png'):
    assert len(identifier) == 2
    df['mean_prederr'] = df['cum_prederr_sum'] / df['n_instances']
    df['std_prederr'] = (df['cum_prederr_sqres'] / df['n_instances']).apply(np.sqrt)

    sorted_df = df.sort_values(by=identifier)
    # create axis vectors for meshgrid. This code assumes that the differences
    # between x/y-values are all a multiple of some smallest delta
    mins = [sorted_df[k].min() for k in identifier]
    maxs = [sorted_df[k].max() for k in identifier]
    mindiffs = [np.diff(sorted_df[k].unique()).min() for k in identifier]
    xvec, yvec = \
        [np.arange(mins[i], maxs[i] + 2*mindiffs[i],
                   mindiffs[i]) - .5*mindiffs[i] for i in range(2)]
    X, Y = np.meshgrid(xvec, yvec)

    # create value array with nans for non-existing data
    # # doesn't work with numpy 1.11. Also, fails if a combination is missing
    # C = np.ones((len(yvec), len(xvec))) * np.nan
    # mask = np.logical_and(np.isin(X + .5*mindiffs[0], sorted_df[identifier[0]]),
    #                       np.isin(Y + .5*mindiffs[1], sorted_df[identifier[1]]))
    # C[mask] = sorted_df['mean_prederr'].as_matrix().reshape(C[mask].shape)

    C = np.ones(len(yvec) * len(xvec)) * np.nan
    C_std = np.ones(len(yvec) * len(xvec)) * np.nan
    data_xy = sorted_df.loc[:, identifier].as_matrix()
    X_params_flat = (X + .5*mindiffs[0]).flatten()
    Y_params_flat = (Y + .5*mindiffs[1]).flatten()
    for i, xy in enumerate(zip(X_params_flat, Y_params_flat)):
        occurrence = np.where(np.all(np.isclose(xy, data_xy), axis=1))[0]
        if len(occurrence) > 0:
            assert len(occurrence) == 1
            C[i] = sorted_df['mean_prederr'].iloc[occurrence[0]]
            C_std[i] = sorted_df['std_prederr'].iloc[occurrence[0]]

    min_idx = np.nanargmin(C)
    print('Minimum {0} +- {1} at ({4}, {5}) = ({2}, {3})'.format(
        C[min_idx], C_std[min_idx],
        X_params_flat[min_idx], Y_params_flat[min_idx], *identifier))
    C = C.reshape(len(yvec), len(xvec))
    C_std = C_std.reshape(len(yvec), len(xvec))

    # save mean
    fig_mean, ax_mean = plt.subplots()
    im = ax_mean.pcolormesh(X, Y, np.ma.masked_where(np.isnan(C), C),
                            cmap=plt.cm.viridis)
    aspect = (C.shape[0] - 1)/(C.shape[1] - 1) \
        * (maxs[0] - mins[0])/(maxs[1] - mins[1])
    ax_mean.set_aspect(float(aspect))
    plt.xlabel(identifier[0])
    plt.ylabel(identifier[1])
    cbar_m = plt.colorbar(im, ax=ax_mean)
    cbar_m.ax.set_ylabel('Mean cum. prediction error')
    fig_mean.savefig(os.path.join(make_figure_folder(), figname) + '.png')

    # save std
    fig_std, ax_std = plt.subplots()
    im = ax_std.pcolormesh(X, Y, np.ma.masked_where(np.isnan(C_std), C_std),
                            cmap=plt.cm.viridis)
    ax_std.set_aspect(float(aspect))
    plt.xlabel(identifier[0])
    plt.ylabel(identifier[1])
    cbar_s = plt.colorbar(im, ax=ax_std)
    cbar_s.ax.set_ylabel('Std. of cum. prediction error')
    fig_std.savefig(os.path.join(make_figure_folder(), figname + '_std') + '.png')


def plot_infsuccess_1d(df, identifier, figname='paramsweep.png'):
    assert len(identifier) == 1
    df['mean_prederr'] = df['cum_prederr_sum'] / df['n_instances']
    df['std_prederr'] = (df['cum_prederr_sqres'] / df['n_instances']).apply(np.sqrt)

    sorted_df = df.sort_values(by=identifier)

    xdata = sorted_df[identifier[0]].as_matrix()
    ydata = sorted_df['mean_prederr'].as_matrix()
    yerr = sorted_df['std_prederr'].as_matrix()
    print("Minimum {} at {} = {}".format(ydata.min(), identifier[0],
                                         xdata[np.argmin(ydata)]))
    plt.errorbar(xdata, ydata, yerr=yerr, fmt='.:')
    plt.xlabel(identifier[0])
    plt.ylabel('Mean cum. prediction error')
    plt.savefig(os.path.join(make_figure_folder(), figname + '.png'))


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

# average over chunks
if 'identifier' in stubdict.keys():
    id_params = stubdict['identifier'].keys()
else:
    id_params = []
    for k in simdict['replacements'].keys():
        if k in df.columns:
            id_params.append(k)
    id_params.remove('start_idx')
print('Group dataframe by: ' + ', '.join('{}'.format(p) for p in id_params))
grouped = df.groupby(id_params, as_index=False)
result = grouped.aggregate(np.sum)
result.pop('start_idx')
if 'tau_fac' in result.keys():
    result.pop('tau_fac')

# plot heatmap of success rate
if len(id_params) == 1:
    plot_infsuccess_1d(result, id_params, figname=expt_name)
elif len(id_params) == 2:
    plot_infsuccess_pcolor(result, id_params, figname=expt_name)
elif 'weight' in result.columns:
    id_params.remove('weight')
    for weight in result['weight'].unique():
        subdf = result.loc[result.weight == weight, :].copy()
        subdf.pop('weight')
        # plot_trec_U(subdf, figname=expt_name + '_w={}'.format(weight))
        plot_infsuccess_pcolor(subdf, id_params, figname=expt_name + '_w={}'.format(weight))
else:
    print('Don\'t know what to do with the data')

from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import sys
import os
import yaml
from lif_pong.utils.data_mgmt import make_figure_folder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_cumerr_pcolor(df, identifier, figname='paramsweep.png', logscale=False,
                       title=''):
    assert len(identifier) == 2
    df['mean_cum_prederr'] = df['cum_prederr_sum'] / df['n_instances']
    df['std_cum_prederr'] = (df['cum_prederr_sqsum'] / df['n_instances'] -
                             df['mean_cum_prederr']**2).apply(np.sqrt)

    sorted_df = df.sort_values(by=identifier)
    mins = [sorted_df[k].min() for k in identifier]
    maxs = [sorted_df[k].max() for k in identifier]
    if logscale:
        xcenters, ycenters = [sorted_df[k].unique() for k in identifier]
        # pcolormesh takes grid edges
        tmp_list = []
        for vec in [xcenters, ycenters]:
            tmp = np.hstack((vec, vec[-1]*np.power(vec[-1]/vec[0], 1/len(vec))))
            shift_factor = np.power(tmp.max() / tmp.min(), -1./(2*len(tmp)))
            tmp_list.append(shift_factor*tmp)
        xedges, yedges = tmp_list

        X, Y = np.meshgrid(xedges, yedges)
        tmp_mesh = np.meshgrid(xcenters, ycenters)
        X_params_flat = tmp_mesh[0].flatten()
        Y_params_flat = tmp_mesh[1].flatten()
        print(xcenters, ycenters)
    else:
        # create axis vectors for meshgrid. This code assumes that the differences
        # between x/y-values are all a multiple of some smallest delta
        mindiffs = [np.diff(sorted_df[k].unique()).min() for k in identifier]
        xedges, yedges = \
            [np.arange(mins[i], maxs[i] + 1.1*mindiffs[i],
                       mindiffs[i]) - .5*mindiffs[i] for i in range(2)]
        X, Y = np.meshgrid(xedges, yedges)
        X_params_flat = (X + .5*mindiffs[0]).flatten()
        Y_params_flat = (Y + .5*mindiffs[1]).flatten()

    # create value array with nans for non-existing data
    # # doesn't work with numpy 1.11. Also, fails if a combination is missing
    # C = np.ones((len(yvec), len(xvec))) * np.nan
    # mask = np.logical_and(np.isin(X + .5*mindiffs[0], sorted_df[identifier[0]]),
    #                       np.isin(Y + .5*mindiffs[1], sorted_df[identifier[1]]))
    # C[mask] = sorted_df['mean_cum_prederr'].as_matrix().reshape(C[mask].shape)

    C = np.ones(X_params_flat.size) * np.nan
    C_std = np.ones(X_params_flat.size) * np.nan
    data_xy = sorted_df.loc[:, identifier].as_matrix()
    for i, xy in enumerate(zip(X_params_flat, Y_params_flat)):
        occurrence = np.where(np.all(np.isclose(xy, data_xy), axis=1))[0]
        if len(occurrence) > 0:
            assert len(occurrence) == 1
            C[i] = sorted_df['mean_cum_prederr'].iloc[occurrence[0]]
            C_std[i] = sorted_df['std_cum_prederr'].iloc[occurrence[0]]

    min_idx = np.nanargmin(C)
    print('Minimum {0} +- {1} at ({4}, {5}) = ({2}, {3})'.format(
        C[min_idx], C_std[min_idx],
        X_params_flat[min_idx], Y_params_flat[min_idx], *identifier))
    C = C.reshape((X.shape[0] - 1, X.shape[1] - 1))
    C_std = C_std.reshape((X.shape[0] - 1, X.shape[1] - 1))

    # save mean
    fig_mean, ax_mean = plt.subplots()
    im = ax_mean.pcolormesh(X, Y, np.ma.masked_where(np.isnan(C), C),
                            cmap=plt.cm.viridis)
    # aspect = (C.shape[0] - 1)/(C.shape[1] - 1) \
    #     * (maxs[0] - mins[0])/(maxs[1] - mins[1])
    # ax_mean.set_aspect(float(aspect))
    if logscale:
        ax_mean.set_xscale('log')
        ax_mean.set_yscale('log')
    plt.xlabel(identifier[0])
    plt.ylabel(identifier[1])
    cbar_m = plt.colorbar(im, ax=ax_mean)
    cbar_m.ax.set_ylabel('Mean cum. prediction error')
    fig_mean.suptitle(title)
    fig_mean.savefig(os.path.join(make_figure_folder(), figname) + '.png')

    # save std
    fig_std, ax_std = plt.subplots()
    im = ax_std.pcolormesh(X, Y, np.ma.masked_where(np.isnan(C_std), C_std),
                           cmap=plt.cm.viridis)
    # ax_std.set_aspect(float(aspect))
    if logscale:
        ax_std.set_xscale('log')
        ax_std.set_yscale('log')
    plt.xlabel(identifier[0])
    plt.ylabel(identifier[1])
    cbar_s = plt.colorbar(im, ax=ax_std)
    cbar_s.ax.set_ylabel('Std. of cum. prediction error')
    fig_std.suptitle(title)
    fig_std.savefig(os.path.join(make_figure_folder(), figname + '_std.png'))


def plot_inferror_pcolor(df, identifier, figname='paramsweep.png', logscale=False,
                         title=''):
    assert len(identifier) == 2
    df['error_rate'] = 1 - df['inf_success'] / df['n_instances']

    sorted_df = df.sort_values(by=identifier)
    # create axis vectors for meshgrid. This code assumes that the differences
    # between x/y-values are all a multiple of some smallest delta
    mins = [sorted_df[k].min() for k in identifier]
    maxs = [sorted_df[k].max() for k in identifier]
    if logscale:
        xcenters, ycenters = [sorted_df[k].unique() for k in identifier]
        # pcolormesh takes grid edges
        tmp_list = []
        for vec in [xcenters, ycenters]:
            tmp = np.hstack((vec, vec[-1]*np.power(vec[-1]/vec[0], 1/len(vec))))
            shift_factor = np.power(tmp.max() / tmp.min(), -1/(2*len(tmp)))
            tmp_list.append(shift_factor*tmp)
        xedges, yedges = tmp_list

        X, Y = np.meshgrid(xedges, yedges)
        tmp_mesh = np.meshgrid(xcenters, ycenters)
        X_params_flat = tmp_mesh[0].flatten()
        Y_params_flat = tmp_mesh[1].flatten()
    else:
        # create axis vectors for meshgrid. This code assumes that the differences
        # between x/y-values are all a multiple of some smallest delta
        mindiffs = [np.diff(sorted_df[k].unique()).min() for k in identifier]
        xvec, yvec = \
            [np.arange(mins[i], maxs[i] + 1.1*mindiffs[i],
                       mindiffs[i]) - .5*mindiffs[i] for i in range(2)]
        X, Y = np.meshgrid(xvec, yvec)
        X_params_flat = (X + .5*mindiffs[0]).flatten()
        Y_params_flat = (Y + .5*mindiffs[1]).flatten()

    # C = np.ones((len(yvec), len(xvec))) * np.nan
    # mask = np.logical_and(np.isin(X + .5*mindiffs[0], sorted_df[identifier[0]]),
    #                       np.isin(Y + .5*mindiffs[1], sorted_df[identifier[1]]))
    # C[mask] = sorted_df['error_rate'].as_matrix().reshape(C[mask].shape)

    C = np.ones(X_params_flat.size) * np.nan
    data_xy = sorted_df.loc[:, identifier].as_matrix()
    for i, xy in enumerate(zip(X_params_flat, Y_params_flat)):
        occurrence = np.where(np.all(np.isclose(xy, data_xy), axis=1))[0]
        if len(occurrence) > 0:
            assert len(occurrence) == 1
            C[i] = sorted_df['error_rate'].iloc[occurrence[0]]

    min_idx = np.nanargmin(C)
    print('Minimum {0} at ({3}, {4}) = ({1}, {2})'.format(
        C[min_idx], X_params_flat[min_idx], Y_params_flat[min_idx], *identifier))
    C = C.reshape((X.shape[0] - 1, X.shape[1] - 1))
    fig, ax = plt.subplots()
    im = ax.pcolormesh(X, Y, np.ma.masked_where(np.isnan(C), C),
                       cmap=plt.cm.viridis, vmin=0, vmax=1)
    # aspect = (C.shape[0] - 1)/(C.shape[1] - 1) \
    #     * (maxs[0] - mins[0])/(maxs[1] - mins[1])
    # ax.set_aspect(float(aspect))
    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    plt.xlabel(identifier[0])
    plt.ylabel(identifier[1])
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Error rate')
    fig.suptitle(title)
    plt.savefig(os.path.join(make_figure_folder(), figname))


def plot_cumerr_1d(df, identifier, fig=None, ax=None, label=None,
                   figname='paramsweep.png', logscale=False):
    assert len(identifier) == 1
    df['mean_cum_prederr'] = df['cum_prederr_sum'] / df['n_instances']
    df['std_cum_prederr'] = (df['cum_prederr_sqsum'] / df['n_instances'] -
                             df['mean_cum_prederr']**2).apply(np.sqrt)

    sorted_df = df.sort_values(by=identifier)

    xdata = sorted_df[identifier[0]].as_matrix()
    ydata = sorted_df['mean_cum_prederr'].as_matrix()
    # take standard deviation or error of mean?
    ystd = sorted_df['std_cum_prederr'].as_matrix()
    yerr = (sorted_df['std_cum_prederr'] /
            df['n_instances'].apply(np.sqrt)).as_matrix()
    print("Minimum {:.2f} at {} = {}".format(ydata.min(), identifier[0],
                                             xdata[np.argmin(ydata)]))
    if fig is None:
        fig, ax = plt.subplots()
        savefig = True
    else:
        savefig = False
    ax.errorbar(xdata, ydata, yerr=yerr, fmt='.:', alpha=.6, label=label)
    # ax.plot(xdata, ydata, '.:', label=label)
    # ax.fill_between(xdata, ydata-ystd, ydata+ystd, alpha=.3)
    if logscale:
        ax.set_xscale('log')
    ax.set_xlabel(identifier[0])
    ax.set_ylabel('Mean cum. prediction error')

    if savefig:
        fig.savefig(os.path.join(make_figure_folder(), figname + '.png'))


def plot_inferror_1d(df, identifier, fig=None, ax=None, label=None,
                     figname='paramsweep.png', logscale=False):
    assert len(identifier) == 1
    df['error_rate'] = 1 - df['inf_success'] / df['n_instances']

    sorted_df = df.sort_values(by=identifier)

    xdata = sorted_df[identifier[0]].as_matrix()
    ydata = sorted_df['error_rate'].as_matrix()
    print("Minimum {} at {} = {}".format(ydata.min(), identifier[0],
                                         xdata[np.argmin(ydata)]))

    if fig is None:
        fig, ax = plt.subplots()
        ax.set(ylim=[0, 1])
        savefig = True
    else:
        savefig = False
    ax.plot(xdata, ydata, '.:', label=label)
    if logscale:
        ax.set_xscale('log')
    ax.set(xlabel=identifier[0], ylabel='Error rate')

    if savefig:
        fig.savefig(os.path.join(make_figure_folder(), figname + '.png'))



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
    plot_cumerr_1d(result, id_params, figname=expt_name + '_cumerr')
    plot_inferror_1d(result, id_params, figname=expt_name + '_inferr')
else:
    response = raw_input(">>> There are more than two identifiers. Plot 1d or 2d?\n")
    if response == '1d':
        n_plotparams = 1
    elif response == '2d':
        n_plotparams = 2
    else:
        print('>>> Wrong input. Aborting...')

    response = raw_input(">>> Plot with log scale? [y/n]\n")
    if response == 'y' or response == 'yes':
        logscale = True
    else:
        logscale = False

    plot_params = []
    while len(plot_params) != n_plotparams:
        if n_plotparams == 2 and len(id_params) == 2:
            plot_params = id_params
        else:
            response = raw_input(
                '>>> Which id-parameters do you want as axes? Options:\n' +
                '\n'.join(['{}: {}'.format(i, param)
                           for i, param in enumerate(id_params)]) + '\n')
            plot_params = [id_params[int(s)] for s in response.split(',')]
            if len(plot_params) != n_plotparams:
                print('>>> Wrong number of parameters.'
                      ' There are {} axes'.format(n_plotparams))

    id_dict = {}
    slice_param = None
    for param in id_params:
        if param in plot_params:
            continue

        if slice_param is None:
            slice_option = '\ns: slice'
        else:
            slice_option = ''

        val_list = result[param].unique()
        response = raw_input(
            '>>> Which value should \"{}\" have? Options:\n'.format(param) +
            '\n'.join(['{}: {}'.format(i, val)
                       for i, val in enumerate(val_list)]) + slice_option + '\n')

        # dialogue for slicing (plot several parameters as different lines)
        if response == 's':
            slice_param = param
            response = raw_input(
                '>>> Which values of \"{}\" should be included? (\"all\" is possible)\n'.format(param))
            if response == 'all':
                slice_vals = val_list
            else:
                slice_vals = [val_list[int(idx)] for idx in response.split(',')]

            # optional labelling
            response = raw_input(
                '>>> Please provide labels for above values. (or type \"skip\")\n')
            if response == 'skip':
                slice_labels = slice_vals
            else:
                slice_labels = response.split(',')
        else:
            id_dict[param] = val_list[int(response)]

    # filter dataframe (except slices)
    if len(id_dict.keys()) > 0:
        id_filter = None
        for k in id_dict.keys():
            if id_filter is None:
                id_filter = result[k] == id_dict[k]
            else:
                id_filter = id_filter & (result[k] == id_dict[k])
        subdf = result.loc[id_filter, :].copy()
        for k in id_dict.keys():
            subdf.pop(k)
    else:
        subdf = result

    if slice_param is None:
        if n_plotparams == 1:
            plot_cumerr_1d(subdf, plot_params, figname=expt_name + '_cumerr', logscale=logscale)
            plot_inferror_1d(subdf, plot_params, figname=expt_name + '_inferr', logscale=logscale)
        if n_plotparams == 2:
            plot_cumerr_pcolor(subdf, plot_params, figname=expt_name + '_cumerr', logscale=logscale)
            plot_inferror_pcolor(subdf, plot_params, figname=expt_name + '_inferr', logscale=logscale)
    else:
        if n_plotparams == 1:
            fig, ax = plt.subplots()
            fig_is, ax_is = plt.subplots()
            ax_is.set(ylim=[0, 1])
            # iterate over slices
            for i, v in enumerate(slice_vals):
                    slicedf = subdf.loc[subdf[slice_param] == v, :].copy()
                    slicedf.pop(slice_param)
                    label = slice_labels[i]
                    plot_cumerr_1d(slicedf, plot_params, fig, ax, label=label, logscale=logscale)
                    plot_inferror_1d(slicedf, plot_params, fig_is, ax_is, label=label, logscale=logscale)
            ax.legend()
            fig.savefig(os.path.join(make_figure_folder(), expt_name +
                                     '_slice_{}_cumerr.png'.format(slice_param)))
            ax_is.legend()
            fig_is.savefig(os.path.join(make_figure_folder(), expt_name +
                                        '_slice_{}_inferr.png'.format(slice_param)))
        if n_plotparams == 2:
            for i, v in enumerate(slice_vals):
                    slicedf = subdf.loc[subdf[slice_param] == v, :].copy()
                    slicedf.pop(slice_param)
                    print(slicedf.columns)
                    label = slice_labels[i]
                    plot_cumerr_pcolor(slicedf, plot_params,
                                       figname=expt_name + '_slice_{}{:02d}_cumerr'.format(slice_param, i),
                                       logscale=logscale, title=label)
                    plot_inferror_pcolor(slicedf, plot_params,
                                         figname=expt_name + '_slice_{}{:02d}_inferr'.format(slice_param, i),
                                         logscale=logscale, title=label)            

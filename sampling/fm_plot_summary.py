from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import sys
import os
import yaml
import matplotlib.pyplot as plt
from lif_pong.utils.data_mgmt import make_figure_folder


# if there are two identifiers, produce heatmap
def plot_infsuccess(df, identifier, figname='paramsweep.png'):
    assert len(identifier) == 2
    df['success_rate'] = df['inf_success'] / df['n_instances']

    sorted_df = df.sort_values(by=identifier)
    gridshape = tuple([len(sorted_df[k].unique()) for k in identifier])

    ygrid = np.reshape(sorted_df[identifier[0]].as_matrix(), gridshape)
    xgrid = np.reshape(sorted_df[identifier[1]].as_matrix(), gridshape)
    dx = np.diff(xgrid, axis=1)[0, 0]
    dy = np.diff(ygrid, axis=0)[0, 0]
    xmin = xgrid.min() - .5*dx
    xmax = xgrid.max() + .5*dx
    ymin = ygrid.min() - .5*dy
    ymax = ygrid.max() + .5*dy
    xyratio = float((xmax - xmin) / (ymax - ymin))
    z = np.reshape(sorted_df['success_rate'].as_matrix(), gridshape)

    print('Max: {}, min: {}'.format(np.nanmax(z), np.nanmin(z)))
    plt.figure()
    plt.imshow(z, cmap=plt.cm.viridis, interpolation='nearest', origin='lower',
               extent=[xmin, xmax, ymin, ymax], vmin=0, vmax=1,
               aspect=xyratio)
    plt.xlabel(identifier[1])
    plt.ylabel(identifier[0])
    plt.colorbar()

    plt.savefig(os.path.join(make_figure_folder(), figname))


def plot_infsuccess_pcolor(df, identifier, figname='paramsweep.png'):
    assert len(identifier) == 2
    df['success_rate'] = df['inf_success'] / df['n_instances']

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
    # C[mask] = sorted_df['success_rate'].as_matrix().reshape(C[mask].shape)

    C = np.ones(len(yvec) * len(xvec)) * np.nan
    data_xy = sorted_df.loc[:, identifier].as_matrix()
    X_params_flat = (X + .5*mindiffs[0]).flatten()
    Y_params_flat = (Y + .5*mindiffs[1]).flatten()
    for i, xy in enumerate(zip(X_params_flat, Y_params_flat)):
        occurrence = np.where(np.all(np.isclose(xy, data_xy), axis=1))[0]
        if len(occurrence) > 0:
            assert len(occurrence) == 1
            C[i] = sorted_df['success_rate'].iloc[occurrence[0]]

    max_idx = np.nanargmax(C)
    print('Maximum {0} at ({3}, {4}) = ({1}, {2})'.format(
        C[max_idx], X_params_flat[max_idx], Y_params_flat[max_idx], *identifier))
    C = C.reshape(len(yvec), len(xvec))
    fig, ax = plt.subplots()
    im = ax.pcolormesh(X, Y, np.ma.masked_where(np.isnan(C), C),
                       cmap=plt.cm.viridis, vmin=0, vmax=1)
    aspect = (C.shape[0] - 1)/(C.shape[1] - 1) \
        * (maxs[0] - mins[0])/(maxs[1] - mins[1])
    ax.set_aspect(float(aspect))
    plt.xlabel(identifier[0])
    plt.ylabel(identifier[1])
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Success rate')
    plt.savefig(os.path.join(make_figure_folder(), figname))


def plot_infsuccess_1d(df, identifier, figname='paramsweep.png'):
    assert len(identifier) == 1
    df['success_rate'] = df['inf_success'] / df['n_instances']

    sorted_df = df.sort_values(by=identifier)

    xdata = sorted_df[identifier[0]].as_matrix()
    ydata = sorted_df['success_rate'].as_matrix()
    print("Maximum {} at {} = {}".format(ydata.max(), identifier[0],
                                         xdata[np.argmax(ydata)]))
    plt.plot(xdata, ydata, '.:')
    plt.xlabel(identifier[0])
    plt.ylabel('Success rate')
    plt.savefig(os.path.join(make_figure_folder(), figname))


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
        # plot_trec_U(subdf, figname=expt_name + '_w={}.png'.format(weight))
        plot_infsuccess_pcolor(subdf, id_params, figname=expt_name + '_w={}.png'.format(weight))
else:
    print('Don\'t know what to do with the data')

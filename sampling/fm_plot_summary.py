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

    sorted_df = df.sort_values(by=identifier[::-1])
    gridshape = tuple([len(sorted_df[k].unique()) for k in identifier])

    xgrid = np.reshape(sorted_df[identifier[0]].as_matrix(), gridshape)
    ygrid = np.reshape(sorted_df[identifier[1]].as_matrix(), gridshape)
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
    plt.xlabel(identifier[0])
    plt.ylabel(identifier[1])
    plt.colorbar()
    # alternatively take pcolormesh, but there is some difference about x/y
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
if len(id_params) == 2:
    plot_infsuccess(result, id_params, figname=expt_name)
elif 'weight' in result.columns:
    id_params.remove('weight')
    for weight in result['weight'].unique():
        subdf = result.loc[result.weight == weight, :].copy()
        subdf.pop('weight')
        # plot_trec_U(subdf, figname=expt_name + '_w={}.png'.format(weight))
        plot_infsuccess(subdf, id_params, figname=expt_name + '_w={}.png'.format(weight))
else:
    print('Don\'t know what to do with the data')

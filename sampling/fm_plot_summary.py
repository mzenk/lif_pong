from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import sys
import os
import yaml
import matplotlib.pyplot as plt


def plot_trec_U(df, figname='paramsweep.png'):
    df['success_rate'] = df['inf_success'] / df['n_instances']

    sorted_df = df.sort_values(by=['U', 'tau_rec'])
    n_U = len(sorted_df['U'].unique())
    n_tau_rec = len(sorted_df['tau_rec'].unique())

    tgrid = np.reshape(sorted_df['tau_rec'].as_matrix(), (n_tau_rec, n_U))
    ugrid = np.reshape(sorted_df['U'].as_matrix(), (n_tau_rec, n_U))
    xmin = tgrid.min() - .5*np.diff(tgrid, axis=1)[0, 0]
    xmax = tgrid.max() + .5*np.diff(tgrid, axis=1)[0, 0]
    ymin = ugrid.min() - .5*np.diff(ugrid, axis=0)[0, 0]
    ymax = ugrid.max() + .5*np.diff(ugrid, axis=0)[0, 0]
    z = np.reshape(sorted_df['success_rate'].as_matrix(), (n_tau_rec, n_U))

    plt.figure()
    plt.imshow(z, cmap=plt.cm.viridis, interpolation='nearest', origin='lower',
               extent=[xmin, xmax, ymin, ymax],
               aspect='auto')
    plt.xlabel('tau_rec')
    plt.xlabel('U')
    plt.colorbar()
    # alternatively take pcolormesh, but there is some difference about x/y
    plt.savefig(figname)

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
# load data from "collect" folder into data frame
with open(os.path.join(collectfolder, expt_name)) as f:
    df = pd.DataFrame(yaml.load(f))

# average over chunks
if 'identifier' in simdict.keys():
    params = simdict['identifier'].keys()
else:
    params = []
    for k in simdict['replacements'].keys():
        if k in df.columns:
            params.append(k)
    params.remove('start_idx')
print('Group dataframe by: ' + ', '.join('{}'.format(p) for p in params))
grouped = df.groupby(params, as_index=False)
result = grouped.aggregate(np.sum)
result.pop('start_idx')
result.pop('tau_fac')

# plot heatmap of success rate
if 'weight' in result.columns:
    for weight in result['weight'].unique():
        subdf = result.loc[result.weight == weight, :].copy()
        subdf.pop('weight')
        plot_trec_U(subdf, figname='paramsweep_w={}.png'.format(weight))
else:
    plot_trec_U(result)

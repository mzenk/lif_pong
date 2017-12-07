from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import sys
import os
import yaml
import matplotlib.pyplot as plt


def plot_trec_U(df, weight=0.):
    df = df[df.weight == weight]
    df.pop('weight')
    df['inf_success'] /= df['n_instances']
    df['inf_std'] /= df['n_instances']

    sorted_df = df.sort_values(by=['tau_rec', 'U'])
    n_U = len(sorted_df['U'].unique())
    n_tau_rec = len(sorted_df['tau_rec'].unique())

    tgrid = np.reshape(sorted_df['tau_rec'].as_matrix(), (n_U, n_tau_rec))
    ugrid = np.reshape(sorted_df['U'].as_matrix(), (n_U, n_tau_rec))
    xmin = tgrid.min() - .5*np.diff(tgrid, axis=0)[0, 0]
    xmax = tgrid.max() + .5*np.diff(tgrid, axis=0)[0, 0]
    ymin = ugrid.min() - .5*np.diff(ugrid, axis=1)[0, 0]
    ymax = ugrid.max() + .5*np.diff(ugrid, axis=1)[0, 0]
    z = np.reshape(sorted_df['inf_success'].as_matrix(), (n_U, n_tau_rec))
    z_std = np.reshape(sorted_df['inf_std'].as_matrix(), (n_U, n_tau_rec))

    plt.imshow(z, cmap='cool', interpolation='nearest', origin='lower',
               extent=[xmin, xmax, ymin, ymax],
               aspect='auto')
    # alternatively take pcolormesh, but there is some difference about x/y
    plt.savefig('testimshow.png')
    print(sorted_df)

if len(sys.argv) != 2:
    print('Wrong number of arguments. Please provide the experiment name')
    sys.exit()
expt_name = sys.argv[1]

collectfolder = os.path.expanduser('~/mnt/bwnemo_mnt/workspace/experiment/collect')
simfolder = os.path.expanduser('~/mnt/bwnemo_mnt/workspace/experiment/simulations')
# load simulation config file
with open(os.path.join(simfolder, '01_runs', expt_name), 'r') as f:
    simdict = yaml.load(f)
# load data from "collect" folder into data frame
with open(os.path.join(collectfolder, expt_name), 'r') as f:
    df = pd.DataFrame(yaml.load(f))

# average over chunks
print(simdict['replacements'].keys())
params = simdict['replacements'].keys()
params.remove('start_idx')
grouped = df.groupby(params, as_index=False)
print(df)
result = grouped.aggregate(np.sum)
result.pop('start_idx')
result.pop('tau_fac')

# plot heatmap of success and its std
plot_trec_U(result)

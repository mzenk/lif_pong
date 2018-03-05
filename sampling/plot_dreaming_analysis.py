from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import sys
import os
import yaml
from lif_pong.utils.data_mgmt import make_figure_folder
import matplotlib.pyplot as plt


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

# make histogram of threshold crossings and stationary values
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
df.hist(column='stat_activity', ax=ax1, bins='auto', grid=False)
df.hist(column='thresh_crossing', ax=ax2, bins=100, grid=False)
fig.savefig(os.path.join(make_figure_folder(), expt_name + '_histo.png'))

quantiles = df.quantile([.25, .5, .75])
print('1st quartile/median/3rd quartile of stat. value: {:.3f} / {:.3f} / {:.3f}'.format(
    quantiles.loc[.25, 'stat_activity'], quantiles.loc[.5, 'stat_activity'],
    quantiles.loc[.75, 'stat_activity']))
print('1st quartile/median/3rd quartile of threshold: {} / {} / {}'.format(
    quantiles.loc[.25, 'thresh_crossing'], quantiles.loc[.5, 'thresh_crossing'],
    quantiles.loc[.75, 'thresh_crossing']))
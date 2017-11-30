#! /usr/bin/env python
import sys
import yaml
import os
'''
what to do here?
- iterate through experiment result folders and compute for each the quantity
  to optimize (Q), i.e. agent performance for inf. agent speed
  -> need script that takes a list of folders, searches the samples.npz in them
     and computes Q (+ error estimate?) from it  [can be used for other expts]
  -> send slurm jobs or equivalent on BW-cluster
- depending on if I run experiment for chunks of data (might be faster if many
  jobs can run in parallel on cluster) I'll have to combine the data of some of
  the folders first
- wait for jobs to finish, then save results and plot
'''
if len(sys.argv) != 2:
    print('Wrong number of arguments. Please provide the experiment name')
    sys.exit()
expt_name = sys.argv[1]

simfolder = '/wang/users/mzenk/cluster_home/experiment/simulations/'
worker_script = '/wang/users/mzenk/cluster_home/Projects/Pong/sampling/lif_analysefm_worker.py'
stub = """
#!/usr/bin/env bash

cd "{folder}" &&
{script} "{yamlfile} {u_idx} {tau_rec_idx}"
"""
# load yaml-config of experiment
config_file = simfolder + '01_runs/' + expt_name
with open(config_file) as config:
    experiment_dict = yaml.load(config)

stub_dict = experiment_dict.pop('stub')
chunksize = stub_dict['general']['chunksize']
replacements = experiment_dict.pop('replacements')
us = replacements['U']
tau_recs = replacements['tau_rec']

analysis_folder = simfolder + expt_name + '/analysis/'
if not os.path.exists(analysis_folder):
    os.makedirs(analysis_folder)
for i, u in enumerate(us):
    for j, tau_rec in enumerate(tau_recs):
        # submit batch jobs of worker scripts
        stub.format(folder=simfolder + expt_name, script=worker_script,
                    yamlfile=config_file, u_idx=i, tau_rec_idx=j)
        # slurm
        # ...
        # bwnemo
        # ...

# plot results (depending on how costly send to cluster)
# ...

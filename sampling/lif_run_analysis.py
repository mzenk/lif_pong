#! /usr/bin/env python
import sys
import yaml
import os
import subprocess
import time

if len(sys.argv) != 2:
    print('Wrong number of arguments. Please provide the experiment name')
    sys.exit()
expt_name = sys.argv[1]

simfolder = '/work/ws/nemo/hd_kq433-data_workspace-0/experiment/simulations'
worker_script = '/home/hd/hd_hd/hd_kq433/git_repos/lif_pong/sampling/lif_fading_memory_analysis.py'

stub = """#!/usr/bin/env bash

#MSUB -l nodes=1:ppn=1
#MSUB -l walltime=00:10:00
#MSUB -l pmem=10000mb
#MSUB -N analysis

source {envscript} &&
cd "{folder}" &&
python {script}
"""
# load yaml-config of experiment
config_file = os.path.join(simfolder, '01_runs', expt_name)
with open(config_file) as config:
    experiment_dict = yaml.load(config)

# for each folder in simulation, compute prediction etc. in worker script
taskfiles = []
for folder in os.listdir(os.path.join(simfolder, expt_name)):
      curr_path = os.path.join(simfolder, expt_name, folder)
      taskfiles.append(os.path.join(curr_path, 'analysis_job'))
      # write job scripts
      with open(taskfiles[-1], 'w') as f:
          content = stub.format(folder=curr_path,
                                script=worker_script,
                                envscript=experiment_dict['envscript'])
          f.write(content)
time.sleep(.1)

for job in taskfiles:
    # submit batch jobs of worker scripts
    jobid = subprocess.check_output(['msub', job])

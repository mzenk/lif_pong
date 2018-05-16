#!/usr/bin/env bash
#MSUB -l nodes=1:ppn=2
#MSUB -l walltime=00:59:00
#MSUB -l pmem=5000mb
#MSUB -N wrap
. /home/hd/hd_hd/hd_kq433/setup-env.sh

cd /home/hd/hd_hd/hd_kq433/git_repos/lif_pong/sampling
# python lif_calibration.py
# python plot_predictions.py comparison_yamls/collection.yaml
python lif_inspect_samples.py sample_inspection.yaml
# python general_animation.py make_video.yaml

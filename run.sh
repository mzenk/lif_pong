#! /bin/bash
#SBATCH --nodes=1
sbatch -p simulation --wrap="python $1 $2"

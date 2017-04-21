#! /bin/bash
sbatch -p simulation --wrap="python $1 $2"
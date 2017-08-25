#!/bin/bash
#SBATCH --nodes=1
##SBATCH --nodelist=HBPHost14,HBPHost15,HBPHost16,HBPHost17
# vary windows and samples
chunk_size=5
starts=($(seq 0 $chunk_size 4))
echo "Send lif sampling jobs to slurm"
for start in "${starts[@]}"; do
		sbatch -p simulation --wrap="python lif_sampling.py $start $chunk_size" 
done

# #!/bin/bash
# #SBATCH --nodes=1
# ##SBATCH --mem-per-node=16000
# ##SBATCH --nodelist=HBPHost14,HBPHost15,HBPHost16,HBPHost17
# chunk_size=
# starts=($(seq 0 $chunk_size 299))
# echo "Send gibbs sampling jobs to slurm"
# for start in "${starts[@]}"; do
# 	sbatch -p simulation --wrap="python gibbs_sampling.py $start $chunk_size"
# done

# sbatch -p simulation --dependency=singleton --wrap="python save_prediction_data.py"

# #!/bin/bash
# #SBATCH --nodes=1
# ##SBATCH --nodelist=HBPHost14,HBPHost15,HBPHost16,HBPHost17
# # vary windows and samples
# chunk_size=5
# starts=($(seq 0 $chunk_size 4))
# echo "Send lif sampling jobs to slurm"
# for start in "${starts[@]}"; do
#         sbatch -p simulation --wrap="python lif_sampling.py $start $chunk_size" 
# done

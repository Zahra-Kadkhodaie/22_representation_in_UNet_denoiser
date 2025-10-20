#!/bin/bash
#SBATCH --job-name=phi_imagenet
#SBATCH -p gpu
#SBATCH -t 72:00:00
#SBATCH -C a100-80gb|h100
##SBATCH -C h100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256GB
##SBATCH --output=slurm_%j.out
##SBATCH --error=slurm_%j.err
#SBATCH --mail-user=zk388@nyu.edu
#SBATCH --mail-type=END,FAIL
##SBATCH --array=1



module --force purge

#python3 generate_c_alpha_dataset.py --alpha=${SLURM_ARRAY_TASK_ID}
#python3 generate_samples_unconditional.py 
#python3 interpolate_mixture.py 
#python3 metamers.py
python3 compute_phis.py
#python3 cluster_phis.py 

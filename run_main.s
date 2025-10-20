#!/bin/bash
#SBATCH --job-name=fine2
#SBATCH -p gpu
#SBATCH -t 165:00:00
##SBATCH -C h100
#SBATCH -C a100-80gb|h100
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=512GB
##SBATCH --output=slurm_%j.out
##SBATCH --error=slurm_%j.err
#SBATCH --mail-user=zk388@nyu.edu
#SBATCH --mail-type=END,FAIL
##SBATCH --array=1,10,100





module --force purge
#module load cuda/11

#module load python/intel/3.8.6
#source /home/zk388/bfcnn_env/bin/activate

cd code

python3 fine_tune_main.py
#python3 main.py --SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}
#python3 phi_classification.py





#!/bin/bash
#SBATCH -A KORHONEN-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --gpus-per-node=1

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment

# Source .bashrc to ensure conda is initialized
source ~/.bashrc

echo "===============Activating conda environment=================="
conda activate lmp
echo ">Python version: $(which python)"
echo "===============Running script=================="

cd /home/yz926/lm-polygraph

HYDRA_CONFIG=/home/yz926/lm-polygraph/examples/configs/TEST_COQA.yaml  python ./scripts/polygraph_eval       save_path="./workdir/output"  

#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH -w ltl-gpu05
#SBATCH -G1

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
# . /etc/profile.d/modules.sh                # Leave this line (enables the module command)
# module purge                               # Removes all modules still loaded
# module load rhel8/default-amp              # REQUIRED - loads the basic environment

conda init
# Source .bashrc to ensure conda is initialized
source ~/.bashrc
conda init

echo "===============Activating conda environment=================="
conda activate mu
echo ">Python version: $(which python)"
echo "===============Running script=================="

# cd /home/yz926/MUncertainty/lm-poly-test
cd ..

HYDRA_CONFIG=/mnt/nas_home/yz926/lm-polygraph/examples/configs/TEST_COQA.yaml  python ./scripts/polygraph_eval       save_path="./workdir/output"  
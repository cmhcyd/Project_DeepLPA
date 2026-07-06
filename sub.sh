#!/bin/bash

######## Part 1 #########
# Script parameters     #
#########################

#SBATCH --partition=placeholder
#SBATCH --account=placeholder
#SBATCH --qos=placeholder         
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:placeholder:1
#SBATCH --job-name=placeholder
#SBATCH --comment="placeholder"
#SBATCH --output=logs/test.log

######## Part 2 ######
# Script workload    #
######################

# list the allocated hosts
srun -l hostname

# list the GPU cards of the host
nvidia-smi

sh fullrec_example.sh
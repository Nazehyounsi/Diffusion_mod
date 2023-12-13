#!/bin/bash


#SBATCH --job-name=Config2

#SBATCH --nodes=1

#SBATCH --gpus-per-node=2

#SBATCH --time=1

#SBATCH --output=%x-%j.out

#SBATCH --error=%x-%j.err

python Training.py --config config2.json --train --gpu

#!/bin/bash


#SBATCH --job-name=Config1

#SBATCH --nodes=1

#SBATCH --gpus-per-node=2

#SBATCH --time=600

#SBATCH --output=%x-%j.out

#SBATCH --error=%x-%j.err

python .\Training.py --config config1.json --train --gpu

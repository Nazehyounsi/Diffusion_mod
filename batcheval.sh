#!/bin/bash


#SBATCH --job-name=Eval13

#SBATCH --nodes=1

#SBATCH --gpus-per-node=2

#SBATCH --time=2500

#SBATCH --output=%x-%j.out

#SBATCH --error=%x-%j.err

python Training.py --config config13.json --evaluate --gpu

#!/bin/bash

#SBATCH --partition=Nom_de_la_partition1

#SBATCH --job-name=Config2

#SBATCH --nodes=1

#SBATCH --gpus-per-node=2

#SBATCH --time=600

#SBATCH –mail-type=ALL

#SBATCH –mail-user=Nezih.younsi@sorbonne.universite.fr

#SBATCH --output=%x-%j.out

#SBATCH --error=%x-%j.err

python .\Training.py --config config2.json --train --gpu

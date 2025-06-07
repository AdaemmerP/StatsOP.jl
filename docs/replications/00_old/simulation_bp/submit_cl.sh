#!/bin/bash
#SBATCH -p dev
#SBATCH -N 1
#SBATCH --ntasks-per-node=72


julia 07_simulation_ooc_sar1.jl

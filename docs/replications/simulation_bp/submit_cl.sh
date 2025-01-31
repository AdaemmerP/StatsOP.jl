#!/bin/bash
#SBATCH -p snowball
#SBATCH -N 20
#SBATCH --ntasks-per-node=72


julia simulation_cls.jl

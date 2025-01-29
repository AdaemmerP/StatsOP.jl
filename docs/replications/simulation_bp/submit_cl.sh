#!/bin/bash
#SBATCH -p snowball
#SBATCH -N 10
#SBATCH --ntasks-per-node=72
#SBATCH -t 00:30:00


julia simulation_cls.jl

#!/bin/bash
#SBATCH -J RANS_PINN
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --partition=3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH -o RANS_job_%j.o
#SBATCH -e RANS_job_%j.e

module load hpcx-mpi/4.1.5rc2s-yflad4v cuda/12.4.0-piq32fy cmake/3.26.3-xi6h36u
mpirun -np 1 nekrs --setup squareCyl.par

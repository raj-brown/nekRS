#!/bin/bash
#SBATCH -J NEKRS_LES
#SBATCH -N 1
#SBATCH -A gk-h100-gcondo
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=144:00:00
#SBATCH --mem=64GB
#SBATCH --partition=h100-gcondo
#SBATCH --gres=gpu:1
#SBATCH -o RANS_job_%j.o
#SBATCH -e RANS_job_%j.e



export NEKRS_HOME=/oscar/data/gk/kshukla1/.local/nekrs
export PATH=$NEKRS_HOME/bin:$PATH

module load hpcx-mpi/2.25.1s-le4f
module load cuda/12.9.0-cinr

srun --mpi=pmix nekrs --setup squareCyl.par
